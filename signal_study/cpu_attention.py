"""Diagnostic-only FP64 CPU attention; never a production/speed fallback."""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .common import write_json
from .reference_audit import explicit_attention
from .reference_precision import POLICY as FP64_POLICY


POLICY = dict(FP64_POLICY, id='target-fp64-cpu-attention-reference-v2',
              sdpa='Target: explicit FP64 CPU QK/softmax/PV; Drafter: unchanged math SDPA',
              attention_check='Every Target call: last query checked against CPU math SDPA at 1e-9',
              note='Diagnostic reference only; CPU transfers and computation are NOT speed measurements.')
CAP = 1e-9


def error(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return None
    return float((a - b).abs().max())


def scores_for(q, k, mask, causal, scale):
    scores = (q @ k.transpose(-1, -2)) * scale
    if causal:
        allowed = torch.arange(k.shape[-2], device=q.device)[None, :] <= \
                  torch.arange(q.shape[-2], device=q.device)[:, None]
        scores.masked_fill_(~allowed, -math.inf)
    elif mask is not None:
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, -math.inf)
        else:
            scores += mask
    return scores


@torch.inference_mode()
def primitive_case(q, k, v, mask=None, *, causal=False, scale=None, native=None):
    """Isolate each GPU primitive using identical CPU inputs for that stage."""
    native = native or F.scaled_dot_product_attention
    scale = q.shape[-1] ** -.5 if scale is None else scale
    qc, kc, vc = (t.detach().cpu() for t in (q, k, v))
    mc = None if mask is None else mask.detach().cpu()
    reference = explicit_attention(qc, kc, vc, mc, is_causal=causal, scale=scale)
    with sdpa_kernel(SDPBackend.MATH):
        cpu_sdpa = native(qc, kc, vc, attn_mask=mc, is_causal=causal, scale=scale)
        gpu_sdpa = native(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)
        gpu_repeat = native(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)
    result = dict(query_shape=list(q.shape), key_shape=list(k.shape), causal=causal,
                  cpu_crosscheck_error=error(reference, cpu_sdpa),
                  sdpa_error=error(gpu_sdpa, reference), sdpa_repeat_error=error(gpu_sdpa, gpu_repeat))
    result['explicit_gpu_error'] = error(explicit_attention(q, k, v, mask, is_causal=causal,
                                                           scale=scale), reference)
    # QK comparison is before masking, so infinities never enter the error metric.
    raw = qc @ kc.transpose(-1, -2)
    result['qk_error'] = error(q @ k.transpose(-1, -2), raw)
    del raw
    scores = scores_for(qc, kc, mc, causal, scale)
    empty = torch.isneginf(scores).all(-1, keepdim=True)
    scores = scores.masked_fill(empty, 0)
    probabilities = scores.softmax(-1).masked_fill(empty, 0)
    common = scores.to(q.device)
    gpu_probs = common.softmax(-1).masked_fill(empty.to(q.device), 0)
    result['softmax_same_scores_error'] = error(gpu_probs, probabilities)
    stable = (common - common.amax(-1, keepdim=True)).exp()
    stable = (stable / stable.sum(-1, keepdim=True)).masked_fill(empty.to(q.device), 0)
    result['exp_sum_same_scores_error'] = error(stable, probabilities)
    result['pv_same_probabilities_error'] = error(probabilities.to(q.device) @ v, probabilities @ vc)
    result['cpu_validated'] = result['cpu_crosscheck_error'] is not None and result['cpu_crosscheck_error'] <= CAP
    return result


@torch.inference_mode()
def primitive_probe(path, *, device='cuda', lengths=(471, 512, 513, 592, 656, 719, 769), heads=32, dim=128):
    report = dict(scope='model_free_primitive_diagnostic', status='running', cap=CAP,
                  torch=torch.__version__, cuda=torch.version.cuda, cases=[])
    write_json(path, report)
    generator = torch.Generator(device='cpu').manual_seed(90218)
    for length in lengths:
        q, k, v = (torch.randn(1, heads, length, dim, dtype=torch.float64,
                              generator=generator).to(device) for _ in range(3))
        for causal in (True, False):
            mask = None
            if not causal:
                mask = torch.ones(1, 1, length, length, dtype=torch.bool, device=device).tril()
                mask[..., 2::7] = False
            item = primitive_case(q, k, v, mask, causal=causal)
            report['cases'].append(item)
            write_json(path, report)
            print(f"Primitive length={length} causal={causal}: {item}", flush=True)
    report['status'] = 'complete' if all(i['cpu_validated'] for i in report['cases']) else 'failed'
    report['note'] = 'GPU disagreement is evidence, not a gate pass. Only the CPU crosscheck gates the CPU reference.'
    write_json(path, report)
    if report['status'] != 'complete':
        raise ValueError('CPU attention reference failed its independent synthetic crosscheck')
    return report


@contextmanager
def cpu_attention(report):
    """Scoped inference wrapper: copy Q/K/V, compute CPU reference, return to original device."""
    original = F.scaled_dot_product_attention
    report.update(calls=0, last_query_checks=0, max_cpu_crosscheck_error=0.0,
                  cap=CAP, actual_input_probes=[], output_policy='FP64 CPU explicit attention')
    sampled_shapes = set()

    def forward(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        if q.dtype != torch.float64:
            kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            if enable_gqa:
                kwargs['enable_gqa'] = True
            return original(q, k, v, **kwargs)
        if dropout_p != 0 or enable_gqa:
            raise ValueError('CPU reference requires inference and expanded GQA keys')
        qc, kc, vc = (t.detach().cpu() for t in (q, k, v))
        mask = None if attn_mask is None else attn_mask.detach().cpu()
        with torch.autocast('cpu', enabled=False):
            output = explicit_attention(qc, kc, vc, mask, is_causal=is_causal, scale=scale)
            if not bool(torch.isfinite(output).all()):
                raise ValueError('nonfinite CPU attention output')
            last_mask = None if mask is None else mask[..., -1:, :]
            if is_causal:
                last_mask = torch.arange(k.shape[-2])[None, :] <= q.shape[-2] - 1
            with sdpa_kernel(SDPBackend.MATH):
                check = original(qc[..., -1:, :], kc, vc, attn_mask=last_mask, scale=scale)
            delta = error(output[..., -1:, :], check)
        report['calls'] += 1
        report['last_query_checks'] += 1
        if delta is None or delta > CAP:
            report['failed_cpu_crosscheck_error'] = delta
            raise ValueError(f'CPU attention crosscheck failed: {delta}')
        report['max_cpu_crosscheck_error'] = max(report['max_cpu_crosscheck_error'], delta)
        # Real model inputs help distinguish synthetic non-reproduction from a data-dependent issue.
        shape = (q.shape[-2], k.shape[-2])
        if q.is_cuda and q.shape[-2] >= 512 and shape not in sampled_shapes and len(sampled_shapes) < 2:
            sampled_shapes.add(shape)
            probe = primitive_case(q, k, v, attn_mask, causal=is_causal, scale=scale, native=original)
            report['actual_input_probes'].append(probe)
            print(f'Actual QKV primitive probe: {probe}', flush=True)
            if not probe['cpu_validated']:
                raise ValueError('CPU full-query attention crosscheck failed on actual model inputs')
        return output.to(q.device)

    F.scaled_dot_product_attention = forward
    try:
        yield report
    finally:
        F.scaled_dot_product_attention = original

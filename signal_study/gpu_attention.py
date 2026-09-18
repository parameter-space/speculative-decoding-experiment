"""FP64 diagnostic attention avoiding the GPU Tensor.softmax path.

Derived from Job 426051's same-input primitive comparisons. This is not a
change to the BF16 production Target or the Drafter, and not a speed benchmark.
"""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from .common import write_json
from .cpu_attention import CAP, error
from .reference_audit import explicit_attention
from .reference_precision import POLICY as FP64_POLICY


POLICY = dict(FP64_POLICY, id='target-fp64-exp-sum-attention-reference-v3',
              sdpa='Target: GPU FP64 QK, max/exp/sum normalization, PV; Drafter: unchanged math SDPA',
              attention_check='Every Target last query vs CPU FP64 at 1e-9; first two long shapes checked in full',
              note='Separate numerical reference, NOT original BF16 S1 or a speed benchmark.')


def exp_sum_attention(q, k, v, mask=None, *, is_causal=False, scale=None, chunk=32):
    if any(t.dtype != torch.float64 for t in (q, k, v)):
        raise ValueError('exp/sum reference requires FP64 Q/K/V')
    if chunk <= 0 or (is_causal and mask is not None):
        raise ValueError('invalid chunk or simultaneous causal/explicit mask')
    scale = q.shape[-1] ** -.5 if scale is None else scale
    result = torch.empty((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=q.dtype)
    for start in range(0, q.shape[-2], chunk):
        stop = min(start + chunk, q.shape[-2])
        scores = (q[..., start:stop, :] @ k.transpose(-1, -2)) * scale
        if is_causal:
            allowed = torch.arange(k.shape[-2], device=q.device)[None, :] <= \
                      torch.arange(start, stop, device=q.device)[:, None]
            scores.masked_fill_(~allowed, -math.inf)
        elif mask is not None:
            part = mask if mask.shape[-2] == 1 else mask[..., start:stop, :]
            if part.dtype == torch.bool:
                scores.masked_fill_(~part, -math.inf)
            else:
                scores += part
        empty = torch.isneginf(scores).all(-1, keepdim=True)
        scores.masked_fill_(empty, 0)
        # Do not use Tensor.softmax, torch.softmax, or SDPA in this FP64 path.
        probabilities = (scores - scores.amax(-1, keepdim=True)).exp()
        probabilities = (probabilities / probabilities.sum(-1, keepdim=True)).masked_fill(empty, 0)
        result[..., start:stop, :] = probabilities @ v
    return result


@contextmanager
def gpu_attention(report):
    original = F.scaled_dot_product_attention
    report.update(calls=0, last_query_checks=0, full_checks=[], max_cpu_error=0.0, cap=CAP,
                  output_policy='GPU FP64 exp/sum attention; CPU outputs are checks only')
    shapes = set()

    def forward(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        if q.dtype != torch.float64:
            kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            if enable_gqa:
                kwargs['enable_gqa'] = True
            return original(q, k, v, **kwargs)
        if dropout_p != 0 or enable_gqa:
            raise ValueError('GPU reference requires inference and expanded GQA keys')
        with torch.autocast(q.device.type, enabled=False):
            actual = exp_sum_attention(q, k, v, attn_mask, is_causal=is_causal, scale=scale)
        if not bool(torch.isfinite(actual).all()):
            raise ValueError('nonfinite exp/sum attention output')
        kc, vc = k.detach().cpu(), v.detach().cpu()
        mask = None if attn_mask is None else attn_mask.detach().cpu()
        last_mask = None if mask is None else mask[..., -1:, :]
        if is_causal:
            last_mask = torch.arange(k.shape[-2])[None, :] <= q.shape[-2] - 1
        with torch.autocast('cpu', enabled=False):
            expected = explicit_attention(q[..., -1:, :].detach().cpu(), kc, vc, last_mask, scale=scale)
            delta = error(actual[..., -1:, :], expected)
            report['calls'] += 1
            report['last_query_checks'] += 1
            if delta is None or delta > CAP:
                report['failed_error'] = delta
                raise ValueError(f'GPU exp/sum vs CPU attention failed: {delta}')
            report['max_cpu_error'] = max(report['max_cpu_error'], delta)
            shape = (q.shape[-2], k.shape[-2])
            if q.shape[-2] >= 512 and shape not in shapes and len(shapes) < 2:
                shapes.add(shape)
                expected = explicit_attention(q.detach().cpu(), kc, vc, mask, is_causal=is_causal, scale=scale)
                delta = error(actual, expected)
                report['full_checks'].append(dict(query_tokens=shape[0], key_tokens=shape[1], error=delta))
                if delta is None or delta > CAP:
                    raise ValueError(f'GPU exp/sum full attention vs CPU failed: {delta}')
        return actual

    F.scaled_dot_product_attention = forward
    try:
        yield report
    finally:
        F.scaled_dot_product_attention = original


@torch.inference_mode()
def probe_gpu_reference(path, *, device='cuda', lengths=(471, 512, 513, 592, 656, 719, 769, 770),
                        heads=32, dim=128):
    """Validate the exact replacement, not only its normalization on CPU scores."""
    report = dict(policy=POLICY['id'], status='running', cap=CAP, cases=[])
    write_json(path, report)
    generator = torch.Generator(device='cpu').manual_seed(90218)
    for length in lengths:
        q, k, v = (torch.randn(1, heads, length, dim, dtype=torch.float64,
                              generator=generator).to(device) for _ in range(3))
        for kind in ('causal', 'physical', 'cached'):
            query = q[..., -1:, :] if kind == 'cached' else q
            mask = None
            if kind != 'causal':
                mask = torch.ones(1, 1, query.shape[-2], length, dtype=torch.bool, device=device)
                if kind == 'physical':
                    mask = mask.tril()
                mask[..., 2::7] = False
            actual = exp_sum_attention(query, k, v, mask, is_causal=kind == 'causal')
            expected = explicit_attention(query.cpu(), k.cpu(), v.cpu(), None if mask is None else mask.cpu(),
                                          is_causal=kind == 'causal')
            delta = error(actual, expected)
            repeat = error(actual, exp_sum_attention(query, k, v, mask, is_causal=kind == 'causal'))
            passed = delta is not None and repeat is not None and max(delta, repeat) <= CAP
            item = dict(tokens=length, mode=kind, cpu_error=delta, repeat_error=repeat,
                        status='passed' if passed else 'failed')
            report['cases'].append(item)
            print(f'GPU exp/sum probe: {item}', flush=True)
            write_json(path, report)
            if not passed:
                report['status'] = 'failed'
                write_json(path, report)
                raise ValueError('GPU exp/sum reference failed synthetic CPU/repeat checks')
    report['status'] = 'passed'
    write_json(path, report)
    return report

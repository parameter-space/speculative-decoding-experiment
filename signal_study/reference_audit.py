"""Read-only FP64 attention cross-checks. Return the original SDPA output unchanged."""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from .capture import target_logits
from .diagnose import compare, snapshot_structure
from .state import preserved_rng


def explicit_attention(query, key, value, mask=None, *, is_causal=False, scale=None, chunk=32):
    """Independent QK/softmax/PV expression, bounded along the query axis."""
    if any(t.dtype != torch.float64 for t in (query, key, value)):
        raise ValueError('reference attention requires FP64 Q/K/V')
    if chunk <= 0:
        raise ValueError('query chunk must be positive')
    if is_causal and mask is not None:
        raise ValueError('causal and explicit mask are mutually exclusive in this audit')
    scale = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    result = torch.empty((*query.shape[:-1], value.shape[-1]), dtype=query.dtype, device=query.device)
    for start in range(0, query.shape[-2], chunk):
        end = min(start + chunk, query.shape[-2])
        scores = (query[..., start:end, :] @ key.transpose(-1, -2)) * scale
        if is_causal:
            allowed = torch.arange(key.shape[-2], device=query.device)[None, :] <= \
                      torch.arange(start, end, device=query.device)[:, None]
            scores.masked_fill_(~allowed, -math.inf)
        elif mask is not None:
            part = mask if mask.shape[-2] == 1 else mask[..., start:end, :]
            if part.dtype == torch.bool:
                scores.masked_fill_(~part, -math.inf)
            else:
                scores += part
        empty = torch.isneginf(scores).all(-1, keepdim=True)
        scores = scores.masked_fill(empty, 0)
        probs = scores.softmax(-1).masked_fill(empty, 0)
        result[..., start:end, :] = probs @ value
    return result


@contextmanager
def audit_attention(report):
    """Single-thread diagnostic scope; BF16 Drafter and production SDPA are untouched."""
    original = F.scaled_dot_product_attention
    report.update(calls=0, max_explicit_error=0.0, max_repeat_error=0.0, events=[],
                  event_count=0, comparison_cap=1e-9, output_policy='original SDPA returned unchanged')

    def observed(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        if enable_gqa:
            kwargs['enable_gqa'] = True
        actual = original(q, k, v, **kwargs)
        if q.dtype != torch.float64:
            return actual
        if dropout_p != 0 or enable_gqa:
            raise ValueError('audit expects inference and already-expanded GQA keys')
        with torch.autocast(q.device.type, enabled=False):
            expected = explicit_attention(q, k, v, attn_mask, is_causal=is_causal, scale=scale)
            again = original(q, k, v, **kwargs)
        report['calls'] += 1
        finite = all(bool(torch.isfinite(t).all()) for t in (actual, expected, again))
        delta = float((actual - expected).abs().max()) if finite else None
        repeat = float((actual - again).abs().max()) if finite else None
        if finite:
            report['max_explicit_error'] = max(report['max_explicit_error'], delta)
            report['max_repeat_error'] = max(report['max_repeat_error'], repeat)
        if not finite or delta > report['comparison_cap'] or repeat > report['comparison_cap']:
            report['event_count'] += 1
            if len(report['events']) < 8:
                event = dict(phase=report.get('phase'), query_shape=list(q.shape), key_shape=list(k.shape),
                             is_causal=is_causal, finite=finite, explicit_error=delta, repeat_error=repeat,
                             mask_dtype=None if attn_mask is None else str(attn_mask.dtype))
                # CPU reference uses the identical final query, keys, values and mask; no model loading.
                mask = None if attn_mask is None else attn_mask[..., -1:, :].detach().cpu()
                if is_causal:
                    mask = torch.arange(k.shape[-2])[None, :] <= q.shape[-2] - 1
                cpu_expected = explicit_attention(q[..., -1:, :].detach().cpu(), k.detach().cpu(),
                                                  v.detach().cpu(), mask, scale=scale)
                event['cpu_last_query'] = {
                    'sdpa_error': float((actual[..., -1:, :].detach().cpu() - cpu_expected).abs().max()) if finite else None,
                    'explicit_error': float((expected[..., -1:, :].detach().cpu() - cpu_expected).abs().max()) if finite else None}
                report['events'].append(event)
        return actual

    F.scaled_dot_product_attention = observed
    try:
        yield report
    finally:
        F.scaled_dot_product_attention = original


def audit_rows(rows, prior):
    if (prior.get('policy') != 'target-fp64-chunked-reference-v1' or prior.get('status') != 'failed'
            or prior.get('stage') != 'finished' or prior.get('planned') != len(rows)):
        raise ValueError('requires a finished failed FP64 v1 preflight summary on the frozen case count')
    failed = [item['prompt_id'] for item in prior.get('failures', [])]
    known = {row['prompt_id'] for row in rows}
    if not failed or len(set(failed)) != len(failed) or not set(failed) <= known:
        raise ValueError('prior failure IDs must be unique and present in frozen inputs')
    control = next((row for row in rows if row['prompt_id'] not in failed), None)
    selected = [row for row in rows if row['prompt_id'] in failed]
    return selected + ([control] if control is not None else [])


@torch.inference_mode()
def probe_snapshot(mod, snap):
    """Localize full/cached/repeat and physical-layout differences without modifying the cache."""
    from transformers import DynamicCache
    report = {'structure': snapshot_structure(snap), 'cache_finite': all(
        bool(torch.isfinite(t).all()) for pair in snap['v_cache']['layers'] for t in pair)}
    with preserved_rng():
        fresh = target_logits(mod, snap)
        cached = target_logits(mod, snap, cached=True)
        report['fresh_cached'] = compare(fresh, cached)
        report['fresh_repeat'] = compare(fresh, target_logits(mod, snap))
        report['cached_repeat'] = compare(cached, target_logits(mod, snap, cached=True))
        prefix = snap['prefix'][None].to(mod.device)
        pos = torch.arange(prefix.shape[1], device=mod.device)[None]
        cache = DynamicCache()
        decoder = mod.v_base.get_decoder()
        decoder(prefix[:, :-1], position_ids=pos[:, :-1], attention_mask=torch.ones_like(prefix[:, :-1]),
                past_key_values=cache, use_cache=True)
        tail = decoder(prefix[:, -1:], position_ids=pos[:, -1:], attention_mask=torch.ones_like(prefix),
                       past_key_values=cache, use_cache=True)
        report['clean_full_split'] = compare(fresh, mod.v_base.lm_head(tail.last_hidden_state[:, -1]))
        del tail, cache
        end = snap['curr'] + 1
        physical = decoder(snap['ids'][:, :end].to(mod.device), position_ids=snap['positions'][:, :end].to(mod.device),
                           attention_mask=snap['mask'][:, :end].to(mod.device), use_cache=False)
        physical_logits = mod.v_base.lm_head(physical.last_hidden_state[:, -1])
        report['fresh_physical'] = compare(fresh, physical_logits)
        report['cached_physical'] = compare(cached, physical_logits)
    return report

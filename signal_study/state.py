from __future__ import annotations

from contextlib import contextmanager
import torch
from transformers import DynamicCache


def cpu(t):
    return t.detach().to("cpu", copy=True)


def cache_to_cpu(cache):
    # Only the pinned DynamicCache contract is supported, never silently drop metadata.
    if type(cache) is not DynamicCache:
        raise TypeError(f"unsupported cache class: {type(cache).__name__}")
    return {"layers": [(cpu(k), cpu(v)) for k, v in cache.to_legacy_cache()],
            "length": int(cache.get_seq_length())}


def restore_cache(state, device):
    result = DynamicCache.from_legacy_cache(tuple((k.to(device, copy=True), v.to(device, copy=True)) for k, v in state["layers"]))
    if int(result.get_seq_length()) != state["length"]:
        raise ValueError("cache length changed during round trip")
    return result


def rng_state():
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


@contextmanager
def preserved_rng(state=None):
    previous = rng_state()
    if state is not None:
        torch.set_rng_state(state["cpu"])
        if state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
    try:
        yield
    finally:
        torch.set_rng_state(previous["cpu"])
        if previous["cuda"]:
            torch.cuda.set_rng_state_all(previous["cuda"])


def logical_prefix(ids, mask, positions, curr):
    if ids.shape[0] != 1:
        raise ValueError("only batch1 is supported")
    active = mask[0, :curr + 1].bool()
    if not active[-1]:
        raise ValueError("boundary has no committed z")
    logical = ids[0, :curr + 1][active]
    actual_pos = positions[0, :curr + 1][active]
    if not torch.equal(actual_pos, torch.arange(len(logical), device=actual_pos.device)):
        raise ValueError("logical positions are not contiguous")
    return cpu(logical)


def distribution(logits):
    result = logits.float().softmax(-1).reshape(-1)
    validate_probability(result)
    return result


def validate_probability(p):
    if p.ndim != 1 or not p.numel() or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("invalid probability vector")
    if abs(p.sum().item() - 1) > 1e-5:
        raise ValueError("probabilities do not sum to one")


def overlap(p, q):
    validate_probability(p)
    validate_probability(q)
    if p.shape != q.shape:
        raise ValueError("vocabulary mismatch")
    return torch.minimum(p, q).sum().item()

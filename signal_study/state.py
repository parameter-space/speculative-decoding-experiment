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


PROBABILITY_SUM_CAP = 1e-5
METRIC_POLICY = {"id": "cpu-fp64-probability-v1", "device": "cpu",
                 "softmax": "float64", "probability_accumulation": "float64",
                 "sum_cap": PROBABILITY_SUM_CAP, "renormalize": False}


def _logit_vector(logits, context):
    # Never merge multiple token/batch distributions into one vocabulary vector.
    if (logits.ndim < 1 or not logits.numel() or logits.numel() != logits.shape[-1]
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError(f"{context}: invalid single-token logits; shape={tuple(logits.shape)}, dtype={logits.dtype}")
    return logits.detach().to("cpu").reshape(-1)


def distribution(logits, *, context="distribution"):
    # Measurement only: do not change model forwards or upstream token sampling.
    result = _logit_vector(logits, context).double().softmax(-1)
    validate_probability(result, context=context)
    return result


def validate_probability(p, *, context="probability"):
    if p.ndim != 1 or not p.numel() or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{context}: invalid probability vector; shape={tuple(p.shape)}, dtype={p.dtype}")
    total = p.sum(dtype=torch.float64).item()
    if abs(total - 1) > PROBABILITY_SUM_CAP:
        raise ValueError(f"{context}: probabilities do not sum to one; shape={tuple(p.shape)}, "
                         f"dtype={p.dtype}, sum={total:.17g}, error={abs(total - 1):.9g}, "
                         f"cap={PROBABILITY_SUM_CAP}")


def probability_audit(logits, *, context):
    """Compare old/new measurement arithmetic on identical logits, without masking failures."""
    vector = _logit_vector(logits, context)
    report = {"context": context, "logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype)}
    for label, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        p = vector.to(dtype).softmax(-1)
        native_sum, sum64 = p.sum().item(), p.sum(dtype=torch.float64).item()
        report[label] = {"native_sum": native_sum, "sum_fp64": sum64,
                         "native_sum_error": abs(native_sum - 1), "sum_fp64_error": abs(sum64 - 1),
                         "native_sum_gate_passed": abs(native_sum - 1) <= PROBABILITY_SUM_CAP,
                         "fp64_sum_gate_passed": abs(sum64 - 1) <= PROBABILITY_SUM_CAP}
    return report


def overlap(p, q):
    validate_probability(p)
    validate_probability(q)
    if p.shape != q.shape:
        raise ValueError("vocabulary mismatch")
    return torch.minimum(p, q).sum(dtype=torch.float64).item()

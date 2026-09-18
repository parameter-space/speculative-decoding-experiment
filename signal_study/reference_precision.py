"""Slow validation reference: FP64 Target arithmetic without an FP64 weight replica.

Only pinned Llama/default-or-llama3 RoPE is supported. Upstream files stay untouched.
Weights retain their stored values; this does not recover checkpoint precision.
"""
from contextlib import contextmanager
from functools import partial

import torch
from torch.nn import functional as F


POLICY = {
    "id": "target-fp64-chunked-reference-v1", "scope": "preflight_only_not_S1_results",
    "target": "FP32 weight storage; bounded temporary FP64 linear weights; FP64 hidden/KV/head",
    "target_cache": "FP64 created by actual generation, never cast from a captured low-precision cache",
    "drafter_and_guidance": "BF16; Target guide input explicitly converted to FP32 before BF16 autocast",
    "norm_and_rope": "FP64 arithmetic on existing norm weights, inv_freq and scaling",
    "linear_chunk_rows": 1024, "sdpa": "math", "tf32": False,
    "tolerances": "unchanged caps and rule, frozen from this reference baseline before endpoints",
    "note": "Slow numerical validation reference, NOT official_eval reproduction or a speed benchmark.",
}


def linear64(module, x, *, chunk_rows=1024, audit=None):
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    x = x.double()
    result = torch.empty((*x.shape[:-1], module.weight.shape[0]), dtype=torch.float64, device=x.device)
    for start in range(0, module.weight.shape[0], chunk_rows):
        stop = min(start + chunk_rows, module.weight.shape[0])
        weight = module.weight[start:stop].double()
        bias = None if module.bias is None else module.bias[start:stop].double()
        result[..., start:stop] = F.linear(x, weight, bias)
        if audit is not None:
            audit["max_temporary_weight_bytes"] = max(audit["max_temporary_weight_bytes"],
                                                       weight.numel() * weight.element_size())
        del weight, bias
    if audit is not None:
        audit["linear_calls"] += 1
    return result


def rms64(module, hidden):
    hidden = hidden.double()
    return module.weight.double() * (hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True)
                                                        + module.variance_epsilon))


def rope64(module, x, position_ids):
    # No dynamic buffer update: supported fixed-frequency RoPE types only.
    if module.rope_type not in ("default", "llama3"):
        raise ValueError("FP64 reference supports only fixed default/llama3 RoPE")
    inv = module.inv_freq.to(device=x.device, dtype=torch.float64)
    freq = position_ids.double()[..., None] * inv[None, None, :]
    emb = torch.cat((freq, freq), dim=-1)
    return emb.cos() * module.attention_scaling, emb.sin() * module.attention_scaling


@contextmanager
def reference_operators(mod, *, chunk_rows=1024):
    from src.models.llama import LlamaRMSNorm, LlamaRotaryEmbedding
    decoder = mod.v_base.get_decoder()
    if decoder.config._attn_implementation != "sdpa":
        raise ValueError("FP64 reference requires SDPA, not eager float32 softmax")
    excluded = {id(m) for m in mod.guidance_embd_layer.modules()}
    originals = []
    audit = {"linear_calls": 0, "max_temporary_weight_bytes": 0, "chunk_rows": chunk_rows}

    def override(module, forward):
        originals.append((module, "forward" in module.__dict__, module.forward))
        module.forward = forward

    try:
        for module in mod.v_base.modules():
            if id(module) in excluded:
                continue
            if isinstance(module, torch.nn.Linear):
                override(module, partial(linear64, module, chunk_rows=chunk_rows, audit=audit))
            elif isinstance(module, LlamaRMSNorm):
                override(module, partial(rms64, module))
            elif isinstance(module, LlamaRotaryEmbedding):
                if module.rope_type not in ("default", "llama3"):
                    raise ValueError("unsupported reference RoPE type")
                override(module, partial(rope64, module))
        embedding = decoder.embed_tokens
        embedding_forward = embedding.forward

        def lookup(*args, **kwargs):
            return embedding_forward(*args, **kwargs).double()

        override(embedding, lookup)
        yield audit
    finally:
        for module, had_override, original in reversed(originals):
            if had_override:
                module.forward = original
            else:
                module.__dict__.pop("forward", None)

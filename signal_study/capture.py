from __future__ import annotations

from types import MethodType
import torch

from .common import digest
from .state import cache_to_cpu, cpu, distribution, logical_prefix, preserved_rng, restore_cache, rng_state


class BoundaryCaptured(Exception):
    """Internal stop after the first real proposal logits, before sampling."""


class Capture:
    def __init__(self, mod, stop=True):
        self.mod, self.stop = mod, stop
        self.round = 0
        self.snapshot = None
        self.pending_snapshot = False
        self.source = None
        self.last_g = self.last_r = self.last_v_logits = None
        self.tensor_map = {}

    def __enter__(self):
        self.original_step = self.mod._spec_dec_step
        self.mod._spec_dec_step = MethodType(self._step, self.mod)
        self.handles = [
            self.mod.guidance_embd_layer.orig.register_forward_hook(self._guide),
            self.mod.v_base.lm_head.register_forward_hook(self._v_head),
            self.mod.d_base.get_decoder().register_forward_pre_hook(self._d_pre, with_kwargs=True),
            self.mod.d_base.lm_head.register_forward_hook(self._d_head),
        ]
        return self

    def __exit__(self, *_):
        self.mod._spec_dec_step = self.original_step
        for handle in self.handles:
            handle.remove()

    def _guide(self, module, args, output):
        self.last_r, self.last_g = cpu(args[0]), cpu(output)
        self.tensor_map = {"R": {"path": "guidance_embd_layer.orig.input", "shape": list(args[0].shape),
                                "dtype": str(args[0].dtype), "device": str(args[0].device)},
                           "G": {"path": "guidance_embd_layer.orig.output", "shape": list(output.shape),
                                "dtype": str(output.dtype), "device": str(output.device)},
                           "layers": self.mod.guidance_embd_layer.in_layer,
                           "layer_convention": "0-based pre-block hidden; capture precedes hidden_states = layer_outputs[0]",
                           "intervention_port": "latent_mod_prep input, before nn.LayerNorm",
                           "scope": "endpoint", "old_conditioned_KV_retained": True}

    def _v_head(self, module, args, output):
        self.last_v_logits = cpu(output)

    def _step(self, bound_mod, ids, positions, mask, curr, guides, d_cache, v_cache, has_ended):
        self.round += 1
        if self.stop and self.round >= 2 and not has_ended.any():
            if self.source is None:
                raise ValueError("missing preceding verifier source")
            prefix = logical_prefix(ids, mask, positions, curr)
            source = self.source
            if source["source_pos"] + 1 != len(prefix) - 1:
                raise ValueError("source/z timing misalignment")
            logical_source = ids[0, :source["source_physical_pos"] + 1][mask[0, :source["source_physical_pos"] + 1].bool()]
            if not torch.equal(cpu(logical_source), prefix[:-1]):
                raise ValueError("source prefix is not the committed prefix minus z")
            self.snapshot = {"ids": cpu(ids), "positions": cpu(positions), "mask": cpu(mask),
                             "curr": int(curr), "guides": cpu(guides), "has_ended": cpu(has_ended),
                             "d_cache": cache_to_cpu(d_cache), "v_cache": cache_to_cpu(v_cache),
                             "prefix": prefix, "prefix_hash": digest(prefix.tolist()), "rng": rng_state(),
                             "round": self.round, "state_origin": "natural_sd_boundary", **source}
            self.pending_snapshot = True
        result = self.original_step(ids, positions, mask, curr, guides, d_cache, v_cache, has_ended)
        na = int(result[3][0])
        physical = curr + na
        # Source G and p_src came from the actual verifier call, not a fresh-prefix substitute.
        self.source = {"G": self.last_g[:, na:na+1].clone(),
                       "p_src_logits": self.last_v_logits[:, na].clone(),
                       "source_pos": int(positions[0, physical]), "source_physical_pos": physical}
        return result

    def _d_pre(self, module, args, kwargs):
        if not self.pending_snapshot:
            return
        ids = args[0] if args else kwargs["input_ids"]
        snapshot = self.snapshot
        start = snapshot["d_cache"]["length"]
        if not torch.equal(cpu(ids), snapshot["ids"][:, start:snapshot["curr"] + 1]):
            raise ValueError("pending draft slice does not match original call")
        snapshot["pending_ids"] = cpu(ids)
        snapshot["pending_positions"] = cpu(kwargs["position_ids"])
        snapshot["pending_mask"] = cpu(kwargs["attention_mask"])
        snapshot["expanded_delta"] = cpu(kwargs["guidance_embeds"])
        self.tensor_map["Delta"] = {"path": "latent_mod_prep.output", "shape": list(snapshot["guides"].shape),
                                    "dtype": str(snapshot["guides"].dtype), "device": str(ids.device)}

    def _d_head(self, module, args, output):
        if self.pending_snapshot:
            self.snapshot["q_original_logits"] = cpu(output)
            self.pending_snapshot = False
            raise BoundaryCaptured()


def capture_prompt(mod, row, max_new_tokens):
    ids, mask = mod.prep_for_gen([row["prompt"]])
    with Capture(mod) as recorder:
        try:
            mod.generate(ids, mask, max_new_tokens=max_new_tokens)
        except BoundaryCaptured:
            pass
    if recorder.snapshot is None or "q_original_logits" not in recorder.snapshot:
        return None, recorder.tensor_map
    return recorder.snapshot, recorder.tensor_map


@torch.no_grad()
def draft_logits(mod, snapshot, guide=None, stored_delta=False):
    device = mod.device
    with preserved_rng(snapshot["rng"]):
        cache = restore_cache(snapshot["d_cache"], device)
        delta = snapshot["guides"].to(device) if stored_delta else mod.latent_mod_prep(
            snapshot["G"].to(device) if guide is None else guide.to(device))
        count = snapshot["pending_ids"].shape[1]
        output = mod.d_base.get_decoder()(
            snapshot["pending_ids"].to(device), position_ids=snapshot["pending_positions"].to(device),
            attention_mask=snapshot["pending_mask"].to(device), past_key_values=cache,
            guidance_embeds=delta.expand(-1, -1, count, -1), use_cache=True)
        return cpu(mod.d_base.lm_head(output.last_hidden_state[:, -1]))


@torch.no_grad()
def target_logits(mod, snapshot, cached=False):
    device = mod.device
    if cached:
        start, end = snapshot["v_cache"]["length"], snapshot["curr"] + 1
        if start >= end:
            raise ValueError("no pending z for target cache")
        output = mod.v_base.get_decoder()(
            snapshot["ids"][:, start:end].to(device),
            attention_mask=snapshot["mask"][:, :end].to(device),
            position_ids=snapshot["positions"][:, start:end].to(device),
            past_key_values=restore_cache(snapshot["v_cache"], device), use_cache=True)
    else:
        prefix = snapshot["prefix"][None].to(device)
        output = mod.v_base.get_decoder()(prefix, attention_mask=torch.ones_like(prefix),
                 position_ids=torch.arange(prefix.shape[1], device=device)[None], use_cache=False)
    return cpu(mod.v_base.lm_head(output.last_hidden_state[:, -1]))


def source_metadata(snapshot, row):
    p = distribution(snapshot["p_src_logits"])
    positive = p > 0
    return {"prompt_id": row["prompt_id"], "domain": row["domain"], "split": row["split"],
            "z_id": int(snapshot["prefix"][-1]), "p_src_argmax": int(p.argmax()),
            "p_src_z": float(p[int(snapshot["prefix"][-1])]),
            "p_src_entropy": float(-(p[positive] * p[positive].log()).sum()),
            "source_length": len(snapshot["prefix"]) - 1, "source_pos": snapshot["source_pos"]}


@torch.no_grad()
def synthetic_snapshot(mod, row):
    from transformers import DynamicCache
    prefix = torch.tensor([row["prefix_ids"]], dtype=torch.long, device=mod.device)
    source, z = prefix[:, :-1], prefix[:, -1:]
    positions = torch.arange(prefix.shape[1], device=mod.device)[None]
    v_cache, d_cache = DynamicCache(), DynamicCache()
    vout = mod.v_base.get_decoder()(source, position_ids=positions[:, :-1],
             attention_mask=torch.ones_like(source), past_key_values=v_cache, use_cache=True, compute_guidance=True)
    g = vout["guide_embd"][:, -1:]
    mod.d_base.get_decoder()(source, position_ids=positions[:, :-1], attention_mask=torch.ones_like(source),
                           past_key_values=d_cache, use_cache=True)
    result = {"ids": cpu(prefix), "positions": cpu(positions), "mask": cpu(torch.ones_like(prefix)),
              "curr": prefix.shape[1] - 1, "guides": cpu(mod.latent_mod_prep(g)), "G": cpu(g),
              "d_cache": cache_to_cpu(d_cache), "v_cache": cache_to_cpu(v_cache), "prefix": cpu(prefix[0]),
              "prefix_hash": digest(prefix[0].tolist()), "rng": rng_state(), "round": None,
              "state_origin": "synthetic_teacher_prefix", "source_pos": prefix.shape[1] - 2,
              "source_physical_pos": prefix.shape[1] - 2, "pending_ids": cpu(z),
              "pending_positions": cpu(positions[:, -1:]), "pending_mask": cpu(torch.ones_like(prefix)),
              "has_ended": torch.tensor([False]),
              "p_src_logits": cpu(mod.v_base.lm_head(vout["out"].last_hidden_state[:, -1]))}
    result["q_original_logits"] = draft_logits(mod, result, stored_delta=True)
    return result

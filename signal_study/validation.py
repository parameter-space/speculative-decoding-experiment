from __future__ import annotations

import torch
from transformers import DynamicCache

from .capture import Capture, draft_logits, target_logits
from .state import distribution, preserved_rng


class ValidationError(RuntimeError):
    pass


def error(a, b):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValidationError("nonfinite or incompatible tensors")
    return float((a.float().cpu() - b.float().cpu()).abs().max())


def require_error(a, b, tolerance, name):
    value = error(a, b)
    if value > tolerance:
        raise ValidationError(f"{name}: max logit/delta error {value:.8g} > {tolerance:.8g}")
    return value


@torch.no_grad()
def baseline_tests(mod, prompts, cfg):
    reports = []
    largest_kernel_error = 0.0
    largest_kernel_tv = 0.0
    was_greedy = mod.greedy_sample
    try:
        mod.greedy_sample = True
        for prompt_index, prompt in enumerate(prompts):
            ids, mask = mod.prep_for_gen([prompt])
            with preserved_rng():
                torch.manual_seed(cfg["seed"])
                vanilla_hookless = mod.generate(ids, mask, max_new_tokens=32)
                torch.manual_seed(cfg["seed"])
                with Capture(mod, stop=False):
                    observed = mod.generate(ids, mask, max_new_tokens=32)
            require_error(vanilla_hookless[0], observed[0], 0, "non-mutating hooks token identity")
            require_error(vanilla_hookless[1]["attention_mask"], observed[1]["attention_mask"], 0, "hook mask identity")
            logical = observed[0][0][observed[1]["attention_mask"][0].bool()]
            suffix = logical[ids.shape[1]:][:32]
            # Exact greedy AR comparison uses the same model, template, precision, and EOS.
            cache = DynamicCache()
            current = ids
            ar = []
            for i in range(len(suffix)):
                output = mod.v_base.get_decoder()(current, past_key_values=cache, use_cache=True)
                token = mod.v_base.lm_head(output.last_hidden_state[:, -1]).argmax(-1)
                ar.append(int(token))
                current = token[:, None]
            if not suffix.numel():
                raise ValidationError("baseline generated no valid token before EOS")
            if ar != suffix.tolist():
                raise ValidationError("greedy SD differs from same-template target AR; investigate, do not proceed")

            # Independent clean-prefix kernel calibration, before any intervention or masked-cache test.
            prefix = ids
            full = mod.v_base.get_decoder()(prefix, use_cache=False)
            logits_full = mod.v_base.lm_head(full.last_hidden_state[:, -1])
            cache = DynamicCache()
            mod.v_base.get_decoder()(prefix[:, :-1], past_key_values=cache, use_cache=True)
            tail = mod.v_base.get_decoder()(prefix[:, -1:], past_key_values=cache, use_cache=True)
            logits_split = mod.v_base.lm_head(tail.last_hidden_state[:, -1])
            kernel_error = error(logits_full, logits_split)
            kernel_tv = float((distribution(logits_full) - distribution(logits_split)).abs().sum() / 2)
            if kernel_error > cfg["alignment_logit_cap"] or kernel_tv > cfg["alignment_tv_cap"]:
                raise ValidationError(
                    "independent clean-prefix kernel discrepancy exceeds predeclared cap: "
                    f"prompt_index={prompt_index}, prefix_tokens={prefix.shape[1]}, "
                    f"max_logit_error={kernel_error:.9g} (cap={cfg['alignment_logit_cap']:.9g}), "
                    f"TV={kernel_tv:.9g} (cap={cfg['alignment_tv_cap']:.9g}), "
                    f"full_argmax={logits_full.argmax(-1).item()}, "
                    f"split_argmax={logits_split.argmax(-1).item()}, "
                    f"full_dtype={logits_full.dtype}, split_dtype={logits_split.dtype}; "
                    "hook identity and greedy AR identity passed for this prompt"
                )
            largest_kernel_error = max(largest_kernel_error, kernel_error)
            largest_kernel_tv = max(largest_kernel_tv, kernel_tv)
            reports.append({"hook_identity": "passed", "greedy_AR_identity": "passed", "compared_tokens": len(ar),
                            "clean_kernel_max_logit_error": kernel_error, "clean_kernel_TV": kernel_tv})
    finally:
        mod.greedy_sample = was_greedy
    tolerance = {"repeat": cfg["repeat_logit_cap"],
                 "alignment": min(cfg["alignment_logit_cap"], max(1e-6, 2 * largest_kernel_error)),
                 "alignment_tv": min(cfg["alignment_tv_cap"], max(1e-6, 2 * largest_kernel_tv)),
                 "policy": "fixed from two independent clean-prefix kernel comparisons before endpoint data; never relaxed after failures"}
    return reports, tolerance


@torch.no_grad()
def validate_snapshot(mod, snapshot, tolerance):
    original = snapshot["q_original_logits"]
    repeat = draft_logits(mod, snapshot, stored_delta=True)
    repeat2 = draft_logits(mod, snapshot, stored_delta=True)
    tests = {"restore": require_error(original, repeat, tolerance["repeat"], "snapshot restore"),
             "repeat": require_error(repeat, repeat2, tolerance["repeat"], "repeat"),
             "G_to_Delta": require_error(snapshot["guides"], mod.latent_mod_prep(snapshot["G"].to(mod.device)),
                                         tolerance["repeat"], "G to Delta"),
             "self_copy": require_error(original, draft_logits(mod, snapshot, snapshot["G"].clone()),
                                        tolerance["repeat"], "self-copy")}
    fresh, cached = target_logits(mod, snapshot), target_logits(mod, snapshot, cached=True)
    tests["p_fresh_cached"] = require_error(fresh, cached, tolerance["alignment"], "logical vs physical target prefix")
    p, q = distribution(fresh), distribution(original)
    tests["p_fresh_cached_TV"] = float((p - distribution(cached)).abs().sum() / 2)
    if tests["p_fresh_cached_TV"] > tolerance["alignment_tv"]:
        raise ValidationError("logical vs physical target probability TV exceeds frozen tolerance")
    if len(p) != len(q):
        raise ValidationError("full-vocabulary size mismatch")
    tests["p_sum"], tests["q_sum"] = float(p.sum()), float(q.sum())
    return p, tests

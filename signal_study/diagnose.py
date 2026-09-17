"""Bounded clean-prefix diagnostics; never launches endpoint experiments or relaxes gates."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import socket

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import DynamicCache

from .common import digest, read_json, read_jsonl, write_json
from .runtime import load
from .state import distribution
from .validation import baseline_tests, error, ValidationError


def compare(a, b):
    return {"max_logit_error": error(a, b),
            "TV": float((distribution(a) - distribution(b)).abs().sum() / 2),
            "argmax_a": int(a.argmax(-1).item()), "argmax_b": int(b.argmax(-1).item())}


def fp32_head(head, hidden):
    # Diagnostic only: identical stored weights, FP32 arithmetic, bounded temporary weight memory.
    # Casting the already-computed native logits to FP32 would not test output-layer rounding.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(hidden.device.type, enabled=False):
            parts = []
            for start in range(0, head.weight.shape[0], 4096):
                bias = None if head.bias is None else head.bias[start:start + 4096].float()
                parts.append(F.linear(hidden.float(), head.weight[start:start + 4096].float(), bias).cpu())
            return torch.cat(parts, dim=-1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@torch.inference_mode()
def probe(model, ids, *, explicit=False, math_backend=False):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
        raise ValueError("probe needs one unpadded prefix with at least two tokens")
    positions = torch.arange(ids.shape[1], device=ids.device)[None]

    def forward(split):
        decoder = model.get_decoder()
        if split:
            cache = DynamicCache()
            kw = dict(attention_mask=torch.ones_like(ids[:, :-1]), position_ids=positions[:, :-1]) if explicit else {}
            decoder(ids[:, :-1], past_key_values=cache, use_cache=True, **kw)
            if cache.get_seq_length() != ids.shape[1] - 1:
                raise ValueError("unexpected prefix cache length")
            kw = dict(attention_mask=torch.ones_like(ids), position_ids=positions[:, -1:]) if explicit else {}
            out = decoder(ids[:, -1:], past_key_values=cache, use_cache=True, **kw)
            if cache.get_seq_length() != ids.shape[1]:
                raise ValueError("unexpected tail cache length")
        else:
            kw = dict(attention_mask=torch.ones_like(ids), position_ids=positions) if explicit else {}
            out = decoder(ids, use_cache=False, **kw)
        hidden = out.last_hidden_state[:, -1]
        return model.lm_head(hidden).cpu(), fp32_head(model.lm_head, hidden), hidden.float().cpu()

    with sdpa_kernel(SDPBackend.MATH) if math_backend else nullcontext():
        full, full_head, full_hidden = forward(False)
        split, split_head, split_hidden = forward(True)
        full_repeat, _, _ = forward(False)
        split_repeat, _, _ = forward(True)
    return {"native": compare(full, split), "fp32_head_only": compare(full_head, split_head),
            "full_repeat": compare(full, full_repeat), "split_repeat": compare(split, split_repeat),
            "hidden_max_error": error(full_hidden, split_hidden), "native_dtype": str(full.dtype),
            "explicit_positions_mask": explicit, "forced_math_sdpa": math_backend}


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0] != "ariel-k2":
        raise ValueError("diagnostic requires an allocated K2 compute job")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("diagnostic requires exactly one visible allocated GPU")
    cfg, data, out = read_json(args.config), Path(args.data_dir), Path(args.output)
    manifest = read_json(data / "manifest.json")
    natural = read_jsonl(data / "natural.jsonl")
    if manifest["config_hash"] != digest(cfg) or manifest["natural_hash"] != digest(natural):
        raise ValueError("frozen config/data hash mismatch")
    rows = [r for r in natural if r["split"] == "smoke"][:2]
    if len(rows) != 2:
        raise ValueError("missing baseline prompts")
    out.mkdir(parents=True, exist_ok=False)
    report = {"scope": "diagnostic_only_not_S1_results", "status": "running", "prompts": [],
              "config": cfg, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0),
              "note": "No tolerances changed. FP32 head/math SDPA are isolated probes, not production fixes."}
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg["seed"])
    try:
        mod, models = load(cfg, manifest, args.upstream, out / "reports")
        report["models"] = models
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Reproduce the same generation warmup and gate before independent probes.
            try:
                baseline, tolerance = baseline_tests(mod, [r["prompt"] for r in rows], cfg)
                report["baseline"] = {"status": "passed", "reports": baseline, "tolerance": tolerance}
            except ValidationError as exc:
                report["baseline"] = {"status": "failed", "error": str(exc)}
            print("Baseline:", report["baseline"]["status"], report["baseline"].get("error", ""), flush=True)
            write_json(out / "diagnostic.json", report)
            for index, row in enumerate(rows):
                ids, mask = mod.prep_for_gen([row["prompt"]])
                if not mask.bool().all():
                    raise ValueError("unpadded single-prompt assumption failed")
                item = {"prompt_index": index, "prompt_id": row["prompt_id"],
                        "prefix_tokens": ids.shape[1], "prefix_hash": digest(ids.cpu().tolist()), "modes": {}}
                report["prompts"].append(item)
                for name, explicit, math in (("default", False, False), ("explicit", True, False), ("math", True, True)):
                    result = probe(mod.v_base, ids, explicit=explicit, math_backend=math)
                    item["modes"][name] = result
                    print(f"prompt={index} mode={name} native_TV={result['native']['TV']:.9g} "
                          f"fp32_head_TV={result['fp32_head_only']['TV']:.9g} "
                          f"full_repeat_TV={result['full_repeat']['TV']:.9g} "
                          f"split_repeat_TV={result['split_repeat']['TV']:.9g}", flush=True)
                    write_json(out / "diagnostic.json", report)
        report["status"] = "complete"
        print("Diagnostic complete; this is NOT a passed S1 experiment.", flush=True)
        return 0
    except Exception as exc:
        # Avoid leaking credentials or signed download URLs in exception text.
        report.update(status="failed", error_type=type(exc).__name__)
        print(f"Diagnostic stopped: {type(exc).__name__}; inspect diagnostic.json", flush=True)
        return 2
    finally:
        write_json(out / "diagnostic.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--upstream", default="vendor/SD-square")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(main(parser.parse_args()))

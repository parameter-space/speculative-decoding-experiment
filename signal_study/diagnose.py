"""Bounded baseline/snapshot diagnostics; no S1 effect measurements or relaxed gates."""
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
from .capture import capture_prompt, draft_logits, target_logits
from .runtime import load
from .state import METRIC_POLICY, distribution, preserved_rng, probability_audit
from .validation import baseline_tests, error, require_error, validate_snapshot, ValidationError


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


def check_baseline(mod, prompts, cfg, *, math_backend=False):
    # Apply consistently to generation, target AR, and clean-prefix comparisons.
    # The context restores backend settings even when a validation gate raises.
    with sdpa_kernel(SDPBackend.MATH) if math_backend else nullcontext():
        try:
            baseline, tolerance = baseline_tests(mod, prompts, cfg)
            return {"status": "passed", "reports": baseline, "tolerance": tolerance}
        except ValidationError as exc:
            return {"status": "failed", "error": str(exc)}


def preflight_rows(natural):
    # Fixed order, one first smoke prompt per domain; never search for easier passing cases.
    selected = {}
    for row in natural:
        if row["split"] == "smoke":
            selected.setdefault(row["domain"], row)
    if len(selected) != 4:
        raise ValueError("preflight requires the four frozen smoke domains")
    return list(selected.values())


def calibration_row(natural, prompt_id):
    selected = [r for r in natural if r["split"] == "calibration" and r["prompt_id"] == prompt_id]
    if len(selected) != 1:
        raise ValueError("calibration probe requires one exact frozen calibration prompt ID")
    return selected[0]


def snapshot_structure(snap):
    """Summaries only: do not serialize prompt text, KV tensors, or raw token sequences."""
    end, start = snap["curr"] + 1, snap["v_cache"]["length"]
    ids, mask, positions = (snap[k][0, :end] for k in ("ids", "mask", "positions"))
    active = mask.bool()
    keys = [int(k.shape[-2]) for k, _ in snap["v_cache"]["layers"]]
    values = [int(v.shape[-2]) for _, v in snap["v_cache"]["layers"]]
    return {"round": snap["round"], "logical_tokens": len(snap["prefix"]),
            "physical_tokens": end, "cache_tokens": start, "pending_tokens": end - start,
            "masked_tokens": int((~active).sum()),
            "pending_active_tokens": int(active[start:].sum()),
            "last_token_active": bool(active[-1]),
            "mask_binary": bool(((mask == 0) | (mask == 1)).all()),
            "active_tokens_match_prefix": torch.equal(ids[active], snap["prefix"]),
            "active_positions_contiguous": torch.equal(positions[active], torch.arange(int(active.sum()))),
            "source_z_aligned": snap["source_pos"] + 1 == len(snap["prefix"]) - 1,
            "cache_layer_count": len(keys),
            "cache_layer_lengths_match": bool(keys) and all(n == start for n in keys + values),
            "pending_slice_valid": 0 <= start < end}


@torch.inference_mode()
def alignment_probe(mod, snap, report):
    """Extra forwards for diagnosis only, after the unmodified production gate ran."""
    report["structure"] = snapshot_structure(snap)
    report["note"] = "No tolerance fitting; FP32 head is diagnostic only. Structure checks do not prove KV contents correct."
    with preserved_rng(), sdpa_kernel(SDPBackend.MATH):
        def observe(cached):
            hidden = []
            handle = mod.v_base.lm_head.register_forward_pre_hook(
                lambda module, args: hidden.append(args[0].detach().clone()))
            try:
                logits = target_logits(mod, snap, cached=cached)
            finally:
                handle.remove()
            if len(hidden) != 1:
                raise ValueError("alignment probe expected one target output head call")
            return logits, fp32_head(mod.v_base.lm_head, hidden[0]), hidden[0].float().cpu()

        fresh, fresh_head, fresh_hidden = observe(False)
        cached, cached_head, cached_hidden = observe(True)
        report["fresh_cached"] = compare(fresh, cached)
        report["fresh_cached_fp32_head_only"] = compare(fresh_head, cached_head)
        report["hidden_max_error"] = error(fresh_hidden, cached_hidden)
        report["fresh_repeat"] = compare(fresh, target_logits(mod, snap))
        report["cached_repeat"] = compare(cached, target_logits(mod, snap, cached=True))
        print(f"Alignment: TV={report['fresh_cached']['TV']:.12g} "
              f"fp32_head_TV={report['fresh_cached_fp32_head_only']['TV']:.12g} "
              f"structure={report['structure']}", flush=True)

        # Same logical endpoint, freshly built clean KV: isolates full/split arithmetic.
        report["clean_split"] = probe(mod.v_base, snap["prefix"][None].to(mod.device),
                                      explicit=True, math_backend=True)
        # Same physical slots/mask/positions but no captured KV: isolates layout from KV history.
        end = snap["curr"] + 1
        out = mod.v_base.get_decoder()(
            snap["ids"][:, :end].to(mod.device),
            attention_mask=snap["mask"][:, :end].to(mod.device),
            position_ids=snap["positions"][:, :end].to(mod.device), use_cache=False)
        hidden = out.last_hidden_state[:, -1]
        physical = mod.v_base.lm_head(hidden).cpu()
        physical_head = fp32_head(mod.v_base.lm_head, hidden)
        report["fresh_physical_rebuild"] = compare(fresh, physical)
        report["cached_physical_rebuild"] = compare(cached, physical)
        report["fresh_physical_rebuild_fp32_head_only"] = compare(fresh_head, physical_head)
        report["cached_physical_rebuild_fp32_head_only"] = compare(cached_head, physical_head)
        report["status"] = "complete"
        print(f"Alignment controls: clean_split_TV={report['clean_split']['native']['TV']:.12g} "
              f"fresh_physical_TV={report['fresh_physical_rebuild']['TV']:.12g} "
              f"cached_physical_TV={report['cached_physical_rebuild']['TV']:.12g}", flush=True)


@torch.inference_mode()
def check_endpoint(mod, row, cfg, tolerance, *, audit=None, checks_out=None, alignment=None):
    with sdpa_kernel(SDPBackend.MATH):
        with preserved_rng():
            torch.manual_seed(cfg["seed"])
            snap, _ = capture_prompt(mod, row, cfg["max_new_tokens"])
        if snap is None:
            raise ValidationError("preflight has no valid normal boundary; not a passed cache test")
        if audit is not None:
            logits_by_path = {"target.fresh": target_logits(mod, snap),
                             "target.cached": target_logits(mod, snap, cached=True),
                             "drafter.original": snap["q_original_logits"],
                             "target.source": snap["p_src_logits"]}
            for name, logits in logits_by_path.items():
                audit[name] = probability_audit(logits, context=name)
                distribution(logits, context=name)
        try:
            _, checks = validate_snapshot(mod, snap, tolerance, diagnostics=checks_out)
        finally:
            # Preserve partial diagnostics and the original gate failure even if a probe fails.
            if alignment is not None:
                try:
                    alignment_probe(mod, snap, alignment)
                except Exception as exc:
                    alignment.update(status="failed", error_type=type(exc).__name__)
                    print(f"Alignment diagnostic failed: {type(exc).__name__}", flush=True)
        # Exercise an intervening guide without calculating research effect estimates.
        draft_logits(mod, snap, -snap["G"])
        checks["A_B_A"] = require_error(snap["q_original_logits"], draft_logits(mod, snap),
                                        tolerance["repeat"], "preflight A-B-A")
        return {"checks": checks, "round": snap["round"], "prefix_tokens": len(snap["prefix"])}


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
    calibration_id = args.calibration_prompt_id
    selected_calibration = calibration_row(natural, calibration_id) if calibration_id else None
    math_gate = args.math_baseline_only or args.math_preflight or bool(calibration_id)
    report = {"scope": "diagnostic_only_not_S1_results", "status": "running", "prompts": [],
              "metric_policy": METRIC_POLICY,
              "baseline_sdpa": "math" if math_gate else "default",
              "config": cfg, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0),
              "note": "CPU FP64 measurement policy; model forwards/sampling and gate caps unchanged. "
                      "Baseline-derived tolerances are recomputed before endpoint data. "
                      "FP32 head probes do not change production forwards."}
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg["seed"])
    try:
        if getattr(args, 'gpu_attention_reference', False):
            from .gpu_attention import probe_gpu_reference
            probe_gpu_reference(out / 'gpu-attention-probe.json')
        if getattr(args, 'cpu_attention_recovery', False):
            from .cpu_attention import primitive_probe
            primitive_probe(out / 'primitives.json')
        mod, models = load(cfg, manifest, args.upstream, out / "reports")
        report["models"] = models
        if getattr(args, "target_fp32_preflight", False) or getattr(args, "target_fp64_reference", False):
            from .live_precision import run_live_preflight
            binding = read_jsonl(data / "binding.jsonl")
            if digest(binding) != manifest["binding_hash"]:
                raise ValueError("frozen binding data hash mismatch")
            if getattr(args, 'cpu_attention_recovery', False):
                from .cpu_recovery import run_recovery
                return run_recovery(mod, natural, binding, cfg, report, out,
                                    read_json(args.reference_audit_report))
            return run_live_preflight(mod, natural, binding, cfg, report, out,
                                      reference=getattr(args, "target_fp64_reference", False),
                                      audit_prior=read_json(args.reference_audit_report)
                                      if getattr(args, 'reference_audit_report', None) else None,
                                      gpu_reference=getattr(args, 'gpu_attention_reference', False))
        if getattr(args, "target_fp32_probe", False):
            if selected_calibration is None:
                raise ValueError("target FP32 probe requires an exact calibration prompt ID")
            from .precision import run_precision
            return run_precision(mod, rows, selected_calibration, cfg, report, out)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Reproduce the same generation warmup and gate before independent probes.
            report["baseline"] = check_baseline(mod, [r["prompt"] for r in rows], cfg,
                                                math_backend=math_gate)
            print("Baseline:", report["baseline"]["status"], report["baseline"].get("error", ""), flush=True)
            write_json(out / "diagnostic.json", report)
            if math_gate:
                passed = report["baseline"]["status"] == "passed"
                report["status"] = "complete" if passed else "failed"
                for index, checks in enumerate(report["baseline"].get("reports", [])):
                    print(f"prompt={index} checks={checks}", flush=True)
                if not passed or args.math_baseline_only:
                    print("Math baseline passed; endpoint validation still pending." if passed
                          else "Math baseline failed; do not start the S1 experiment.", flush=True)
                    return 0 if passed else 2
                report.update(status="running", endpoints=[])
                for row in ([selected_calibration] if calibration_id else preflight_rows(natural)):
                    item = {"prompt_id": row["prompt_id"], "domain": row["domain"], "status": "running"}
                    if calibration_id:
                        item["probability_audit"] = {}
                        item["alignment_audit"] = {}
                    item["checks"] = {}
                    report["endpoints"].append(item)
                    write_json(out / "diagnostic.json", report)
                    print(f"Endpoint starting: {row['domain']} {row['prompt_id']}", flush=True)
                    item.update(check_endpoint(mod, row, cfg, report["baseline"]["tolerance"],
                                               audit=item.get("probability_audit"), checks_out=item["checks"],
                                               alignment=item.get("alignment_audit")))
                    item["status"] = "passed"
                    print(f"Endpoint passed: {row['domain']} checks={item['checks']}", flush=True)
                    write_json(out / "diagnostic.json", report)
                report["status"] = "complete"
                print("Exact calibration probe passed; full calibration and S1 still pending." if calibration_id
                      else "Math preflight passed (4 natural endpoints); full S1 experiment not run.", flush=True)
                return 0
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
        if "precision_probe" in report:
            report["precision_probe"].update(status="failed", error_type=type(exc).__name__)
        if isinstance(exc, ValidationError):
            report["error"] = str(exc)
            print(f"Validation error: {exc}", flush=True)
        elif isinstance(exc, ValueError):
            from .run import safe_error
            report["error"] = safe_error(exc)
            print(f"Diagnostic error: {report['error']}", flush=True)
        for item in report.get("endpoints", []):
            if item["status"] == "running":
                item["status"] = "failed"
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
    parser.add_argument('--reference-audit-report',
                        help='Prior failed FP64 preflight summary: audit its failures plus one control; no full preflight')
    parser.add_argument('--cpu-attention-recovery', action='store_true',
                        help='Separate CPU-attention reference: primitive check, failed subset, then full preflight')
    parser.add_argument('--gpu-attention-reference', action='store_true',
                        help='FP64 exp/sum GPU attention reference: CPU crosschecks and all frozen preflight cases')
    parser.add_argument("--target-fp32-probe", action="store_true",
                        help="Diagnostic only: capture native endpoint, then rebuild Target forwards/KV in FP32")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--target-fp64-reference", action="store_true",
                      help="Slow FP64 live Target reference with bounded weight chunks; no S1 effects")
    mode.add_argument("--target-fp32-preflight", action="store_true",
                      help="FP32 Target throughout live generation; all frozen endpoint checks, no S1 effects")
    mode.add_argument("--math-baseline-only", action="store_true",
                        help="Run the complete baseline under math SDPA, then stop before probes/endpoints")
    mode.add_argument("--math-preflight", action="store_true",
                      help="Math baseline plus four natural snapshot validation cases; no S1 effect measurements")
    mode.add_argument("--calibration-prompt-id",
                      help="Math baseline plus one exact calibration snapshot and FP32/FP64 probability audit")
    args = parser.parse_args()
    if args.reference_audit_report and not args.target_fp64_reference:
        parser.error('--reference-audit-report requires --target-fp64-reference')
    if args.cpu_attention_recovery and not (args.target_fp64_reference and args.reference_audit_report):
        parser.error('--cpu-attention-recovery requires --target-fp64-reference and --reference-audit-report')
    if args.gpu_attention_reference and (not args.target_fp64_reference or args.cpu_attention_recovery
                                         or args.reference_audit_report):
        parser.error('--gpu-attention-reference requires FP64 reference without CPU recovery or subset audit')
    if args.target_fp32_probe and not args.calibration_prompt_id:
        parser.error("--target-fp32-probe requires --calibration-prompt-id")
    raise SystemExit(main(args))

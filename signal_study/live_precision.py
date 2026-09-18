"""Separate preflight policies: FP32/FP64 Target real SD generation, BF16 Drafter.

This is not official_eval reproduction and produces no S1 effect measurements.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .capture import capture_prompt, draft_logits, synthetic_snapshot, target_logits
from .common import digest, write_json
from .diagnose import check_baseline, compare, snapshot_structure
from .state import preserved_rng
from .validation import ValidationError, require_error, validate_snapshot


POLICY = {
    "id": "target-fp32-live-preflight-v1", "scope": "preflight_only_not_S1_results",
    "target": "stored weights promoted to FP32; decoder/head autocast disabled",
    "target_cache": "FP32 from actual generation, never converted from captured BF16 KV",
    "drafter_and_guidance": "existing BF16 weights/autocast retained",
    "sdpa": "math", "tf32": False,
    "tolerances": "unchanged caps/rule; recomputed from FP32 baseline before endpoints",
    "note": "Different numerical Target from official_eval; regenerated trajectories may differ.",
}


def cache_dtypes(cache):
    return sorted({str(t.dtype) for pair in cache["layers"] for t in pair})


@contextmanager
def live_target_precision(mod, *, target_dtype=torch.float32):
    """Scoped wrappers, not observation hooks, so Capture's identity test remains meaningful."""
    decoder, head, guide = mod.v_base.get_decoder(), mod.v_base.lm_head, mod.guidance_embd_layer
    device_type = next(mod.v_base.parameters()).device.type
    # Guide is registered under Target too; keep its original precision independently.
    tensors = list(mod.v_base.parameters()) + list(mod.v_base.buffers())
    dtypes = [(t, t.dtype) for t in tensors if t.is_floating_point()]
    guide_ids = {id(t) for t in list(guide.parameters()) + list(guide.buffers())}
    old_flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    originals = [(obj, "forward" in obj.__dict__, obj.forward) for obj in (decoder, head, guide)]
    audit = {"target_decoder_calls": 0, "target_head_calls": 0, "guide_calls": 0,
             "checked_fp32_cache_calls": 0, "checked_target_cache_calls": 0,
             "target_dtype": str(target_dtype)}
    decoder_forward, head_forward, guide_forward = (item[2] for item in originals)

    def target_decoder(*args, **kwargs):
        with torch.autocast(device_type, enabled=False):
            result = decoder_forward(*args, **kwargs)
        out = result["out"] if "out" in result else result
        if out.last_hidden_state.dtype != target_dtype:
            raise ValidationError(f"live Target decoder hidden is not {target_dtype}")
        cache = out.past_key_values
        if cache is not None:
            if any(t.dtype != target_dtype for pair in zip(cache.key_cache, cache.value_cache) for t in pair):
                raise ValidationError(f"live Target KV is not {target_dtype}")
            audit["checked_target_cache_calls"] += 1
            audit["checked_fp32_cache_calls"] += int(target_dtype == torch.float32)
        audit["target_decoder_calls"] += 1
        return result

    def target_head(hidden, *args, **kwargs):
        if hidden.dtype != target_dtype:
            raise ValidationError(f"live Target head input is not {target_dtype}")
        with torch.autocast(device_type, enabled=False):
            result = head_forward(hidden, *args, **kwargs)
        if result.dtype != target_dtype:
            raise ValidationError(f"live Target logits are not {target_dtype}")
        audit["target_head_calls"] += 1
        return result

    def guidance(*args, **kwargs):
        # Decoder disables autocast, but its attached learned guide remains BF16.
        # Autocast does not downcast FP64 inputs automatically.
        if args and args[0].dtype == torch.float64:
            args = (args[0].float(), *args[1:])
        with torch.autocast(device_type, dtype=torch.bfloat16):
            result = guide_forward(*args, **kwargs)
        audit["guide_calls"] += 1
        return result

    try:
        with torch.no_grad():
            for tensor, _ in dtypes:
                if id(tensor) not in guide_ids:
                    tensor.data = tensor.data.float()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        decoder.forward, head.forward, guide.forward = target_decoder, target_head, guidance
        with torch.autocast(device_type, dtype=torch.bfloat16), sdpa_kernel(SDPBackend.MATH):
            yield audit
    finally:
        for obj, had_override, original in originals:
            if had_override:
                obj.forward = original
            else:
                obj.__dict__.pop("forward", None)
        with torch.no_grad():
            for tensor, dtype in dtypes:
                tensor.data = tensor.data.to(dtype)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_flags


# Existing tests/callers keep the FP32-only default behavior.
live_target_fp32 = live_target_precision


def selected_rows(natural, binding, cfg):
    calibration = [r for r in natural if r["split"] == "calibration"]
    smoke = [r for r in natural if r["split"] == "smoke"]
    if (len(calibration) != cfg["calibration_count"] or len(smoke) != 4 * cfg["smoke_per_domain"]
            or len(binding) != 2 * cfg["binding_pairs"]):
        raise ValueError("live preflight requires all frozen calibration/smoke/binding cases")
    return calibration + smoke + binding, smoke[:2]


def inspect_snapshot(mod, snap, tolerance, item, *, target_dtype=torch.float32):
    item["prefix_hash"] = snap["prefix_hash"]
    item["structure"] = snapshot_structure(snap)
    item["target_kv_dtypes"] = cache_dtypes(snap["v_cache"])
    item["draft_kv_dtypes"] = cache_dtypes(snap["d_cache"])
    item["source_logits_dtype"] = str(snap["p_src_logits"].dtype)
    item["draft_logits_dtype"] = str(snap["q_original_logits"].dtype)
    if item["target_kv_dtypes"] != [str(target_dtype)] or item["source_logits_dtype"] != str(target_dtype):
        raise ValidationError(f"captured live Target cache/logits must be {target_dtype}")
    # Record both discrepancies even if the first production gate subsequently fails.
    item["target_alignment"] = compare(target_logits(mod, snap), target_logits(mod, snap, cached=True))
    validate_snapshot(mod, snap, tolerance, diagnostics=item["checks"])
    draft_logits(mod, snap, -snap["G"])
    item["checks"]["A_B_A"] = require_error(
        snap["q_original_logits"], draft_logits(mod, snap), tolerance["repeat"], "live preflight A-B-A")


@torch.inference_mode()
def run_live_preflight(mod, natural, binding, cfg, report, out, *, reference=False, audit_prior=None,
                       cpu_reference=False, gpu_reference=False):
    if gpu_reference and (not reference or cpu_reference or audit_prior is not None):
        raise ValueError('GPU exp/sum reference requires FP64 full preflight without another attention override')
    if reference:
        from .reference_precision import POLICY as policy, reference_operators
    else:
        policy = POLICY
    if cpu_reference:
        if not reference:
            raise ValueError('CPU attention requires the FP64 reference policy')
        from .cpu_attention import POLICY as policy, cpu_attention
    if gpu_reference:
        from .gpu_attention import POLICY as policy, gpu_attention
    target_dtype = torch.float64 if reference else torch.float32
    label = "FP64 reference" if reference else "FP32"
    rows, baseline_rows = selected_rows(natural, binding, cfg)
    if audit_prior is not None:
        if not reference:
            raise ValueError('operator audit requires FP64 reference')
        from .reference_audit import audit_rows, audit_attention, probe_snapshot
        rows = audit_rows(rows, audit_prior)
        label = 'FP64 reference audit (subset)'
    report.update(scope=policy["scope"], live_precision_policy=policy,
                  baseline_sdpa='cpu-explicit-fp64' if cpu_reference else 'math',
                  note="Preflight-only numerical policy override; config/models describe original loading policy.",
                  endpoints=[], status="running", stage="baseline")
    if audit_prior is not None:
        report.update(scope='reference_failure_audit_only_not_full_preflight',
                      audit_scope='all previous failures plus first nonfailed frozen control; same baseline',
                      attention_audit={})
    if cpu_reference:
        label = 'FP64 CPU-attention reference' + (' (subset)' if audit_prior is not None else '')
        report['cpu_attention'] = {}
    if gpu_reference:
        label = 'FP64 GPU exp/sum reference'
        report.update(baseline_sdpa='gpu-exp-sum-fp64', gpu_attention={})
    device_type = next(mod.v_base.parameters()).device.type
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def save():
        counts = dict(Counter(item["status"] for item in report["endpoints"]))
        summary = {"scope": report["scope"], "status": report["status"], "stage": report["stage"],
                   "policy": policy["id"], "planned": len(rows), "counts": counts,
                   "baseline": report.get("baseline"),
                   "error_type": report.get("error_type"), "error": report.get("error"),
                   "failures": [{k: item[k] for k in ("prompt_id", "domain", "status", "error_type", "error",
                                  "target_alignment", "target_kv_dtypes", "draft_kv_dtypes") if k in item}
                                for item in report["endpoints"] if item["status"] != "passed"],
                   "max_target_TV": max((i["target_alignment"]["TV"] for i in report["endpoints"]
                                         if "target_alignment" in i), default=None),
                   "dtype_audit": report.get("dtype_audit"), "reference_arithmetic": report.get("reference_arithmetic"),
                   "gpu_memory": report.get("gpu_memory")}
        if cpu_reference:
            summary['cpu_attention'] = report['cpu_attention']
        if gpu_reference:
            summary['gpu_attention'] = report['gpu_attention']
        if audit_prior is not None:
            summary.update(audit_scope=report['audit_scope'], attention_audit=report['attention_audit'],
                           audit_cases=report['endpoints'])
        write_json(out / "preflight-summary.json", summary)
        write_json(out / "diagnostic.json", report)

    try:
        with ExitStack() as stack:
            if reference:
                report["reference_arithmetic"] = stack.enter_context(reference_operators(mod))
            if cpu_reference:
                stack.enter_context(cpu_attention(report['cpu_attention']))
            if gpu_reference:
                stack.enter_context(gpu_attention(report['gpu_attention']))
            if audit_prior is not None and not cpu_reference:
                stack.enter_context(audit_attention(report['attention_audit']))
                report['attention_audit']['phase'] = 'baseline'
            audit = stack.enter_context(live_target_precision(mod, target_dtype=target_dtype))
            report["dtype_audit"] = audit
            report["baseline"] = check_baseline(mod, [r["prompt"] for r in baseline_rows], cfg, math_backend=True)
            print(f"Live {label} baseline:", report["baseline"], flush=True)
            if report["baseline"]["status"] != "passed":
                report["status"] = "failed"
                return 2
            tolerance = report["baseline"]["tolerance"]
            report["stage"] = "endpoints"
            save()
            for index, row in enumerate(rows):
                if audit_prior is not None:
                    report['attention_audit']['phase'] = row['prompt_id'] + ':capture'
                item = {"prompt_id": row["prompt_id"], "domain": row["domain"], "split": row["split"],
                        "status": "running", "checks": {}}
                report["endpoints"].append(item)
                save()
                snap = None
                try:
                    with preserved_rng():
                        torch.manual_seed(cfg["seed"])
                        snap = (synthetic_snapshot(mod, row) if row["domain"] == "binding"
                                else capture_prompt(mod, row, cfg["max_new_tokens"])[0])
                    if snap is None:
                        raise ValidationError("no normal boundary; not a passed endpoint")
                    if audit_prior is not None:
                        report['attention_audit']['phase'] = row['prompt_id'] + ':validation'
                    inspect_snapshot(mod, snap, tolerance, item, target_dtype=target_dtype)
                    item["status"] = "passed"
                except (ValidationError, ValueError) as exc:
                    from .run import safe_error
                    item.update(status="failed", error_type=type(exc).__name__, error=safe_error(exc))
                    # Collect validation failures without fitting tolerances to these cases.
                finally:
                    if audit_prior is not None and snap is not None:
                        report['attention_audit']['phase'] = row['prompt_id'] + ':controls'
                        try:
                            item['alignment_probe'] = probe_snapshot(mod, snap)
                            if cpu_reference:
                                for name in ('fresh_repeat', 'cached_repeat'):
                                    if item['alignment_probe'][name]['max_logit_error'] > tolerance['repeat']:
                                        raise ValidationError('CPU reference repeat control failed')
                        except Exception as exc:
                            from .run import safe_error
                            item['alignment_probe'] = dict(status='failed', error=safe_error(exc))
                            item['status'] = 'failed'
                    del snap
                print(f"Live {label} {index + 1}/{len(rows)} {row['domain']} {item['status']} "
                      f"alignment={item.get('target_alignment')} error={item.get('error', '')}", flush=True)
                save()
            report["stage"] = "finished"
            report["status"] = "passed" if all(i["status"] == "passed" for i in report["endpoints"]) else "failed"
            if audit_prior is not None and not cpu_reference and report['attention_audit']['event_count']:
                report.update(status='failed', error='attention operator disagreement or nonfinite/repeat error detected')
            print(f"Live {label} preflight {report['status']}; counts="
                  f"{dict(Counter(i['status'] for i in report['endpoints']))}. NOT a completed S1 experiment.", flush=True)
            return 0 if report["status"] == "passed" else 2
    except Exception as exc:
        from .run import safe_error
        report.update(status="failed", error_type=type(exc).__name__, error=safe_error(exc))
        for item in report["endpoints"]:
            if item["status"] == "running":
                item.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        if device_type == "cuda":
            report["gpu_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        save()

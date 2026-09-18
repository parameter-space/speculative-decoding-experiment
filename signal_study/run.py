from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import sys
import socket
from collections import Counter
from contextlib import contextmanager, ExitStack

import torch

from .capture import capture_prompt, draft_logits, source_metadata, synthetic_snapshot
from .common import append_jsonl, choose_donor, digest, file_digest, read_json, read_jsonl, validate_splits, write_json
from .runtime import environment, load, sdpa_context
from .parallel import partition_rows
from .state import METRIC_POLICY, distribution, overlap, preserved_rng
from .validation import ValidationError, baseline_tests, require_error, validate_snapshot


CONDITIONS = ("original", "self_copy", "mean", "mean_rms", "matched_donor")


def execution_policy(reference=False):
    if not reference:
        return {"id": "s1-official-eval-v1", "scope": "S1_endpoint_effect_measurement",
                "precision": "official_eval"}
    from .gpu_attention import POLICY
    return dict(POLICY, id="s1-target-fp64-exp-sum-v1", reference_policy_id=POLICY["id"],
                scope="S1_endpoint_effect_measurement", precision="target_FP64_drafter_BF16",
                note="Separate numerical-reference S1; not official_eval reproduction or a speed benchmark.")


@contextmanager
def execution_context(mod, reference, tests):
    with ExitStack() as stack:
        if reference:
            from .reference_precision import reference_operators
            from .gpu_attention import gpu_attention
            from .live_precision import live_target_precision
            tests["reference_arithmetic"] = stack.enter_context(reference_operators(mod))
            tests["gpu_attention"] = {}
            stack.enter_context(gpu_attention(tests["gpu_attention"]))
            tests["dtype_audit"] = stack.enter_context(live_target_precision(mod, target_dtype=torch.float64))
        yield


def safe_error(exc):
    # Never include raw HTTP exceptions that may contain signed download URLs or credentials.
    if isinstance(exc, (ValidationError, ValueError, FileExistsError)):
        return re.sub(r"https?://\S+|hf_[A-Za-z0-9]+", "[redacted]", str(exc))[:600]
    return f"{type(exc).__name__}: inspect locally; remote URL/credential details omitted"


def top5(tokenizer, p):
    values, indices = p.topk(min(5, len(p)))
    return [{"id": int(i), "text": tokenizer.decode([int(i)]), "probability": float(v)} for v, i in zip(values, indices)]


def case_rows(mod, row, snap, p, tests, tolerance, mean=None, pool=(), donor_g=None, partner=None):
    meta = source_metadata(snap, row)
    q_orig = distribution(snap["q_original_logits"], context="endpoint.drafter.original")
    a_orig = overlap(p, q_orig)
    choices = {"original": snap["G"], "self_copy": snap["G"].clone()}
    donor = None
    if partner is not None:
        choices["binding_swap"] = donor_g
    else:
        choices["mean"] = mean
        mean_rms = float(mean.float().square().mean().sqrt()) if mean is not None else 0
        recipient_rms = snap["G"].float().square().mean().sqrt()
        choices["mean_rms"] = (mean.float() * (recipient_rms / mean_rms)).to(snap["G"].dtype) if mean_rms > 1e-12 else None
        donor = choose_donor(meta, pool)
        choices["matched_donor"] = donor["G"] if donor else None
    rows, details = [], []
    for condition, guide in choices.items():
        result = {**meta, "model": "SD2", "scope": "endpoint", "port": "G_pre_LayerNorm", "depth": 1,
                  "base_problem_id": row.get("base_problem_id"), "state_origin": snap["state_origin"],
                  "prefix_hash": snap["prefix_hash"], "p_prefix_hash": snap["prefix_hash"],
                  "q_prefix_hash": snap["prefix_hash"], "receiver_pos": len(snap["prefix"]) - 1,
                  "condition": condition, "pending_tokens": snap["pending_ids"].shape[1],
                  "d_cache_length": snap["d_cache"]["length"], "v_cache_length": snap["v_cache"]["length"],
                  "status": "ok", "skip_reason": None, "extra_target_calls_per_snapshot": 2,
                  "donor_prompt_id": donor["prompt_id"] if condition == "matched_donor" and donor else None}
        if guide is None:
            result.update(status="NA", skip_reason="no_eligible_calibration_donor" if condition == "matched_donor" else "mean_RMS_at_or_below_1e-12")
            rows.append(result)
            continue
        logits = draft_logits(mod, snap, guide)
        if condition in ("original", "self_copy"):
            require_error(snap["q_original_logits"], logits, tolerance["repeat"], condition)
        q = distribution(logits, context=f"endpoint.drafter.{condition}")
        a = overlap(p, q)
        result.update(A=a, delta_A_vs_original=a - a_orig, U_original_over_control=a_orig - a,
                      signal_rms=float(guide.float().square().mean().sqrt()))
        if partner is not None:
            label = partner["label_id"]
            result.update(donor_label=partner["label"], donor_label_id=label,
                          recipient_label=row["label"], recipient_label_id=row["label_id"],
                          donor_label_probability=float(q[label]),
                          donor_label_probability_change=float(q[label] - q_orig[label]),
                          donor_prompt_id=partner["prompt_id"] if condition == "binding_swap" else None)
        details.append({"condition": condition, "A": a, "top5": top5(mod.tok, q)})
        rows.append(result)
    after = draft_logits(mod, snap)
    tests["A_B_A"] = require_error(snap["q_original_logits"], after, tolerance["repeat"], "A-B-A")
    for item in rows:
        item["max_logit_diff_sham"] = max(tests["restore"], tests["self_copy"], tests["A_B_A"])
    return rows, {"prompt_id": row["prompt_id"], "prefix_tail": mod.tok.decode(snap["prefix"][-120:].tolist()),
                  "target_top5": top5(mod.tok, p), "conditions": details, "tests": tests}


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    cfg, out, data = read_json(args.config), Path(args.output), Path(args.data_dir)
    reference = getattr(args, "target_fp64_reference", False)
    policy = execution_policy(reference)
    if reference and (getattr(args, "shard_index", 0) != 0 or getattr(args, "shard_count", 1) != 1):
        raise ValueError("FP64 S1 reference requires a single unsharded worker")
    if cfg["depth"] != 1 or cfg["batch_size"] != 1 or cfg["precision"] != "official_eval":
        raise ValueError("requires depth1, batch1, official_eval loading config; precision override is explicit")
    if out.exists():
        raise FileExistsError("output exists; use a new run directory")
    for sub in ("manifests", "trace", "results", "reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    results, cases, tests = [], [], {"status": "running", "stage": "environment", "cases": [],
                                    "metric_policy": METRIC_POLICY, "execution_policy": policy}
    write_json(out / "manifests/execution_policy.json", policy)

    def checkpoint():
        # Interrupted runs must never leave apparently publishable numeric rows.
        for result in results:
            result.update(execution_policy_id=policy["id"], run_valid=False)
        tests.update(stage=stage)
        write_json(out / "reports/tests.json", tests)
        write_csv(out / "results/S1_endpoint.csv", results)

    status, stage = "failed", "environment"
    try:
        if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0] != "ariel-k2":
            raise ValueError("real checkpoint runs require an allocated Slurm compute job")
        sdpa_policy = "math" if reference else os.environ.get("S1_SDPA_BACKEND", "default")
        attention_context = sdpa_context(sdpa_policy)
        tests["sdpa_kernel_policy"] = sdpa_policy
        tests["target_attention_policy"] = "gpu-exp-sum-fp64" if reference else sdpa_policy
        write_json(out / "manifests/environment.json", environment())
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("real smoke requires exactly one visible allocated CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is unavailable")
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(cfg["seed"])
        dataset_manifest = read_json(data / "manifest.json")
        if dataset_manifest["config_hash"] != digest(cfg):
            raise ValueError("config differs from the frozen data manifest")
        natural, binding = read_jsonl(data / "natural.jsonl"), read_jsonl(data / "binding.jsonl")
        validate_splits(natural + binding)
        if digest(natural) != dataset_manifest["natural_hash"] or digest(binding) != dataset_manifest["binding_hash"]:
            raise ValueError("dataset hash mismatch")
        write_json(out / "manifests/data.json", dataset_manifest)
        write_json(out / "manifests/run.json", cfg)
        write_json(out / "manifests/metric_policy.json", METRIC_POLICY)
        write_json(out / "manifests/implementation.json", {str(p.relative_to(Path(__file__).parent)): file_digest(p)
                   for p in sorted(Path(__file__).parent.glob("*.py"))})
        stage = "model_load"
        mod, models = load(cfg, dataset_manifest, args.upstream, out / "reports")
        models["sdpa_kernel_policy"] = sdpa_policy
        models["metric_policy"] = METRIC_POLICY
        models["execution_policy"] = policy
        if reference:
            models["loading_precision"] = models.pop("precision", None)
            models["precision"] = {"target_storage": "float32 (original loaded values)",
                                   "target_compute_and_KV": "float64",
                                   "drafter_and_guidance": "bfloat16", "metric_softmax": "CPU float64",
                                   "note": policy["note"]}
        models["sdpa_policy_scope"] = ("Drafter math SDPA; Target GPU FP64 exp/sum override for all forwards"
                                       if reference else "all baseline/calibration/generation/endpoint forward calls")
        write_json(out / "manifests/models.json", models)
        evaluation = [r for r in natural if r["split"] == "smoke"]
        calibration = [r for r in natural if r["split"] == "calibration"]
        if (len(calibration) != cfg["calibration_count"] or len(evaluation) != 4 * cfg["smoke_per_domain"]
                or len(binding) != 2 * cfg["binding_pairs"]):
            raise ValueError("unexpected smoke manifest counts")
        baseline_prompts = [r["prompt"] for r in evaluation[:2]]
        shard_index, shard_count = getattr(args, "shard_index", 0), getattr(args, "shard_count", 1)
        evaluation, binding = partition_rows(evaluation, binding, shard_index, shard_count)
        tests["shard"] = {"index": shard_index, "count": shard_count}
        write_json(out / "manifests/shard.json", {"index": shard_index, "count": shard_count,
                   "evaluation_prompt_ids": [r["prompt_id"] for r in evaluation],
                   "binding_prompt_ids": [r["prompt_id"] for r in binding]})
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), attention_context, \
                execution_context(mod, reference, tests):
            stage = "baseline"
            print(f"S1 measurement starting: {policy['id']}", flush=True)
            tests["baseline"], tolerance = baseline_tests(mod, baseline_prompts, cfg)
            tests["tolerance"] = tolerance
            print(f"Baseline passed; frozen tolerance: {tolerance}", flush=True)
            write_json(out / "reports/tests.json", tests)
            stage = "calibration"
            pool = []
            for number, row in enumerate(calibration, 1):
                print(f"Calibration {number}/{len(calibration)}: {row['prompt_id']}", flush=True)
                checkpoint()
                with preserved_rng():
                    torch.manual_seed(cfg["seed"])
                    snap, tensor_map = capture_prompt(mod, row, cfg["max_new_tokens"])
                if snap is None:
                    tests["cases"].append({"prompt_id": row["prompt_id"], "split": "calibration", "status": "NA", "reason": "no_valid_normal_boundary"})
                    continue
                _, checks = validate_snapshot(mod, snap, tolerance)
                pool.append({**source_metadata(snap, row), "G": snap["G"]})
                tests["cases"].append({"prompt_id": row["prompt_id"], "split": "calibration", "status": "ok", "checks": checks})
                write_json(out / "tensor_map.json", tensor_map)
                del snap
            if not pool:
                raise ValidationError("no validated calibration signal")
            mean = torch.stack([r["G"].float() for r in pool]).mean(0).to(pool[0]["G"].dtype)
            write_json(out / "manifests/calibration.json", {"requested": len(calibration), "valid": len(pool),
                      "prompt_ids": [r["prompt_id"] for r in pool], "scope": "UltraChat train_sft only"})
            stage = "natural_endpoints"
            for number, row in enumerate(evaluation, 1):
                print(f"Natural effects {number}/{len(evaluation)}: {row['prompt_id']}", flush=True)
                checkpoint()
                with preserved_rng():
                    torch.manual_seed(cfg["seed"])
                    snap, _ = capture_prompt(mod, row, cfg["max_new_tokens"])
                if snap is None:
                    for condition in CONDITIONS:
                        results.append({"prompt_id": row["prompt_id"], "condition": condition, "status": "NA", "skip_reason": "no_valid_normal_boundary"})
                    tests["cases"].append({"prompt_id": row["prompt_id"], "status": "NA", "reason": "no_valid_normal_boundary"})
                    continue
                p, checks = validate_snapshot(mod, snap, tolerance)
                rows, detail = case_rows(mod, row, snap, p, checks, tolerance, mean, pool)
                results.extend(rows)
                cases.append(detail)
                save_boundary(out, row, snap, checks, args.save_snapshots)
                tests["cases"].append({"prompt_id": row["prompt_id"], "status": "ok", "checks": checks})
                del snap
                checkpoint()
            stage = "binding_endpoints"
            grouped = {}
            for row in binding:
                grouped.setdefault(row["base_problem_id"], []).append(row)
            for pair in grouped.values():
                if len(pair) != 2:
                    raise ValueError("binding pair is incomplete")
                # Two small CPU snapshots only; no persistent collection of GPU caches.
                snaps = [synthetic_snapshot(mod, row) for row in pair]
                for index, row in enumerate(pair):
                    print(f"Binding effects: {row['prompt_id']}", flush=True)
                    checkpoint()
                    snap, partner = snaps[index], pair[1 - index]
                    p, checks = validate_snapshot(mod, snap, tolerance)
                    rows, detail = case_rows(mod, row, snap, p, checks, tolerance,
                                           donor_g=snaps[1 - index]["G"], partner=partner)
                    results.extend(rows)
                    cases.append(detail)
                    save_boundary(out, row, snap, checks, args.save_snapshots)
                    tests["cases"].append({"prompt_id": row["prompt_id"], "status": "ok", "checks": checks})
                del snaps
                checkpoint()
        tests["peak_memory"] = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved(),
                                "scope": "post-load baseline/calibration/S1 effects, not throughput"}
        status = "complete" if len(cases) == len(evaluation) + len(binding) and len(pool) == len(calibration) else "partial"
    except Exception as exc:
        tests["error"] = safe_error(exc)
        # Numeric rows obtained before a fatal invariant failure are NOT publishable S1 results.
        for result in results:
            result["run_valid"] = False
        tests["failed_prompt_id"] = locals().get("row", {}).get("prompt_id")
        if locals().get("row"):
            failed_row = locals()["row"]
            results.append({"prompt_id": failed_row["prompt_id"], "condition": "validation_gate",
                            "status": "failed", "skip_reason": safe_error(exc), "run_valid": False})
        print(f"Stopped at {stage}: {safe_error(exc)}", file=sys.stderr)
    finally:
        for result in results:
            result["run_valid"] = status == "complete"
            result["checkpoint_revision"] = cfg["checkpoint_revision"]
            result["metric_policy_id"] = METRIC_POLICY["id"]
            result["execution_policy_id"] = policy["id"]
        tests.update(status=status, stage=stage)
        write_json(out / "reports/tests.json", tests)
        write_csv(out / "results/S1_endpoint.csv", results)
        write_json(out / "results/cases.json", cases)
        lines = ["# S1 endpoint cases", "", f"Run status: {status}", f"Execution policy: {policy['id']}", "",
                 "Endpoint sensitivity only; not removal of all target information, full reasoning validation, or a speed result.", ""]
        for detail in cases:
            lines.extend([f"## {detail['prompt_id']}", "", f"Prefix tail: {detail['prefix_tail']}", "",
                          f"Target top5: {detail['target_top5']}", "",
                          f"Conditions: {detail['conditions']}", "", f"Checks: {detail['tests']}", ""])
        incomplete = [r for r in results if r["status"] != "ok"]
        if incomplete:
            lines.extend(["## Failed / unavailable conditions", "", *[str(r) for r in incomplete], ""])
        (out / "results/S1_cases.md").write_text("\n".join(lines), encoding="utf-8")
        (out / "reports/HANDOFF.md").write_text(
            f"# S1 run handoff\n\nStatus: **{status}**. Stage: {stage}.\n\n"
            f"Execution policy: {policy['id']}. Precision: {policy['precision']}.\n\n"
            f"Completed endpoint cases: {len(cases)}. Row statuses: {dict(Counter(r['status'] for r in results))}.\n\n"
            f"Error: {tests.get('error', 'none')}. All identity/alignment evidence is in tests.json.\n\n"
            "No model training, S2/EAGLE implementation, or speed claim. If failed, numeric rows are not valid research results. "
            "If complete, inspect individual cases before choosing S2 or EAGLE endpoint replication.\n", encoding="utf-8")
        print(f"S1 measurement {status}: {len(cases)} endpoint cases; results: {out / 'results/S1_endpoint.csv'}", flush=True)
    return 0 if status == "complete" else 2


def save_boundary(out, row, snap, checks, snapshots):
    boundary_id = digest([row["prompt_id"], snap["prefix_hash"]])[:20]
    append_jsonl(out / "trace/boundaries.jsonl", {"boundary_id": boundary_id, "prompt_id": row["prompt_id"],
       "state_origin": snap["state_origin"], "prefix_ids": snap["prefix"].tolist(), "prefix_hash": snap["prefix_hash"],
       "source_pos": snap["source_pos"], "source_physical_pos": snap["source_physical_pos"], "z_pos": len(snap["prefix"]) - 1,
       "pending_ids": snap["pending_ids"].tolist(), "pending_positions": snap["pending_positions"].tolist(),
       "physical_mask": snap["pending_mask"].tolist(), "d_cache_length": snap["d_cache"]["length"],
       "v_cache_length": snap["v_cache"]["length"], "checks": checks})
    if snapshots:
        # Opt-in: full CPU snapshots cost several GiB over a run; never serialize models.
        torch.save(snap, out / "trace" / f"{boundary_id}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--upstream", default="vendor/SD-square")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--target-fp64-reference", action="store_true",
                        help="Explicit S1 FP64 Target/GPU exp-sum reference; not official_eval reproduction")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    raise SystemExit(run(parser.parse_args()))

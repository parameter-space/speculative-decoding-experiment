"""Independent single-GPU workers over disjoint endpoint cases, not tensor parallelism."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

from .common import append_jsonl, digest, read_json, read_jsonl, write_json


def partition_rows(evaluation, binding, index, count):
    if count not in (1, 2) or not 0 <= index < count:
        raise ValueError("supported shard count is 1 or 2, with a valid shard index")
    problem_ids = sorted({r["base_problem_id"] for r in binding})
    selected = set(problem_ids[index::count])
    return evaluation[index::count], [r for r in binding if r["base_problem_id"] in selected]


def worker_devices(visible, count):
    if count not in (1, 2):
        raise ValueError("exactly one or two allocated GPUs are required")
    if not visible:
        raise ValueError("Slurm CUDA_VISIBLE_DEVICES is required; do not guess physical GPU IDs")
    identifiers = [x.strip() for x in visible.split(",")]
    if len(identifiers) != count or any(not x for x in identifiers) or len(set(identifiers)) != count:
        raise ValueError("CUDA device mask and visible count disagree")
    return identifiers


def merge_outputs(root, count, codes):
    root = Path(root)
    rows, cases, reports, manifests, boundaries = [], [], [], [], []
    for index in range(count):
        folder = root / f"worker-{index}"
        if not (folder / "reports/tests.json").is_file():
            raise ValueError(f"worker {index} has no test report; inspect its stderr locally")
        report = read_json(folder / "reports/tests.json")
        reports.append({"shard_index": index, "exit_code": codes[index], "report": report})
        for path in ("manifests/run.json", "manifests/data.json", "manifests/models.json", "manifests/calibration.json"):
            if not (folder / path).is_file():
                raise ValueError(f"worker {index} incomplete: {path}")
        manifests.append([read_json(folder / p) for p in ("manifests/run.json", "manifests/data.json", "manifests/models.json", "manifests/calibration.json")])
        with (folder / "results/S1_endpoint.csv").open(encoding="utf-8", newline="") as stream:
            rows.extend(dict(row, shard_index=index) for row in csv.DictReader(stream))
        cases.extend(read_json(folder / "results/cases.json"))
        if (folder / "trace/boundaries.jsonl").exists():
            boundaries.extend(read_jsonl(folder / "trace/boundaries.jsonl"))
    if any(digest(item) != digest(manifests[0]) for item in manifests[1:]):
        raise ValueError("workers used different config/data/model/calibration manifests")
    keys = [(r["prompt_id"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate endpoint-condition rows across workers")
    if len({c["prompt_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate completed cases across workers")
    cfg = manifests[0][0]
    expected = 4 * cfg["smoke_per_domain"] + 2 * cfg["binding_pairs"]
    complete = all(code == 0 for code in codes) and all(r["report"]["status"] == "complete" for r in reports) and len(cases) == expected
    status = "complete" if complete else "incomplete"
    for row in rows:
        if not complete:
            row["run_valid"] = False
    (root / "results").mkdir(exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["status"]
    with (root / "results/S1_endpoint.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "results/cases.json", cases)
    write_json(root / "reports/tests.json", {"status": status, "workers": reports,
               "completed_cases": len(cases), "expected_cases": expected,
               "parallel_mode": "independent model replicas over disjoint cases; no speed claim"})
    for name, value in zip(("run", "data", "models", "calibration"), manifests[0]):
        write_json(root / f"manifests/{name}.json", value)
    write_json(root / "manifests/environment.json", {"workers": [read_json(root / f"worker-{i}/manifests/environment.json") for i in range(count)]})
    write_json(root / "tensor_map.json", {"workers": [read_json(root / f"worker-{i}/tensor_map.json") for i in range(count)]})
    for boundary in boundaries:
        append_jsonl(root / "trace/boundaries.jsonl", boundary)
    text = "# Parallel S1 cases\n\n" + "\n\n".join((root / f"worker-{i}/results/S1_cases.md").read_text(encoding="utf-8") for i in range(count))
    (root / "results/S1_cases.md").write_text(text, encoding="utf-8")
    (root / "reports/HANDOFF.md").write_text(
        f"# Parallel S1 handoff\n\nStatus: {status}. Completed cases: {len(cases)}/{expected}.\n\n"
        "Each allocated GPU ran a full Target+Drafter replica. Calibration/baseline were repeated identically per worker; "
        "evaluation prompts and complete binding pairs were partitioned, not duplicated. "
        "Worker reports retain individual checks and peak VRAM. No inference speedup claim.\n", encoding="utf-8")
    return 0 if complete else 2


def launch(args):
    import torch
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("parallel launch requires a Slurm allocation")
    devices = worker_devices(os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    processes, streams = [], []
    try:
        for index, identifier in enumerate(devices):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=identifier, OMP_NUM_THREADS="8")
            command = [sys.executable, "-m", "signal_study.run", "--config", args.config,
                       "--upstream", args.upstream, "--data-dir", args.data_dir,
                       "--output", str(root / f"worker-{index}"),
                       "--shard-index", str(index), "--shard-count", str(len(devices))]
            stdout = (root / f"worker-{index}.out").open("w", encoding="utf-8")
            stderr = (root / f"worker-{index}.err").open("w", encoding="utf-8")
            streams.extend([stdout, stderr])
            processes.append(subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr))
        codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for stream in streams:
            stream.close()
    try:
        return merge_outputs(root, len(devices), codes)
    except Exception as exc:
        write_json(root / "reports/tests.json", {"status": "failed", "stage": "merge", "reason": type(exc).__name__, "worker_exit_codes": codes})
        (root / "reports/HANDOFF.md").write_text("# Parallel run incomplete\n\nInspect worker reports. No valid combined result.\n", encoding="utf-8")
        print("Parallel run incomplete; inspect per-worker reports and local logs.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--upstream", default="vendor/SD-square")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(launch(parser.parse_args()))

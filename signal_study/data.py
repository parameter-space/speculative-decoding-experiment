from __future__ import annotations

import argparse
import os
import random
import socket
from pathlib import Path

from .common import append_jsonl, digest, read_json, validate_splits, write_json


def prompt_tokens(tokenizer, prompt):
    # Match TrainingModule.prep_for_gen's user-only chat template.
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        add_generation_prompt=True, enable_thinking=False)


def allocate(candidates, config):
    n = config["smoke_per_domain"]
    p = config["pilot_per_domain"]
    h = config["holdout_per_domain_max"]
    if len(candidates) < n + p:
        raise ValueError("insufficient eligible prompts for disjoint smoke and pilot")
    result = []
    for split, part in (("smoke", candidates[:n]), ("pilot", candidates[n:n+p]),
                        ("holdout", candidates[n+p:n+p+h])):
        result.extend(dict(row, split=split) for row in part)
    return result


def natural_rows(spec, split, tokenizer, cfg, root, locked_revision):
    """Read only public Parquet data, never execute a dataset repository script."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    files = HfApi().list_repo_files(spec["id"], repo_type="dataset", revision=locked_revision)
    prefix = f"{spec['config']}/{split}/"
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not shards:
        raise ValueError(f"no converted Parquet shards for {spec['id']} {prefix}; no fallback dataset")
    rows, rejected, seen = [], [], set()
    scanned = 0
    for shard in shards:
        path = hf_hub_download(spec["id"], shard, repo_type="dataset", revision=locked_revision,
                               cache_dir=root / "dataset_hub")
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                row_id = str(row.get("prompt_id", row.get("task_id", row.get("id", f"{shard}:{offset}"))))
                offset += 1
                scanned += 1
                prompt = str(row[spec["field"]])
                if spec["domain"] == "summary":
                    prompt = f"Summarize the following document: {prompt}"
                content_hash = digest(prompt)
                meta = {"prompt_id": f"{spec['domain']}:{split}:{row_id}", "domain": spec["domain"],
                        "dataset": spec["id"], "dataset_revision": locked_revision,
                        "dataset_config": spec["config"], "dataset_split": split,
                        "shard": shard, "row_id": row_id, "row_in_shard": offset - 1,
                        "prompt": prompt, "content_hash": content_hash}
                length = len(prompt_tokens(tokenizer, prompt))
                reason = "duplicate_content" if content_hash in seen else (
                    "input_too_long" if length > cfg["max_input_tokens"] else None)
                seen.add(content_hash)
                if reason:
                    rejected.append({"prompt_id": meta["prompt_id"], "reason": reason, "tokens": length})
                else:
                    rows.append(dict(meta, input_tokens=length))
                if scanned >= cfg["max_scan_rows"]:
                    break
            if scanned >= cfg["max_scan_rows"]:
                break
        if scanned >= cfg["max_scan_rows"]:
            break
    rows.sort(key=lambda r: digest([cfg["seed"], r["prompt_id"]]))
    return rows, {"dataset": spec["id"], "split": split, "scanned": scanned,
                  "eligible": len(rows), "excluded": rejected, "candidate_pool": "first max_scan_rows in sorted shards"}


def binding_rows(tokenizer, count, seed):
    rng = random.Random(seed)
    labels = ["RED", "BLUE", "GREEN", "BLACK", "WHITE", "GOLD", "PINK", "BROWN"]
    keys = ["alpha", "beta", "gamma", "delta", "theta", "omega", "sigma", "kappa"]
    key_pairs = [(a, b) for a in keys for b in keys if a != b]
    rng.shuffle(key_pairs)
    if count > len(key_pairs):
        raise ValueError("smoke binding template exhausted; implement new disjoint template families before expanding")
    result = []
    for pair_index in range(count):
        key_a, key_b = key_pairs[pair_index]
        variants = None
        for label_a in labels:
            for label_b in labels:
                if label_a == label_b:
                    continue
                pair = []
                for side, first, second in (("A", label_a, label_b), ("B", label_b, label_a)):
                    prompt = f"Dictionary: {key_a}={first}, {key_b}={second}. What is the value of {key_a}? Reply with the label only."
                    head = prompt_tokens(tokenizer, prompt)
                    bridge = tokenizer.encode("The requested label is", add_special_tokens=False)
                    # Prefix must stay unchanged when appending each space-prefixed label.
                    text_prefix = tokenizer.decode(head + bridge, skip_special_tokens=False)
                    prefix = tokenizer.encode(text_prefix, add_special_tokens=False)
                    full = tokenizer.encode(text_prefix + " " + first, add_special_tokens=False)
                    if full[:-1] != prefix or len(full) != len(prefix) + 1:
                        pair = []
                        break
                    pair.append({"prompt_id": f"binding-smoke-{pair_index:04d}-{side}",
                                 "base_problem_id": f"binding-smoke-{pair_index:04d}",
                                 "split": "smoke", "domain": "binding", "side": side,
                                 "prompt": prompt, "prefix_ids": prefix, "bridge_ids": bridge,
                                 "label": first, "label_id": full[-1],
                                 "template_family": "dictionary_query_v1", "key_names": [key_a, key_b],
                                 "state_origin": "synthetic_teacher_prefix"})
                if len(pair) == 2 and len(pair[0]["prefix_ids"]) == len(pair[1]["prefix_ids"]):
                    if pair[0]["prefix_ids"][-len(bridge):] == pair[1]["prefix_ids"][-len(bridge):]:
                        variants = pair
                        break
            if variants:
                break
        if variants is None:
            raise ValueError("no equal-length binding pair with verified one-token labels")
        # Deterministically vary keys; duplicates are rejected, not silently counted twice.
        result.extend(variants)
    validate_splits(result)
    return result


def prepare(config_path, root):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    cfg, root = read_json(config_path), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "natural.jsonl").exists() or (root / "binding.jsonl").exists():
        raise FileExistsError("use a new data directory; immutable split files already exist")
    api = HfApi()
    target_rev = cfg["target_revision"]
    draft_rev = cfg["drafter_revision"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["target"], revision=target_rev)
    all_rows, audits, revisions = [], [], {}
    for spec in cfg["datasets"]:
        rev = spec["revision"]
        revisions[spec["id"]] = rev
        if "calibration_split" in spec:
            candidates, audit = natural_rows(spec, spec["calibration_split"], tokenizer, cfg, root, rev)
            if len(candidates) < cfg["calibration_count"]:
                raise ValueError("calibration pool too small")
            all_rows.extend(dict(r, split="calibration") for r in candidates[:cfg["calibration_count"]])
            audits.append(audit)
        candidates, audit = natural_rows(spec, spec["eval_split"], tokenizer, cfg, root, rev)
        used = {r["content_hash"] for r in all_rows}
        candidates = [r for r in candidates if r["content_hash"] not in used]
        all_rows.extend(allocate(candidates, cfg))
        audits.append(audit)
    binding = binding_rows(tokenizer, cfg["binding_pairs"], cfg["seed"])
    validate_splits(all_rows + binding)
    for name, rows in (("natural", all_rows), ("binding", binding)):
        for row in rows:
            append_jsonl(root / f"{name}.jsonl", row)
    write_json(root / "manifest.json", {"config_hash": digest(cfg), "config": cfg,
               "target_revision": target_rev, "drafter_revision": draft_rev,
               "dataset_revisions": revisions, "natural_hash": digest(all_rows),
               "binding_hash": digest(binding), "audits": audits,
               "calibration_scope": "UltraChat train_sft only; absent same-domain donors remain NA"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0] != "ariel-k2":
        raise SystemExit("Data preparation must run inside your allocated ariel-k2 compute job.")
    prepare(args.config, args.data_dir)

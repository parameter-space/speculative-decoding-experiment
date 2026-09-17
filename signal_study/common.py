from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def overlap_reference(p, q):
    """Independent float64/CPU reference, with no epsilon or renormalization."""
    import numpy as np
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1 or p.size == 0:
        raise ValueError("p/q must have identical non-empty full-vocabulary shape")
    for distribution in (p, q):
        if not np.isfinite(distribution).all() or (distribution < 0).any():
            raise ValueError("invalid probabilities")
        if not np.isclose(distribution.sum(), 1, atol=1e-5, rtol=0):
            raise ValueError("unnormalized probabilities")
    return float(np.minimum(p, q).sum())


def choose_donor(recipient, candidates):
    eligible = []
    for donor in candidates:
        if donor["split"] != "calibration" or donor["prompt_id"] == recipient["prompt_id"]:
            continue
        if any(donor[k] != recipient[k] for k in ("domain", "z_id", "p_src_argmax")):
            continue
        if donor["source_length"] // 64 != recipient["source_length"] // 64:
            continue
        if math.floor(donor["p_src_entropy"] / 0.5) != math.floor(recipient["p_src_entropy"] / 0.5):
            continue
        eligible.append(donor)
    return min(eligible, key=lambda d: (abs(d["p_src_z"] - recipient["p_src_z"]), d["prompt_id"], d["source_pos"]), default=None)


def validate_splits(rows):
    ids, contents, problems = {}, {}, {}
    for row in rows:
        split = row["split"]
        for mapping, key in ((ids, row["prompt_id"]), (contents, digest(row["prompt"]))):
            if key in mapping:
                raise ValueError(f"duplicate prompt: {row['prompt_id']}")
            mapping[key] = split
        if row.get("base_problem_id"):
            key = row["base_problem_id"]
            if key in problems and problems[key] != split:
                raise ValueError(f"synthetic split leakage: {key}")
            problems[key] = split


def normalized_error(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("incompatible or nonfinite comparison")
    return float(np.max(np.abs(a - b)))

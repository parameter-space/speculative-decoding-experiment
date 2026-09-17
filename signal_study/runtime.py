from __future__ import annotations

import gc
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

from .common import file_digest, read_json, write_json


def import_upstream(repo, commit):
    repo = Path(repo).resolve()
    head = subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != commit or dirty:
        raise ValueError("upstream must be clean and at the pinned commit")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(repo))
    module = importlib.import_module("main")
    module.PRETTY_PRINT = False
    return module


def environment():
    import torch
    result = {"python": platform.python_version(), "platform": platform.system(),
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "gpus": []}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            result["gpus"].append({"name": prop.name, "total_memory_bytes": prop.total_memory,
                                   "capability": [prop.major, prop.minor]})
        result["bf16_supported"] = torch.cuda.is_bf16_supported()
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
        result["driver_query"] = query.stdout.strip() if query.returncode == 0 else "query_failed"
    # Do not dump environment variables, tokens, UUIDs, IPs, or shell history.
    return result


def load(cfg, dataset_manifest, repo, report_dir):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    upstream = import_upstream(repo, cfg["upstream_commit"])
    from src.models.llama import LlamaForCausalLM
    config_path = hf_hub_download(cfg["checkpoint"], "config.json", revision=cfg["checkpoint_revision"])
    model_cfg = read_json(config_path)
    expected = {"drafter": cfg["drafter"].lower(), "verifier": cfg["target"].lower(),
                "method": "guided-drafter", "guide_method": "merged", "d_layer": "all", "v_layer": [3, 16, 29]}
    for key, value in expected.items():
        actual = model_cfg.get(key)
        if key in ("drafter", "verifier") and isinstance(actual, str):
            actual = actual.lower()
        if actual != value:
            raise ValueError(f"unexpected checkpoint {key}: {actual}")
    revisions = {cfg["target"].lower(): dataset_manifest["target_revision"],
                 cfg["drafter"].lower(): dataset_manifest["drafter_revision"]}
    canonical = {name.lower(): name for name in (cfg["target"], cfg["drafter"])}

    def pinned_loader(name, float16=False):
        # CPU construction matches upstream; no second GPU checkpoint allocation.
        return LlamaForCausalLM.from_pretrained(
            canonical[name.lower()], revision=revisions[name.lower()],
            torch_dtype=torch.float16 if float16 else torch.float32,
            attn_implementation=cfg["attention_backend"], low_cpu_mem_usage=True,
            use_safetensors=True), "llama"

    class PinnedTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return AutoTokenizer.from_pretrained(canonical[name.lower()], revision=revisions[name.lower()], **kwargs)

    old_loader, old_tokenizer = upstream.load_model, upstream.AutoTokenizer
    upstream.load_model, upstream.AutoTokenizer = pinned_loader, PinnedTokenizer
    try:
        mod = upstream.TrainingModule(**{k: v for k, v in model_cfg.items() if k != "_instantiator"})
    finally:
        upstream.load_model, upstream.AutoTokenizer = old_loader, old_tokenizer
    mod.d_base.to(torch.bfloat16)
    mod.latent_mod_prep.to(torch.bfloat16)
    mod.guidance_embd_layer.to(torch.bfloat16)
    path = hf_hub_download(cfg["checkpoint"], "pytorch_model.bin", revision=cfg["checkpoint_revision"])
    checksum = file_digest(path)
    if checksum != cfg["checkpoint_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    # Tensor-only unpickler, explicit CPU mapping, and file-backed storage.
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    allowed = ("d_base.", "latent_mod_prep.", "guidance_embd_layer.")
    extra = [k for k in state if not k.startswith(allowed)]
    keys_report = {"unexpected_top_level_keys": extra, "modules": {}}
    write_json(report_dir / "checkpoint_keys.json", keys_report)
    if extra:
        raise ValueError("unapproved checkpoint keys; see checkpoint_keys.json")
    for prefix, obj in (("d_base.", mod.d_base), ("latent_mod_prep.", mod.latent_mod_prep),
                        ("guidance_embd_layer.", mod.guidance_embd_layer)):
        subset = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        reference = obj.state_dict()
        missing = sorted(set(reference) - set(subset))
        unexpected = sorted(set(subset) - set(reference))
        shape_errors = [k for k in set(reference) & set(subset) if reference[k].shape != subset[k].shape]
        keys_report["modules"][prefix] = {"missing": missing, "unexpected": unexpected, "shape_errors": shape_errors}
        write_json(report_dir / "checkpoint_keys.json", keys_report)
        if missing or unexpected or shape_errors:
            raise ValueError(f"checkpoint mismatch in {prefix}")
        obj.load_state_dict(subset, strict=True)
    del subset, state
    gc.collect()
    mod.NG = cfg["draft_length"]
    mod.greedy_sample = False
    mod.eval().requires_grad_(False).to("cuda")
    if mod.v_base.config.vocab_size != mod.d_base.config.vocab_size:
        raise ValueError("target/draft vocabulary sizes differ")
    details = {"checkpoint": cfg["checkpoint"], "checkpoint_revision": cfg["checkpoint_revision"],
               "checkpoint_sha256": checksum, "upstream_commit": cfg["upstream_commit"],
               "checkpoint_config": model_cfg, "target_revision": dataset_manifest["target_revision"],
               "drafter_revision": dataset_manifest["drafter_revision"], "precision": {
                   "target_weights": "float16", "drafter_and_guidance_weights": "bfloat16",
                   "cuda_autocast": "bfloat16", "metric_softmax": "float32",
                   "note": "Matches official eval weight/autocast choices; not uniformly BF16 target weights."},
               "attention_backend": cfg["attention_backend"], "models": {}}
    for name, obj in (("target", mod.v_base), ("drafter", mod.d_base), ("delta_projection", mod.latent_mod_prep),
                      ("guidance_projection", mod.guidance_embd_layer)):
        details["models"][name] = {"parameters": sum(p.numel() for p in obj.parameters()),
                                  "weight_bytes": sum(p.numel() * p.element_size() for p in obj.parameters())}
    details["unique_total_weight_bytes"] = sum(p.numel() * p.element_size() for p in mod.parameters())
    return mod, details

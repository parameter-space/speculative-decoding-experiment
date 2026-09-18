"""Terminal precision diagnostic; never resume S1 after mutating Target precision."""
import gc

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import DynamicCache

from .capture import capture_prompt
from .common import digest, write_json
from .diagnose import alignment_probe, check_baseline, compare, snapshot_structure
from .state import preserved_rng
from .validation import ValidationError, validate_snapshot


@torch.inference_mode()
def fp32_rebuild(model, snap):
    """No captured KV is read: every split cache starts empty, including physical layout."""
    device = next(model.parameters()).device
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            # Promotion preserves stored values, not lost checkpoint precision.
            model.float()
            if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError("Target parameters are not all FP32")
            logical = snap["prefix"][None].to(device)
            end = snap["curr"] + 1
            physical = snap["ids"][:, :end].to(device)
            mask = snap["mask"][:, :end].to(device)
            positions = snap["positions"][:, :end].to(device)
            cache_dtypes = set()

            def forward(ids, attention, pos, split):
                decoder = model.get_decoder()
                if split:
                    cache = DynamicCache()
                    decoder(ids[:, :-1], attention_mask=attention[:, :-1], position_ids=pos[:, :-1],
                            past_key_values=cache, use_cache=True)
                    if cache.get_seq_length() != ids.shape[1] - 1:
                        raise ValueError("FP32 rebuilt cache has wrong prefix length")
                    out = decoder(ids[:, -1:], attention_mask=attention, position_ids=pos[:, -1:],
                                  past_key_values=cache, use_cache=True)
                    for k, v in zip(cache.key_cache, cache.value_cache):
                        cache_dtypes.update((str(k.dtype), str(v.dtype)))
                    if cache.get_seq_length() != ids.shape[1] or cache_dtypes != {"torch.float32"}:
                        raise ValueError("rebuilt KV must be full-length FP32")
                else:
                    out = decoder(ids, attention_mask=attention, position_ids=pos, use_cache=False)
                hidden = out.last_hidden_state[:, -1]
                logits = model.lm_head(hidden)
                if hidden.dtype != torch.float32 or logits.dtype != torch.float32:
                    raise ValueError("Target hidden/logits are not FP32")
                return logits.cpu()

            inputs = {"logical": (logical, torch.ones_like(logical),
                                   torch.arange(logical.shape[1], device=device)[None]),
                      "physical": (physical, mask, positions)}
            outputs, repeats = {}, {}
            for name, values in inputs.items():
                for split in (False, True):
                    key = name + ("_split" if split else "_full")
                    outputs[key] = forward(*values, split)
                    repeats[key] = compare(outputs[key], forward(*values, split))
            return {"status": "complete", "autocast": False, "tf32": False, "sdpa": "math",
                    "weight_dtype": "torch.float32", "kv_dtypes": sorted(cache_dtypes),
                    "cache_origin": "new empty DynamicCache per forward; captured KV never used",
                    "logical_full_split": compare(outputs["logical_full"], outputs["logical_split"]),
                    "logical_physical_full": compare(outputs["logical_full"], outputs["physical_full"]),
                    "logical_physical_split": compare(outputs["logical_full"], outputs["physical_split"]),
                    "physical_full_split": compare(outputs["physical_full"], outputs["physical_split"]),
                    "repeat": repeats}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


@torch.inference_mode()
def run_precision(mod, rows, row, cfg, report, out):
    report["note"] = ("Diagnostic only; native gate outcome retained. Target promotion uses existing stored "
                      "weight values. FP32 completion is NOT S1 validation or a production precision change.")
    item = {"prompt_id": row["prompt_id"], "status": "running", "native_gate": {}, "native_alignment": {}}
    report["precision_probe"] = item
    with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.MATH):
        baseline = check_baseline(mod, [r["prompt"] for r in rows], cfg, math_backend=True)
        report["baseline"] = baseline
        print("Baseline:", baseline["status"], flush=True)
        if baseline["status"] != "passed":
            report["status"] = item["status"] = "failed"
            return 2
        with preserved_rng():
            torch.manual_seed(cfg["seed"])
            snap, _ = capture_prompt(mod, row, cfg["max_new_tokens"])
        if snap is None:
            raise ValidationError("precision probe has no normal boundary")
        item["structure"] = snapshot_structure(snap)
        item["prefix_hash"] = digest(snap["prefix"].tolist())
        item["native_gate"]["checks"] = {}
        try:
            validate_snapshot(mod, snap, baseline["tolerance"], diagnostics=item["native_gate"]["checks"])
            item["native_gate"]["status"] = "passed"
        except ValidationError as exc:
            item["native_gate"].update(status="failed", error=str(exc))
        print("Native endpoint:", item["native_gate"]["status"], item["structure"], flush=True)
        alignment_probe(mod, snap, item["native_alignment"])
    # Exit BF16 autocast before conversion so its cached cast weights are released.
    # Only retain inputs, not original KV or Drafter/guide outputs, for the FP32 phase.
    inputs = {k: snap[k] for k in ("prefix", "ids", "mask", "positions", "curr")}
    del snap
    gc.collect()
    torch.cuda.empty_cache()
    write_json(out / "diagnostic.json", report)
    print("Target full FP32 starting: autocast/TF32 off; all KV rebuilt from inputs.", flush=True)
    item["fp32"] = fp32_rebuild(mod.v_base, inputs)
    for key in ("logical_full_split", "logical_physical_full", "logical_physical_split", "physical_full_split"):
        result = item["fp32"][key]
        print(f"FP32 {key}: TV={result['TV']:.12g} max_logit_error={result['max_logit_error']:.12g}", flush=True)
    item["status"] = report["status"] = "complete"
    print("Precision diagnostic complete; native gate outcome retained. NOT a passed S1 experiment.", flush=True)
    return 0

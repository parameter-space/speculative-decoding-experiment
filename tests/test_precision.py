import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from signal_study.capture import capture_prompt
from signal_study.precision import fp32_rebuild, run_precision
from signal_study.validation import ValidationError
from test_tiny_upstream import tiny_model


class PrecisionTests(unittest.TestCase):
    def fixture(self):
        torch.set_num_threads(2)
        mod = tiny_model()
        with patch("torch.cuda.synchronize"), torch.inference_mode():
            snap, _ = capture_prompt(mod, dict(prompt="toy"), 32)
        return mod, snap

    def test_fp32_new_cache_masked_layout_and_preserved_weights(self):
        mod, snap = self.fixture()
        # Deliberately invalid captured KV must never be touched by FP32 reconstruction.
        snap["v_cache"] = None
        for name in ("ids", "mask", "positions"):
            value = snap[name]
            snap[name] = torch.cat((value[:, :2], torch.zeros_like(value[:, :1]), value[:, 2:]), dim=1)
        snap["curr"] += 1
        mod.v_base.half()
        before = {k: p.float().clone() for k, p in mod.v_base.named_parameters()}
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = fp32_rebuild(mod.v_base, snap)
        self.assertEqual(result["kv_dtypes"], ["torch.float32"])
        for key in ("logical_full_split", "logical_physical_full", "logical_physical_split", "physical_full_split"):
            self.assertLess(result[key]["TV"], 1e-6)
        for item in result["repeat"].values():
            self.assertEqual(item["TV"], 0)
        for key, p in mod.v_base.named_parameters():
            self.assertEqual(p.dtype, torch.float32)
            self.assertTrue(torch.equal(p, before[key]))
        self.assertEqual(flags, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        json.dumps(result, allow_nan=False)

    def test_backend_flags_restored_on_error(self):
        mod, snap = self.fixture()
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with patch.object(mod.v_base, "get_decoder", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                fp32_rebuild(mod.v_base, snap)
        self.assertEqual(flags, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))

    def test_native_failure_retained_after_successful_precision_diagnostic(self):
        mod, snap = self.fixture()
        report = {}
        with tempfile.TemporaryDirectory() as folder, \
                patch("signal_study.precision.check_baseline", return_value=dict(status="passed", tolerance={})), \
                patch("signal_study.precision.capture_prompt", return_value=(snap, {})), \
                patch("signal_study.precision.validate_snapshot", side_effect=ValidationError("original TV failure")), \
                patch("signal_study.precision.alignment_probe"), \
                patch("signal_study.precision.fp32_rebuild", return_value={k: dict(TV=0., max_logit_error=0.) for k in
                    ("logical_full_split", "logical_physical_full", "logical_physical_split", "physical_full_split")}), \
                patch("torch.cuda.empty_cache"):
            result = run_precision(mod, [dict(prompt="toy")], dict(prompt_id="toy"),
                                   dict(seed=11, max_new_tokens=32), report, Path(folder))
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["precision_probe"]["native_gate"]["status"], "failed")
        self.assertEqual(report["precision_probe"]["native_gate"]["error"], "original TV failure")

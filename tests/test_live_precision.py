import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from signal_study.capture import capture_prompt, target_logits
from signal_study.live_precision import live_target_fp32, run_live_preflight, selected_rows
from signal_study.validation import ValidationError
from test_tiny_upstream import tiny_model


class LivePrecisionTests(unittest.TestCase):
    def model(self):
        torch.set_num_threads(2)
        mod = tiny_model()
        mod.v_base.half()
        mod.d_base.bfloat16()
        mod.guidance_embd_layer.bfloat16()
        mod.latent_mod_prep.bfloat16()
        return mod

    def data(self):
        cfg = dict(seed=11, max_new_tokens=32, calibration_count=2, smoke_per_domain=1,
                   binding_pairs=1, repeat_logit_cap=1e-6, alignment_logit_cap=.25, alignment_tv_cap=.005)
        rows = [dict(prompt_id=f"cal{i}", prompt="toy", domain="dialogue", split="calibration") for i in range(2)]
        rows += [dict(prompt_id=d, prompt="toy", domain=d, split="smoke")
                 for d in ("dialogue", "math", "code", "summary")]
        binding = [dict(prompt_id=f"binding{i}", prefix_ids=[1, 2, i + 3, 4], domain="binding", split="smoke")
                   for i in range(2)]
        return cfg, rows, binding

    def test_real_generation_has_fp32_target_and_bf16_draft_cache_and_restores(self):
        mod = self.model()
        original = {name: p.clone() for name, p in mod.named_parameters()}
        flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        with torch.inference_mode(), patch("torch.cuda.synchronize"):
            with live_target_fp32(mod) as audit:
                snap, _ = capture_prompt(mod, dict(prompt="toy"), 32)
                self.assertIsNotNone(snap)
                self.assertEqual({t.dtype for pair in snap["v_cache"]["layers"] for t in pair}, {torch.float32})
                self.assertEqual({t.dtype for pair in snap["d_cache"]["layers"] for t in pair}, {torch.bfloat16})
                self.assertEqual(snap["p_src_logits"].dtype, torch.float32)
                self.assertEqual(snap["q_original_logits"].dtype, torch.bfloat16)
                self.assertEqual(snap["G"].dtype, torch.bfloat16)
                self.assertEqual(target_logits(mod, snap, cached=True).dtype, torch.float32)
                self.assertGreater(audit["checked_fp32_cache_calls"], 0)
                self.assertGreater(audit["guide_calls"], 0)
        for name, p in mod.named_parameters():
            self.assertEqual(p.dtype, original[name].dtype)
            self.assertTrue(torch.equal(p, original[name]), name)
        self.assertNotIn("forward", mod.v_base.get_decoder().__dict__)
        self.assertEqual(flags, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))

    def test_forward_and_precision_restore_even_on_exception(self):
        mod = self.model()
        decoder = mod.v_base.get_decoder()
        before = decoder.forward
        with self.assertRaisesRegex(RuntimeError, "test"), live_target_fp32(mod):
            raise RuntimeError("test")
        self.assertEqual(decoder.forward, before)
        self.assertEqual(mod.v_base.lm_head.weight.dtype, torch.float16)
        self.assertEqual(next(mod.guidance_embd_layer.parameters()).dtype, torch.bfloat16)

    def test_all_endpoint_types_and_report_on_tiny_actual_model(self):
        cfg, natural, binding = self.data()
        report = {}
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.synchronize"):
            code = run_live_preflight(self.model(), natural, binding, cfg, report, Path(folder))
            summary = json.loads((Path(folder) / "preflight-summary.json").read_text())
        self.assertEqual(code, 0, report)
        self.assertEqual(summary["counts"], {"passed": 8})
        self.assertEqual(summary["planned"], 8)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(all(i["target_kv_dtypes"] == ["torch.float32"] for i in report["endpoints"]))

    def test_validation_failures_are_collected_without_changing_frozen_tolerance(self):
        cfg, natural, binding = self.data()
        frozen = dict(repeat=1e-6, alignment=1e-5, alignment_tv=1e-6)
        report = {}
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.synchronize"), \
                patch("signal_study.live_precision.check_baseline", return_value=dict(status="passed", tolerance=frozen)), \
                patch("signal_study.live_precision.inspect_snapshot", side_effect=ValidationError("test TV gate")):
            code = run_live_preflight(self.model(), natural, binding, cfg, report, Path(folder))
            summary = json.loads((Path(folder) / "preflight-summary.json").read_text())
        self.assertEqual(code, 2)
        self.assertEqual(summary["counts"], {"failed": 8})
        self.assertEqual(len(summary["failures"]), 8)
        self.assertEqual(frozen, dict(repeat=1e-6, alignment=1e-5, alignment_tv=1e-6))

    def test_baseline_failure_does_not_capture_or_claim_pass(self):
        cfg, natural, binding = self.data()
        report = {}
        with tempfile.TemporaryDirectory() as folder, \
                patch("signal_study.live_precision.check_baseline", return_value=dict(status="failed", error="baseline test")), \
                patch("signal_study.live_precision.capture_prompt") as capture:
            code = run_live_preflight(self.model(), natural, binding, cfg, report, Path(folder))
            summary = json.loads((Path(folder) / "preflight-summary.json").read_text())
        capture.assert_not_called()
        self.assertEqual(code, 2)
        self.assertEqual(summary["stage"], "baseline")
        self.assertEqual(summary["status"], "failed")

    def test_incomplete_dataset_rejected(self):
        cfg, natural, binding = self.data()
        with self.assertRaises(ValueError):
            selected_rows(natural[:-1], binding, cfg)

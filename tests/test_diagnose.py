import json
import unittest
from unittest.mock import patch

import torch

from signal_study.diagnose import (alignment_probe, calibration_row, check_baseline, check_endpoint,
                                  fp32_head, preflight_rows, probe, snapshot_structure)
from signal_study.capture import capture_prompt
from signal_study.validation import ValidationError
from test_tiny_upstream import tiny_model


class DiagnosticTests(unittest.TestCase):
    def capture_fixture(self):
        torch.set_num_threads(2)
        mod = tiny_model()
        with patch("torch.cuda.synchronize"), torch.inference_mode():
            snap, _ = capture_prompt(mod, dict(prompt="toy"), 32)
        return mod, snap

    def test_alignment_probe_reports_controls_without_mutation(self):
        mod, snap = self.capture_fixture()
        before = [(k.clone(), v.clone()) for k, v in snap["v_cache"]["layers"]]
        rng = torch.get_rng_state().clone()
        report = {}
        alignment_probe(mod, snap, report)
        self.assertEqual(report["status"], "complete")
        for key in ("active_tokens_match_prefix", "active_positions_contiguous", "source_z_aligned",
                    "cache_layer_lengths_match", "pending_slice_valid", "mask_binary", "last_token_active"):
            self.assertTrue(report["structure"][key], key)
        for key in ("fresh_cached", "fresh_cached_fp32_head_only", "fresh_physical_rebuild",
                    "cached_physical_rebuild", "fresh_physical_rebuild_fp32_head_only",
                    "cached_physical_rebuild_fp32_head_only"):
            self.assertLess(report[key]["TV"], 1e-6, key)
        self.assertEqual(report["fresh_repeat"]["TV"], 0)
        self.assertEqual(report["cached_repeat"]["TV"], 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for (k, v), (bk, bv) in zip(snap["v_cache"]["layers"], before):
            self.assertTrue(torch.equal(k, bk))
            self.assertTrue(torch.equal(v, bv))
        json.dumps(report, allow_nan=False)

    def test_structure_flags_wrong_positions_tokens_and_cache_length(self):
        _, snap = self.capture_fixture()
        snap["positions"] = snap["positions"].clone()
        snap["positions"][0, snap["curr"]] += 1
        snap["prefix"] = snap["prefix"].clone()
        snap["prefix"][-1] += 1
        snap["v_cache"]["length"] += 1
        report = snapshot_structure(snap)
        self.assertFalse(report["active_positions_contiguous"])
        self.assertFalse(report["active_tokens_match_prefix"])
        self.assertFalse(report["cache_layer_lengths_match"])

    def test_alignment_probe_with_a_masked_physical_cache_slot(self):
        mod, snap = self.capture_fixture()
        start = snap["v_cache"]["length"]
        # Insert an ignored slot at the cache tail without altering the logical prefix.
        for name in ("ids", "mask", "positions"):
            value = snap[name]
            snap[name] = torch.cat((value[:, :start], torch.zeros_like(value[:, :1]), value[:, start:]), dim=1)
        snap["curr"] += 1
        snap["v_cache"]["layers"] = [
            (torch.cat((k, torch.zeros_like(k[..., :1, :])), dim=-2),
             torch.cat((v, torch.zeros_like(v[..., :1, :])), dim=-2))
            for k, v in snap["v_cache"]["layers"]]
        snap["v_cache"]["length"] += 1
        report = {}
        alignment_probe(mod, snap, report)
        self.assertEqual(report["structure"]["masked_tokens"], 1)
        self.assertTrue(report["structure"]["active_tokens_match_prefix"])
        self.assertTrue(report["structure"]["active_positions_contiguous"])
        self.assertLess(report["fresh_cached"]["TV"], 1e-6)
        self.assertLess(report["fresh_physical_rebuild"]["TV"], 1e-6)

    def test_tv_failure_keeps_numeric_checks_and_original_error_when_probe_fails(self):
        mod, snap = self.capture_fixture()
        fresh = torch.zeros_like(snap["q_original_logits"])
        cached = fresh.clone()
        cached[..., 0] = .2
        checks, alignment = {}, {}
        tolerance = dict(repeat=1e-6, alignment=.25, alignment_tv=1e-6)
        with patch("signal_study.diagnose.capture_prompt", return_value=(snap, {})), \
                patch("signal_study.validation.target_logits", side_effect=[fresh, cached]), \
                patch("signal_study.diagnose.alignment_probe", side_effect=RuntimeError("probe failure")):
            with self.assertRaisesRegex(ValidationError, r"TV=.*cap=.*cache_tokens="):
                check_endpoint(mod, {}, dict(seed=11, max_new_tokens=32), tolerance,
                               checks_out=checks, alignment=alignment)
        self.assertGreater(checks["p_fresh_cached_TV"], tolerance["alignment_tv"])
        self.assertEqual(checks["alignment_tv_tolerance"], 1e-6)
        self.assertLessEqual(checks["restore"], tolerance["repeat"])
        self.assertNotIn("A_B_A", checks)
        self.assertEqual(alignment, dict(status="failed", error_type="RuntimeError"))
        self.assertEqual(tolerance, dict(repeat=1e-6, alignment=.25, alignment_tv=1e-6))
        json.dumps(checks, allow_nan=False)

    def test_calibration_selection_is_exact_and_split_restricted(self):
        row = dict(split="calibration", prompt_id="failed-case")
        self.assertIs(calibration_row([row], "failed-case"), row)
        for rows in ([], [dict(split="smoke", prompt_id="failed-case")], [row, row]):
            with self.assertRaises(ValueError):
                calibration_row(rows, "failed-case")

    def test_probability_audit_covers_all_snapshot_paths(self):
        audit = {}
        with patch("torch.cuda.synchronize"):
            result = check_endpoint(tiny_model(), dict(prompt="toy"), dict(seed=11, max_new_tokens=32),
                                    dict(repeat=1e-6, alignment=1e-5, alignment_tv=1e-6), audit=audit)
        self.assertEqual(result["checks"]["A_B_A"], 0)
        self.assertEqual(set(audit), {"target.fresh", "target.cached", "drafter.original", "target.source"})
        self.assertTrue(all(item["float64"]["fp64_sum_gate_passed"] for item in audit.values()))

    def test_preflight_selection_is_fixed_and_domain_balanced(self):
        rows = [dict(split="smoke", domain=str(i % 4), prompt_id=str(i)) for i in range(8)]
        self.assertEqual([r["prompt_id"] for r in preflight_rows(rows)], ["0", "1", "2", "3"])
        with self.assertRaises(ValueError):
            preflight_rows(rows[:3])

    def test_endpoint_math_snapshot_checks(self):
        torch.set_num_threads(2)
        cfg = dict(seed=11, max_new_tokens=32)
        with patch("torch.cuda.synchronize"):
            result = check_endpoint(tiny_model(), dict(prompt="toy"), cfg,
                                    dict(repeat=1e-6, alignment=1e-5, alignment_tv=1e-6))
        self.assertEqual(result["round"], 2)
        self.assertEqual(result["checks"]["A_B_A"], 0)

    def test_missing_boundary_does_not_pass(self):
        with patch("signal_study.diagnose.capture_prompt", return_value=(None, {})):
            with self.assertRaises(ValidationError):
                check_endpoint(None, {}, dict(seed=11, max_new_tokens=32), {})

    def test_math_baseline_scope_and_failure_restore(self):
        def flags():
            return (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                    torch.backends.cuda.math_sdp_enabled())
        before = flags()
        def fail(*args):
            self.assertEqual(flags(), (False, False, True))
            raise ValidationError("test gate")
        with patch("signal_study.diagnose.baseline_tests", side_effect=fail):
            result = check_baseline(None, [], {}, math_backend=True)
        self.assertEqual(result, dict(status="failed", error="test gate"))
        self.assertEqual(flags(), before)

    def test_math_baseline_with_tiny_model(self):
        torch.set_num_threads(2)
        cfg = dict(seed=11, repeat_logit_cap=1e-6, alignment_logit_cap=.01, alignment_tv_cap=.001)
        with patch("torch.cuda.synchronize"), torch.inference_mode():
            result = check_baseline(tiny_model(), ["a", "b"], cfg, math_backend=True)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["reports"]), 2)

    def test_probe_cache_positions_and_repeat(self):
        torch.set_num_threads(2)
        model = tiny_model().v_base
        ids = (torch.arange(113)[None] % 63)
        for explicit, math in ((False, False), (True, False), (True, True)):
            result = probe(model, ids, explicit=explicit, math_backend=math)
            self.assertLess(result["native"]["TV"], 1e-6)
            self.assertLess(result["fp32_head_only"]["TV"], 1e-6)
            self.assertEqual(result["full_repeat"]["max_logit_error"], 0)
            self.assertEqual(result["split_repeat"]["max_logit_error"], 0)

    def test_head_chunking_and_setting_restore(self):
        head = torch.nn.Linear(8, 4200)
        hidden = torch.randn(1, 8)
        old = torch.backends.cuda.matmul.allow_tf32
        with torch.inference_mode():
            actual = fp32_head(head, hidden)
            torch.testing.assert_close(actual, head(hidden))
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old)

    def test_short_prefix_rejected(self):
        with self.assertRaises(ValueError):
            probe(None, torch.ones(1, 1, dtype=torch.long))

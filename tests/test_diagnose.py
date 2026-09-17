import unittest
from unittest.mock import patch

import torch

from signal_study.diagnose import check_baseline, fp32_head, probe
from signal_study.validation import ValidationError
from test_tiny_upstream import tiny_model


class DiagnosticTests(unittest.TestCase):
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

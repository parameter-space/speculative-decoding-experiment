import unittest

import torch

from signal_study.diagnose import fp32_head, probe
from test_tiny_upstream import tiny_model


class DiagnosticTests(unittest.TestCase):
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

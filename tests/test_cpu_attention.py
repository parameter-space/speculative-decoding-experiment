import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from signal_study.cpu_attention import cpu_attention, primitive_probe
from signal_study.cpu_recovery import run_recovery
from signal_study.live_precision import run_live_preflight
import test_live_precision as fixtures


class CPUAttentionTests(unittest.TestCase):
    def test_long_causal_and_physical_masks_match_independent_cpu_sdpa(self):
        torch.set_num_threads(2)
        rng = torch.Generator().manual_seed(62)
        native = F.scaled_dot_product_attention
        report = {}
        for length in (471, 513, 769):
            q, k, v = (torch.randn(1, 2, length, 128, dtype=torch.float64, generator=rng) for _ in range(3))
            for causal in (True, False):
                mask = None
                if not causal:
                    mask = torch.ones(1, 1, length, length, dtype=torch.bool).tril()
                    mask[..., 2::7] = False
                with sdpa_kernel(SDPBackend.MATH):
                    expected = native(q, k, v, attn_mask=mask, is_causal=causal)
                with cpu_attention(report):
                    actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
                self.assertEqual(report['last_query_checks'], 1)
        self.assertIs(F.scaled_dot_product_attention, native)

    def test_mask_single_query_all_masked_and_additive(self):
        q = torch.randn(1, 2, 1, 8, dtype=torch.float64)
        k, v = (torch.randn(1, 2, 23, 8, dtype=torch.float64) for _ in range(2))
        for mask in (torch.zeros(1, 1, 1, 23, dtype=torch.bool),
                     torch.zeros(1, 1, 1, 23, dtype=torch.float64).masked_fill(
                         torch.arange(23)[None, None, None, :] % 3 == 0, -torch.inf)):
            with sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            with cpu_attention({}):
                actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            torch.testing.assert_close(actual, expected)

    def test_lower_precision_untouched_and_restore_on_error(self):
        native = F.scaled_dot_product_attention
        q = torch.randn(1, 2, 5, 8)
        expected = native(q, q, q)
        report = {}
        with self.assertRaisesRegex(RuntimeError, 'stop'), cpu_attention(report):
            torch.testing.assert_close(F.scaled_dot_product_attention(q, q, q), expected, atol=0, rtol=0)
            self.assertEqual(report['calls'], 0)
            raise RuntimeError('stop')
        self.assertIs(F.scaled_dot_product_attention, native)

    def test_cpu_crosscheck_failure_is_not_silently_accepted(self):
        native = F.scaled_dot_product_attention
        q = torch.randn(1, 2, 5, 8, dtype=torch.float64)
        def wrong(*args, **kwargs):
            return native(*args, **kwargs) + .1
        with patch.object(F, 'scaled_dot_product_attention', wrong), cpu_attention({}):
            with self.assertRaisesRegex(ValueError, 'crosscheck failed'):
                F.scaled_dot_product_attention(q, q, q)

    def test_primitive_probe_compares_each_stage_and_preserves_rng(self):
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'primitives.json'
            report = primitive_probe(path, device='cpu', lengths=(17, 513), heads=2, dim=16)
            self.assertEqual(json.loads(path.read_text())['status'], 'complete')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(report['cases']), 4)
        for case in report['cases']:
            for key in ('qk_error', 'softmax_same_scores_error', 'pv_same_probabilities_error',
                        'exp_sum_same_scores_error', 'sdpa_error', 'explicit_gpu_error'):
                self.assertLess(case[key], 1e-10)

    def test_tiny_actual_model_cpu_reference_subset_and_full(self):
        fixture = fixtures.LivePrecisionTests()
        cfg, rows, binding = fixture.data()
        prior = dict(policy='target-fp64-chunked-reference-v1', status='failed', stage='finished', planned=8,
                     failures=[dict(prompt_id='cal1')])
        report = {}
        with tempfile.TemporaryDirectory() as folder, patch('torch.cuda.synchronize'):
            code = run_recovery(fixture.model(), rows, binding, cfg, report, Path(folder), prior)
            subset = json.loads((Path(folder) / 'subset/preflight-summary.json').read_text())
            full = json.loads((Path(folder) / 'full/preflight-summary.json').read_text())
        self.assertEqual(code, 0, report)
        self.assertEqual(subset['planned'], 2)
        self.assertEqual(full['planned'], 8)
        self.assertEqual(full['counts'], {'passed': 8})
        self.assertEqual(full['policy'], 'target-fp64-cpu-attention-reference-v2')
        self.assertGreater(full['cpu_attention']['calls'], 0)
        self.assertEqual(full['cpu_attention']['calls'], full['cpu_attention']['last_query_checks'])
        self.assertEqual(full['baseline']['tolerance']['alignment_tv'], 1e-6)

    def test_failed_subset_blocks_full_preflight(self):
        def failed(mod, natural, binding, cfg, report, out, **kwargs):
            report['status'] = 'failed'
            return 2
        report = {}
        with tempfile.TemporaryDirectory() as folder, \
                patch('signal_study.cpu_recovery.run_live_preflight', side_effect=failed) as run:
            self.assertEqual(run_recovery(None, [], [], {}, report, Path(folder), {}), 2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(report['full_preflight'], 'not run')

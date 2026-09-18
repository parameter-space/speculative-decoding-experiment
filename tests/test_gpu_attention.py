import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from signal_study.gpu_attention import exp_sum_attention, gpu_attention, probe_gpu_reference
from signal_study.live_precision import run_live_preflight
import test_live_precision as fixtures


class GPUAttentionReferenceTests(unittest.TestCase):
    def test_long_odd_even_causal_physical_and_cached_shapes(self):
        torch.set_num_threads(2)
        rng = torch.Generator().manual_seed(91)
        for length in (471, 512, 513, 719, 769, 770):
            k, v = (torch.randn(1, 2, length, 16, dtype=torch.float64, generator=rng) for _ in range(2))
            for kind in ('causal', 'bool', 'float', 'cached', 'empty'):
                q = torch.randn(1, 2, 1 if kind == 'cached' else 35, 16,
                                dtype=torch.float64, generator=rng)
                mask = None
                if kind != 'causal':
                    mask = torch.ones(1, 1, q.shape[-2], length, dtype=torch.bool)
                    mask[..., 2::7] = False
                    if kind == 'empty':
                        mask[..., 0, :] = False
                    if kind == 'float':
                        mask = torch.zeros_like(mask, dtype=torch.float64).masked_fill(~mask, -torch.inf)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                              is_causal=kind == 'causal', scale=.3)
                # The replacement must not accidentally use the suspect implementation.
                with patch.object(torch.Tensor, 'softmax', side_effect=AssertionError('forbidden')):
                    actual = exp_sum_attention(q, k, v, mask, is_causal=kind == 'causal', scale=.3, chunk=7)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_crosschecks_restore_and_leave_drafter_untouched(self):
        native = F.scaled_dot_product_attention
        q = torch.randn(1, 2, 513, 16, dtype=torch.float64)
        draft = q.float()
        expected_draft = native(draft, draft, draft)
        report = {}
        with self.assertRaisesRegex(RuntimeError, 'stop'), gpu_attention(report):
            actual = F.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.testing.assert_close(actual, exp_sum_attention(q, q, q, is_causal=True), rtol=0, atol=0)
            torch.testing.assert_close(F.scaled_dot_product_attention(draft, draft, draft), expected_draft,
                                       rtol=0, atol=0)
            self.assertEqual(report['calls'], 1)
            self.assertEqual(len(report['full_checks']), 1)
            raise RuntimeError('stop')
        self.assertIs(F.scaled_dot_product_attention, native)

    def test_faulty_replacement_stops_on_cpu_crosscheck(self):
        q = torch.randn(1, 2, 3, 8, dtype=torch.float64)
        def wrong(*args, **kwargs):
            return exp_sum_attention(*args, **kwargs) + .01
        with patch('signal_study.gpu_attention.exp_sum_attention', side_effect=wrong), gpu_attention({}):
            with self.assertRaisesRegex(ValueError, 'vs CPU attention failed'):
                F.scaled_dot_product_attention(q, q, q)

    def test_probe_and_rng(self):
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            result = probe_gpu_reference(Path(folder) / 'probe.json', device='cpu', lengths=(513, 769),
                                         heads=2, dim=16)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['cases']), 6)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_failed_probe_is_saved_and_blocks_following_cases(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('signal_study.gpu_attention.error', return_value=.1):
            path = Path(folder) / 'probe.json'
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                probe_gpu_reference(path, device='cpu', lengths=(3, 5), heads=1, dim=8)
            report = json.loads(path.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(len(report['cases']), 1)

    def test_tiny_actual_model_frozen_full_preflight(self):
        fixture = fixtures.LivePrecisionTests()
        cfg, rows, binding = fixture.data()
        report = {}
        with tempfile.TemporaryDirectory() as folder, patch('torch.cuda.synchronize'):
            code = run_live_preflight(fixture.model(), rows, binding, cfg, report, Path(folder),
                                      reference=True, gpu_reference=True)
            summary = json.loads((Path(folder) / 'preflight-summary.json').read_text())
        self.assertEqual(code, 0, report)
        self.assertEqual(summary['counts'], {'passed': 8})
        self.assertEqual(summary['policy'], 'target-fp64-exp-sum-attention-reference-v3')
        self.assertEqual(summary['baseline']['tolerance']['alignment_tv'], 1e-6)
        self.assertGreater(summary['gpu_attention']['calls'], 0)
        self.assertEqual(report['baseline_sdpa'], 'gpu-exp-sum-fp64')

    def test_incompatible_modes_fail_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'without another attention override'):
            run_live_preflight(None, [], [], {}, {}, None, reference=True,
                               cpu_reference=True, gpu_reference=True)

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from signal_study.reference_audit import explicit_attention, audit_attention, audit_rows, probe_snapshot
from signal_study.reference_precision import reference_operators
from signal_study.live_precision import live_target_precision, run_live_preflight
from signal_study.capture import capture_prompt
from signal_study.validation import validate_snapshot, ValidationError
import test_live_precision as fixtures


class ReferenceAuditTests(unittest.TestCase):
    def test_logit_gate_failure_retains_probability_measurements(self):
        mod = fixtures.LivePrecisionTests().model()
        with torch.inference_mode(), patch('torch.cuda.synchronize'), reference_operators(mod), \
                live_target_precision(mod, target_dtype=torch.float64):
            snap, _ = capture_prompt(mod, dict(prompt='toy'), 32)
            a = torch.zeros(1, 64, dtype=torch.float64)
            b = a.clone()
            b[0, 0] = 1
            checks = {}
            with patch('signal_study.validation.target_logits', side_effect=[a, b]):
                with self.assertRaisesRegex(ValidationError, 'logical vs physical'):
                    validate_snapshot(mod, snap, dict(repeat=1e-6, alignment=1e-6, alignment_tv=1e-6),
                                      diagnostics=checks)
        self.assertEqual(checks['p_fresh_cached'], 1)
        self.assertGreater(checks['p_fresh_cached_TV'], 1e-6)
        self.assertIn('p_sum', checks)

    def test_explicit_matches_math_sdpa_masked_causal_and_chunk_tail(self):
        torch.manual_seed(21)
        for length in (1, 5, 33):
            q = torch.randn(1, 4, length, 8, dtype=torch.float64)
            k, v = (torch.randn(1, 4, 39, 8, dtype=torch.float64) for _ in range(2))
            for kind in ('causal', 'bool', 'float', 'empty'):
                mask = None
                if kind != 'causal':
                    mask = torch.ones(1, 1, length, 39, dtype=torch.bool)
                    mask[..., 2::3] = False
                    if kind == 'empty':
                        mask[..., 0, :] = False
                    if kind == 'float':
                        mask = torch.zeros_like(mask, dtype=torch.float64).masked_fill(~mask, torch.finfo(torch.float64).min)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=kind == 'causal', scale=.3)
                actual = explicit_attention(q, k, v, mask, is_causal=kind == 'causal', scale=.3, chunk=3)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_auditor_returns_original_output_and_restores_even_on_exception(self):
        q = torch.randn(1, 2, 5, 8, dtype=torch.float64)
        native = F.scaled_dot_product_attention
        report = {}
        def faulty(*args, **kwargs):
            return native(*args, **kwargs) + .01
        with patch.object(F, 'scaled_dot_product_attention', faulty):
            with self.assertRaisesRegex(RuntimeError, 'test'), audit_attention(report):
                actual = F.scaled_dot_product_attention(q, q, q)
                torch.testing.assert_close(actual, faulty(q, q, q), rtol=0, atol=0)
                raise RuntimeError('test')
            self.assertIs(F.scaled_dot_product_attention, faulty)
        self.assertIs(F.scaled_dot_product_attention, native)
        self.assertEqual(report['event_count'], 1)
        self.assertGreater(report['events'][0]['cpu_last_query']['sdpa_error'], .009)
        self.assertLess(report['events'][0]['cpu_last_query']['explicit_error'], 1e-12)

    def test_auditor_does_not_compare_drafter_dtype(self):
        q = torch.randn(1, 2, 5, 8)
        report = {}
        expected = F.scaled_dot_product_attention(q, q, q)
        with audit_attention(report):
            actual = F.scaled_dot_product_attention(q, q, q)
        self.assertEqual(report['calls'], 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_selection_keeps_all_failures_plus_one_control_and_rejects_unknown(self):
        rows = [dict(prompt_id=str(i)) for i in range(8)]
        prior = dict(policy='target-fp64-chunked-reference-v1', status='failed', stage='finished', planned=8,
                     failures=[dict(prompt_id='1'), dict(prompt_id='3')])
        self.assertEqual([r['prompt_id'] for r in audit_rows(rows, prior)], ['1', '3', '0'])
        prior['failures'].append(dict(prompt_id='absent'))
        with self.assertRaises(ValueError):
            audit_rows(rows, prior)

    def test_probe_preserves_cache_and_measures_double_repeats(self):
        mod = fixtures.LivePrecisionTests().model()
        with torch.inference_mode(), patch('torch.cuda.synchronize'), reference_operators(mod), \
                live_target_precision(mod, target_dtype=torch.float64):
            snap, _ = capture_prompt(mod, dict(prompt='toy'), 32)
            before = [(k.clone(), v.clone()) for k, v in snap['v_cache']['layers']]
            report = probe_snapshot(mod, snap)
        for a, b in zip(before, snap['v_cache']['layers']):
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        for key in ('fresh_cached', 'fresh_repeat', 'cached_repeat', 'clean_full_split', 'fresh_physical', 'cached_physical'):
            self.assertLess(report[key]['max_logit_error'], 1e-12)

    def test_tiny_selected_audit_reports_subset_not_full_preflight(self):
        cfg, rows, binding = fixtures.LivePrecisionTests().data()
        prior = dict(policy='target-fp64-chunked-reference-v1', status='failed', stage='finished', planned=8,
                     failures=[dict(prompt_id='cal1')])
        report = {}
        with tempfile.TemporaryDirectory() as folder, patch('torch.cuda.synchronize'):
            code = run_live_preflight(fixtures.LivePrecisionTests().model(), rows, binding, cfg, report,
                                      Path(folder), reference=True, audit_prior=prior)
            summary = json.loads((Path(folder) / 'preflight-summary.json').read_text())
        self.assertEqual(code, 0, report)
        self.assertEqual(summary['planned'], 2)
        self.assertEqual(summary['scope'], 'reference_failure_audit_only_not_full_preflight')
        self.assertGreater(summary['attention_audit']['calls'], 0)
        self.assertEqual(summary['attention_audit']['event_count'], 0)
        self.assertEqual(len(summary['audit_cases']), 2)
        self.assertTrue(all('alignment_probe' in i for i in summary['audit_cases']))

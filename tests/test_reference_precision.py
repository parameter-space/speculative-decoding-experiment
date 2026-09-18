import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from signal_study.capture import capture_prompt, target_logits
from signal_study.live_precision import live_target_precision, run_live_preflight
from signal_study.reference_precision import linear64, reference_operators, rms64, rope64
from signal_study.validation import error
import test_live_precision as fixtures


class ReferencePrecisionTests(unittest.TestCase):
    def test_chunked_linear_matches_dense_double_without_weight_mutation(self):
        torch.manual_seed(4)
        for bias in (False, True):
            layer = torch.nn.Linear(17, 23, bias=bias).float()
            before = layer.weight.clone()
            x = torch.randn(2, 7, 17).double()
            audit = dict(linear_calls=0, max_temporary_weight_bytes=0)
            with torch.inference_mode():
                actual = linear64(layer, x, chunk_rows=5, audit=audit)
                expected = F.linear(x, layer.weight.double(), None if layer.bias is None else layer.bias.double())
            torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
            self.assertEqual(actual.dtype, torch.float64)
            self.assertTrue(torch.equal(layer.weight, before))
            self.assertEqual(layer.weight.dtype, torch.float32)
            self.assertEqual(audit['linear_calls'], 1)
            self.assertLessEqual(audit['max_temporary_weight_bytes'], 5 * 17 * 8)

    def test_error_preserves_sub_fp32_difference(self):
        a = torch.tensor([1.0], dtype=torch.float64)
        b = a + 1e-10
        self.assertGreater(error(a, b), 0)
        self.assertEqual(error(a.float(), b.float()), 0)

    def test_actual_generation_has_double_target_bf16_draft_and_restores(self):
        mod = fixtures.LivePrecisionTests().model()
        before = {name: p.clone() for name, p in mod.named_parameters()}
        with torch.inference_mode(), patch('torch.cuda.synchronize'):
            with reference_operators(mod, chunk_rows=13) as arithmetic, \
                    live_target_precision(mod, target_dtype=torch.float64) as audit:
                snap, _ = capture_prompt(mod, dict(prompt='toy'), 32)
                self.assertIsNotNone(snap)
                self.assertEqual({t.dtype for pair in snap['v_cache']['layers'] for t in pair}, {torch.float64})
                self.assertEqual({t.dtype for pair in snap['d_cache']['layers'] for t in pair}, {torch.bfloat16})
                self.assertEqual(snap['G'].dtype, torch.bfloat16)
                self.assertEqual(snap['p_src_logits'].dtype, torch.float64)
                self.assertLess(error(target_logits(mod, snap), target_logits(mod, snap, cached=True)), 1e-12)
                self.assertGreater(audit['checked_target_cache_calls'], 0)
                self.assertGreater(arithmetic['linear_calls'], 0)
                self.assertNotIn(torch.float64, {p.dtype for p in mod.parameters()})
        for name, p in mod.named_parameters():
            self.assertEqual(p.dtype, before[name].dtype)
            self.assertTrue(torch.equal(p, before[name]), name)
        self.assertTrue(all('forward' not in m.__dict__ for m in mod.v_base.modules()))

    def test_reference_operators_restore_after_exception(self):
        mod = fixtures.LivePrecisionTests().model()
        before = {id(m): m.forward for m in mod.v_base.modules()}
        with self.assertRaisesRegex(RuntimeError, 'test'), reference_operators(mod), \
                live_target_precision(mod, target_dtype=torch.float64):
            raise RuntimeError('test')
        for m in mod.v_base.modules():
            self.assertEqual(m.forward, before[id(m)])

    def test_full_preflight_uses_double_real_generation(self):
        cfg, natural, binding = fixtures.LivePrecisionTests().data()
        report = {}
        with tempfile.TemporaryDirectory() as folder, patch('torch.cuda.synchronize'):
            code = run_live_preflight(fixtures.LivePrecisionTests().model(), natural, binding, cfg, report,
                                      Path(folder), reference=True)
            summary = json.loads((Path(folder) / 'preflight-summary.json').read_text())
        self.assertEqual(code, 0, report)
        self.assertEqual(summary['counts'], {'passed': 8})
        self.assertEqual(summary['policy'], 'target-fp64-chunked-reference-v1')
        self.assertLess(summary['max_target_TV'], 1e-12)
        self.assertEqual(summary['baseline']['tolerance']['alignment_tv'], 1e-6)
        self.assertTrue(all(i['target_kv_dtypes'] == ['torch.float64'] for i in report['endpoints']))

    def test_norm_and_rope_keep_double_precision(self):
        mod = fixtures.LivePrecisionTests().model()
        decoder = mod.v_base.get_decoder()
        x = torch.randn(1, 3, 32, dtype=torch.float64)
        expected = x / (x.square().mean(-1, keepdim=True) + decoder.norm.variance_epsilon).sqrt()
        expected *= decoder.norm.weight.double()
        torch.testing.assert_close(rms64(decoder.norm, x), expected, atol=1e-15, rtol=1e-15)
        positions = torch.tensor([[0, 123, 2047]])
        rope = decoder.rotary_emb
        for rope_type in ('default', 'llama3'):
            rope.rope_type = rope_type
            cos, sin = rope64(rope, x, positions)
            freq = (rope.inv_freq.double()[None, :, None] @ positions.double()[:, None, :]).transpose(1, 2)
            emb = torch.cat((freq, freq), -1)
            self.assertEqual(cos.dtype, torch.float64)
            torch.testing.assert_close(cos, emb.cos() * rope.attention_scaling, rtol=0, atol=0)
            torch.testing.assert_close(sin, emb.sin() * rope.attention_scaling, rtol=0, atol=0)

    def test_unsupported_attention_and_rope_rejected_and_restored(self):
        mod = fixtures.LivePrecisionTests().model()
        decoder = mod.v_base.get_decoder()
        decoder.config._attn_implementation = 'eager'
        with self.assertRaisesRegex(ValueError, 'SDPA'), reference_operators(mod):
            pass
        decoder.config._attn_implementation = 'sdpa'
        decoder.rotary_emb.rope_type = 'dynamic'
        with self.assertRaisesRegex(ValueError, 'RoPE'), reference_operators(mod):
            pass
        self.assertTrue(all('forward' not in m.__dict__ for m in mod.v_base.modules()))

"""Actual pinned SD² decoder/capture tests with tiny random CPU models, not 8B results."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import DynamicCache, LlamaConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/SD-square"))
import main as upstream
from src.models.llama import LlamaForCausalLM

from signal_study.capture import Capture, capture_prompt, draft_logits, synthetic_snapshot, target_logits
from signal_study.common import overlap_reference
from signal_study.state import cache_to_cpu, logical_prefix, overlap, distribution, restore_cache
from signal_study.validation import validate_snapshot, ValidationError, baseline_tests
from signal_study.run import case_rows


class ToyTokenizer:
    eos_token = "END"

    def apply_chat_template(self, messages, **kwargs):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, tokens, **kwargs):
        return " ".join(str(x) for x in tokens)


def tiny_model():
    def loader(name, float16=False):
        target = "3.1" in name
        config = LlamaConfig(vocab_size=64, hidden_size=32 if target else 16,
                             intermediate_size=64 if target else 32, num_hidden_layers=4 if target else 2,
                             num_attention_heads=4 if target else 2, num_key_value_heads=2,
                             max_position_embeddings=256, eos_token_id=63, pad_token_id=0,
                             attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        return LlamaForCausalLM(config), "llama"
    torch.manual_seed(5)
    with patch.object(upstream, "load_model", loader), patch.object(upstream.AutoTokenizer, "from_pretrained", return_value=ToyTokenizer()):
        mod = upstream.TrainingModule(v_layer=[0, 1, 3], ngram=4)
    mod.eval().requires_grad_(False)
    mod.eot_id = 999  # no random EOS; this affects only the tiny fixture
    mod.pad_token_id = 0
    with torch.no_grad():
        mod.latent_mod_prep.w_guide.weight.normal_(std=.08)
    upstream.PRETTY_PRINT = False
    return mod


class TinyUpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.mod = tiny_model()

    def capture(self):
        with patch("torch.cuda.synchronize"), torch.inference_mode():
            return capture_prompt(self.mod, {"prompt": "toy"}, 32)[0]

    def test_real_normal_boundary_roundtrip(self):
        with torch.inference_mode():
            snap = self.capture()
            self.assertEqual(snap["round"], 2)
            self.assertEqual(snap["source_pos"] + 1, len(snap["prefix"]) - 1)
            self.assertGreater(snap["pending_ids"].shape[1], 1)
            p, checks = validate_snapshot(self.mod, snap, dict(repeat=1e-6, alignment=1e-5, alignment_tv=1e-6))
            self.assertLess(checks["restore"], 1e-6)
            q = distribution(draft_logits(self.mod, snap))
            self.assertAlmostEqual(overlap(p, q), overlap_reference(p.numpy(), q.numpy()), places=6)

    def test_A_B_A_changes_signal_not_cached_state(self):
        with torch.inference_mode():
            snap = self.capture()
            a = draft_logits(self.mod, snap)
            b = draft_logits(self.mod, snap, -snap["G"])
            a2 = draft_logits(self.mod, snap)
            self.assertTrue(torch.equal(a, a2))
            self.assertGreater(float((a - b).abs().max()), 1e-5)

    def test_cache_restores_independent_storage(self):
        with torch.inference_mode():
            snap = self.capture()
            first = restore_cache(snap["d_cache"], "cpu")
            second = restore_cache(snap["d_cache"], "cpu")
            before = second.key_cache[0].clone()
            first.key_cache[0].zero_()
            self.assertTrue(torch.equal(second.key_cache[0], before))
            self.assertTrue(torch.equal(snap["d_cache"]["layers"][0][0], before))

    def test_synthetic_prefix_and_cached_target_agree(self):
        with torch.inference_mode():
            row = dict(prefix_ids=[1, 4, 9, 6, 3, 7])
            snap = synthetic_snapshot(self.mod, row)
            self.assertEqual(snap["pending_ids"].shape[1], 1)
            self.assertEqual(snap["state_origin"], "synthetic_teacher_prefix")
            self.assertLess(float((target_logits(self.mod, snap) - target_logits(self.mod, snap, True)).abs().max()), 1e-5)

    def test_corrupted_mask_cannot_pass_prefix_check(self):
        with torch.inference_mode():
            snap = self.capture()
            positions = snap["positions"].clone()
            positions[0, snap["curr"]] += 1
            with self.assertRaises(ValueError):
                logical_prefix(snap["ids"], snap["mask"], positions, snap["curr"])

    def test_non_mutating_hooks_preserve_generation(self):
        ids, mask = self.mod.prep_for_gen(["toy"])
        with patch("torch.cuda.synchronize"), torch.inference_mode():
            torch.manual_seed(11)
            a = self.mod.generate(ids, mask, max_new_tokens=32)
            torch.manual_seed(11)
            with Capture(self.mod, stop=False):
                b = self.mod.generate(ids, mask, max_new_tokens=32)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1]["attention_mask"], b[1]["attention_mask"]))

    def test_corrupted_guide_rejected_by_identity_gate(self):
        with torch.inference_mode():
            snap = self.capture()
            snap["G"].neg_()
            with self.assertRaises(ValidationError):
                validate_snapshot(self.mod, snap, dict(repeat=1e-6, alignment=1e-5, alignment_tv=1e-6))

    def test_R_uses_pre_block_states(self):
        ids = torch.tensor([[1, 2, 3]])
        seen = {}
        handles = []
        for index in [0, 1, 3]:
            def hook(module, args, i=index):
                seen[i] = args[0].clone()
            handles.append(self.mod.v_base.get_decoder().layers[index].register_forward_pre_hook(hook))
        try:
            with torch.inference_mode(), Capture(self.mod, stop=False) as recorder:
                self.mod.v_base.get_decoder()(ids, compute_guidance=True, use_cache=False)
            self.assertTrue(torch.equal(recorder.last_r, torch.cat([seen[i] for i in [0, 1, 3]], -1)))
        finally:
            for handle in handles:
                handle.remove()

    def test_greedy_baseline_and_empirical_tolerance_gate(self):
        cfg = dict(seed=11, repeat_logit_cap=1e-6, alignment_logit_cap=.01, alignment_tv_cap=.001)
        with patch("torch.cuda.synchronize"), torch.inference_mode():
            report, tolerance = baseline_tests(self.mod, ["first", "second"], cfg)
        self.assertEqual(len(report), 2)
        self.assertEqual(report[0]["greedy_AR_identity"], "passed")
        self.assertLessEqual(tolerance["alignment"], .01)

    def test_all_endpoint_conditions_and_signs(self):
        with torch.inference_mode():
            snap = self.capture()
            tol = dict(repeat=1e-6, alignment=1e-5, alignment_tv=1e-6)
            p, checks = validate_snapshot(self.mod, snap, tol)
            row = dict(prompt_id="test", domain="math", split="smoke")
            rows, detail = case_rows(self.mod, row, snap, p, checks, tol, mean=torch.zeros_like(snap["G"]))
        by_name = {r["condition"]: r for r in rows}
        self.assertEqual(by_name["mean_rms"]["status"], "NA")
        self.assertEqual(by_name["matched_donor"]["status"], "NA")
        self.assertEqual(by_name["original"]["delta_A_vs_original"], 0)
        self.assertEqual(by_name["mean"]["delta_A_vs_original"], -by_name["mean"]["U_original_over_control"])
        self.assertIn("A_B_A", detail["tests"])

    def test_binding_swap_reports_direction(self):
        with torch.inference_mode():
            left = dict(prompt_id="left", prefix_ids=[1, 2, 3, 4], domain="binding", split="smoke", label="red", label_id=10)
            right = dict(prompt_id="right", prefix_ids=[1, 3, 2, 4], domain="binding", split="smoke", label="blue", label_id=11)
            a, b = synthetic_snapshot(self.mod, left), synthetic_snapshot(self.mod, right)
            tol = dict(repeat=1e-6, alignment=1e-5, alignment_tv=1e-6)
            p, checks = validate_snapshot(self.mod, a, tol)
            rows, _ = case_rows(self.mod, left, a, p, checks, tol, donor_g=b["G"], partner=right)
        swap = next(r for r in rows if r["condition"] == "binding_swap")
        self.assertEqual(swap["donor_label_id"], 11)
        self.assertEqual(swap["donor_prompt_id"], "right")
        self.assertIn("donor_label_probability_change", swap)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from signal_study.state import distribution, overlap, probability_audit, validate_probability


class ProbabilityTests(unittest.TestCase):
    def test_large_vocabulary_peaked_logits(self):
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            logits = torch.zeros(1, 128256, dtype=torch.bfloat16)
            logits[0, 0] = 20
            original = logits.clone()
            p = distribution(logits, context="test.target")
            audit = probability_audit(logits, context="test.target")
            self.assertEqual(p.dtype, torch.float64)
            self.assertEqual(p.device.type, "cpu")
            self.assertEqual(p.shape, (128256,))
            self.assertLess(abs(p.sum().item() - 1), 1e-10)
            self.assertTrue(audit["float64"]["fp64_sum_gate_passed"])
            # Do not require FP32 failure: reduction behavior depends on the PyTorch build.
            self.assertIn("native_sum_error", audit["float32"])
            self.assertTrue(torch.equal(logits, original))
            self.assertAlmostEqual(overlap(p, p), 1, places=10)
        finally:
            torch.set_num_threads(previous)

    def test_only_single_vocabulary_vector_accepted(self):
        expected = distribution(torch.arange(5, dtype=torch.float32))
        for shape in ((5,), (1, 5), (1, 1, 5)):
            torch.testing.assert_close(distribution(torch.arange(5.).reshape(shape)), expected)
        for shape in ((), (0,), (2, 5), (1, 2, 5), (0, 5)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape="):
                distribution(torch.zeros(shape))

    def test_bad_logits_rejected(self):
        for logits in (torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                       torch.tensor([-float("inf")]), torch.ones(4, dtype=torch.long)):
            with self.assertRaises(ValueError):
                distribution(logits)

    def test_sum_gate_not_relaxed_or_renormalized(self):
        p = torch.tensor([.5, .5001], dtype=torch.float64)
        before = p.clone()
        with self.assertRaisesRegex(ValueError, r"test.bad: probabilities do not sum to one.*sum=.*cap="):
            validate_probability(p, context="test.bad")
        self.assertTrue(torch.equal(p, before))
        for p in (torch.tensor([-.1, 1.1]), torch.tensor([float("nan")]), torch.ones(1, 1)):
            with self.assertRaises(ValueError):
                validate_probability(p)

    def test_overlap_uses_double_accumulation_and_checks_vocabulary(self):
        p, q = distribution(torch.tensor([1., 2., 3.])), distribution(torch.tensor([3., 2., 1.]))
        self.assertEqual(overlap(p, q), torch.minimum(p, q).sum(dtype=torch.float64).item())
        with self.assertRaisesRegex(ValueError, "vocabulary mismatch"):
            overlap(p, distribution(torch.ones(2)))

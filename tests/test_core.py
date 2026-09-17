import unittest

import numpy as np

from signal_study.common import choose_donor, overlap_reference, validate_splits
from signal_study.data import allocate


class CoreTests(unittest.TestCase):
    def test_overlap_identity_and_tv(self):
        p, q = np.array([.1, .2, .7]), np.array([.2, .4, .4])
        self.assertAlmostEqual(overlap_reference(p, q), 1 - abs(p - q).sum() / 2)
        self.assertAlmostEqual(overlap_reference(p, p), 1)

    def test_overlap_rejects_invalid_inputs(self):
        for p, q in (([.1, .1], [.5, .5]), ([float('nan'), 1], [0, 1]), ([1, -1], [0, 1]), ([1], [0, 1])):
            with self.assertRaises(ValueError):
                overlap_reference(p, q)

    def test_donor_matching_and_tie_break(self):
        recipient = dict(prompt_id="recipient", domain="math", z_id=3, p_src_argmax=4,
                         source_length=129, p_src_entropy=1.1, p_src_z=.3)
        donor = dict(recipient, split="calibration", prompt_id="b", source_pos=128, p_src_z=.4)
        earlier = dict(donor, prompt_id="a")
        wrong_split = dict(earlier, split="holdout")
        self.assertEqual(choose_donor(recipient, [donor, earlier, wrong_split])["prompt_id"], "a")
        self.assertIsNone(choose_donor(recipient, [dict(donor, domain="dialogue")]))

    def test_no_self_donor_or_wrong_bin(self):
        r = dict(prompt_id="r", domain="math", z_id=3, p_src_argmax=4, source_length=63, p_src_entropy=.49, p_src_z=.3)
        d = dict(r, prompt_id="d", split="calibration", source_pos=1)
        for bad in (dict(d, prompt_id="r"), dict(d, source_length=64), dict(d, p_src_entropy=.5)):
            self.assertIsNone(choose_donor(r, [bad]))

    def test_content_leakage_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits([dict(prompt_id="a", prompt="same", split="calibration"),
                             dict(prompt_id="b", prompt="same", split="smoke")])

    def test_base_problem_leakage_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits([dict(prompt_id="a", prompt="first", split="train", base_problem_id="x"),
                             dict(prompt_id="b", prompt="second", split="holdout", base_problem_id="x")])

    def test_small_dataset_never_duplicated(self):
        cfg = dict(smoke_per_domain=4, pilot_per_domain=32, holdout_per_domain_max=128)
        rows = allocate([dict(prompt_id=str(i)) for i in range(150)], cfg)
        self.assertEqual(len(rows), 150)
        self.assertEqual(sum(r["split"] == "holdout" for r in rows), 114)
        self.assertEqual(len({r["prompt_id"] for r in rows}), 150)


if __name__ == "__main__":
    unittest.main()

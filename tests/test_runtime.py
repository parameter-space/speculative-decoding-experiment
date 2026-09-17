import unittest

import torch

from signal_study.runtime import sdpa_context


class RuntimePolicyTests(unittest.TestCase):
    def test_math_context_and_restore(self):
        def flags():
            return (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                    torch.backends.cuda.math_sdp_enabled())
        before = flags()
        with sdpa_context("math"):
            self.assertEqual(flags(), (False, False, True))
        self.assertEqual(flags(), before)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            sdpa_context("typo")

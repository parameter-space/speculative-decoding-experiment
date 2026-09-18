"""Offline orchestration test. CUDA/platform checks are mocked; model math stays CPU."""
from contextlib import ExitStack, nullcontext
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from signal_study.common import append_jsonl, digest, read_json, write_json
from signal_study.run import run
from signal_study.validation import ValidationError
from signal_study.state import METRIC_POLICY
from test_tiny_upstream import tiny_model


class PipelineTests(unittest.TestCase):
    def test_reports_and_all_conditions_end_to_end_with_tiny_cpu_fixture(self):
        self._run_tiny_pipeline("default")

    def test_math_policy_full_pipeline_and_manifest(self):
        self._run_tiny_pipeline("math")

    def test_fp64_reference_effects_and_policy_labels(self):
        self._run_tiny_pipeline("default", reference=True)

    def test_fp64_failure_invalidates_earlier_rows_and_restores(self):
        self._run_tiny_pipeline("math", reference=True, fail=True)

    def test_fp64_sharding_rejected(self):
        args = SimpleNamespace(config=Path(__file__).resolve().parents[1] / "configs/smoke.json",
                               output="unused", data_dir="unused", target_fp64_reference=True,
                               shard_index=0, shard_count=2)
        with self.assertRaisesRegex(ValueError, "single unsharded"):
            run(args)

    def _run_tiny_pipeline(self, policy, reference=False, fail=False):
        mod = tiny_model()
        if reference:
            mod.v_base.half()
            mod.d_base.bfloat16()
            mod.guidance_embd_layer.bfloat16()
            mod.latent_mod_prep.bfloat16()
        before = {name: p.clone() for name, p in mod.named_parameters()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = read_json(Path(__file__).resolve().parents[1] / "configs/smoke.json")
            cfg.update(smoke_per_domain=1, calibration_count=2, binding_pairs=1, max_new_tokens=32)
            natural = [dict(prompt_id=f"cal-{i}", prompt=f"calibration {i}", domain="dialogue", split="calibration") for i in range(2)]
            natural += [dict(prompt_id=f"eval-{domain}", prompt=f"evaluation {domain}", domain=domain, split="smoke")
                        for domain in ("dialogue", "math", "code", "summary")]
            binding = [dict(prompt_id=f"pair-{side}", prompt=f"binding {side}", prefix_ids=ids,
                            base_problem_id="pair", domain="binding", split="smoke", label=side, label_id=label)
                       for side, ids, label in (("A", [1, 2, 3, 4], 10), ("B", [1, 3, 2, 4], 11))]
            write_json(root / "config.json", cfg)
            for name, rows in (("natural", natural), ("binding", binding)):
                for row in rows:
                    append_jsonl(root / f"{name}.jsonl", row)
            write_json(root / "manifest.json", dict(config_hash=digest(cfg), natural_hash=digest(natural),
                       binding_hash=digest(binding), target_revision="tiny-fixture", drafter_revision="tiny-fixture"))
            args = SimpleNamespace(config=root / "config.json", output=root / "out", data_dir=root,
                                   upstream="unused-test-fixture", save_snapshots=False, target_fp64_reference=reference)
            with ExitStack() as stack:
                stack.enter_context(patch.dict("os.environ", {"SLURM_JOB_ID": "offline-test", "S1_SDPA_BACKEND": policy}))
                stack.enter_context(patch("signal_study.run.socket.gethostname", return_value="ariel-k2"))
                stack.enter_context(patch("signal_study.run.environment", return_value={"kind": "MOCK_CPU_TEST"}))
                stack.enter_context(patch("signal_study.run.load", return_value=(mod, {"kind": "tiny random CPU fixture"})))
                real_autocast = torch.autocast
                def cpu_autocast(*a, **kw):
                    device = a[0] if a else kw.get("device_type")
                    return real_autocast(*a, **kw) if reference and device == "cpu" else nullcontext()
                stack.enter_context(patch("torch.autocast", side_effect=cpu_autocast))
                if fail:
                    from signal_study.run import case_rows
                    calls = []
                    def fault(*a, **kw):
                        calls.append(1)
                        if len(calls) == 2:
                            raise ValidationError("deliberate effect failure")
                        return case_rows(*a, **kw)
                    stack.enter_context(patch("signal_study.run.case_rows", side_effect=fault))
                for name, value in (("is_available", True), ("device_count", 1), ("is_bf16_supported", True),
                                    ("get_rng_state_all", []), ("max_memory_allocated", 0), ("max_memory_reserved", 0)):
                    stack.enter_context(patch(f"torch.cuda.{name}", return_value=value))
                for name in ("synchronize", "reset_peak_memory_stats"):
                    stack.enter_context(patch(f"torch.cuda.{name}"))
                code = run(args)
            with (root / "out/results/S1_endpoint.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            report = read_json(root / "out/reports/tests.json")
            expected_policy = "s1-target-fp64-exp-sum-v1" if reference else "s1-official-eval-v1"
            self.assertTrue(all(r["execution_policy_id"] == expected_policy for r in rows))
            for name, p in mod.named_parameters():
                self.assertEqual(p.dtype, before[name].dtype)
                self.assertTrue(torch.equal(p, before[name]), name)
            if fail:
                self.assertEqual(code, 2)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(all(r["run_valid"] == "False" for r in rows))
                self.assertEqual(len(rows), 6)
                return
            self.assertEqual(len(rows), 26)
            self.assertTrue(all(r["run_valid"] == "True" for r in rows))
            self.assertEqual({r["condition"] for r in rows},
                             {"original", "self_copy", "mean", "mean_rms", "matched_donor", "binding_swap"})
            for row in rows:
                if row["condition"] in ("original", "self_copy"):
                    self.assertEqual(float(row["U_original_over_control"]), 0)
            if reference:
                self.assertEqual(report["dtype_audit"]["target_dtype"], "torch.float64")
                self.assertGreater(report["gpu_attention"]["calls"], 0)
                self.assertEqual(report["tolerance"]["alignment_tv"], 1e-6)
                self.assertEqual(report["execution_policy"]["scope"], "S1_endpoint_effect_measurement")
                policy = "math"
            self.assertEqual(code, 0)
            self.assertEqual(read_json(root / "out/reports/tests.json")["status"], "complete")
            self.assertEqual(read_json(root / "out/manifests/models.json")["sdpa_kernel_policy"], policy)
            self.assertEqual(read_json(root / "out/reports/tests.json")["sdpa_kernel_policy"], policy)
            self.assertEqual(read_json(root / "out/manifests/metric_policy.json"), METRIC_POLICY)
            self.assertEqual(read_json(root / "out/manifests/models.json")["metric_policy"], METRIC_POLICY)
            self.assertEqual(read_json(root / "out/reports/tests.json")["metric_policy"], METRIC_POLICY)
            self.assertIn(METRIC_POLICY["id"], (root / "out/results/S1_endpoint.csv").read_text())
            for required in ("manifests/environment.json", "manifests/models.json", "tensor_map.json",
                             "trace/boundaries.jsonl", "results/S1_endpoint.csv", "results/S1_cases.md",
                             "reports/tests.json", "reports/HANDOFF.md"):
                self.assertTrue((root / "out" / required).is_file(), required)
            self.assertEqual(len(read_json(root / "out/results/cases.json")), 6)

    def test_no_slurm_run_stops_without_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(config=Path(__file__).resolve().parents[1] / "configs/smoke.json",
                                   output=Path(folder) / "out", data_dir=folder, upstream="unused", save_snapshots=False)
            with patch.dict("os.environ", {"SLURM_JOB_ID": ""}), patch("signal_study.run.load") as loader:
                code = run(args)
            self.assertEqual(code, 2)
            loader.assert_not_called()
            self.assertEqual(read_json(Path(folder) / "out/reports/tests.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()

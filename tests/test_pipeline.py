"""Offline orchestration test. CUDA/platform checks are mocked; model math stays CPU."""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from signal_study.common import append_jsonl, digest, read_json, write_json
from signal_study.run import run
from test_tiny_upstream import tiny_model


class PipelineTests(unittest.TestCase):
    def test_reports_and_all_conditions_end_to_end_with_tiny_cpu_fixture(self):
        mod = tiny_model()
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
                                   upstream="unused-test-fixture", save_snapshots=False)
            with ExitStack() as stack:
                stack.enter_context(patch.dict("os.environ", {"SLURM_JOB_ID": "offline-test"}))
                stack.enter_context(patch("signal_study.run.socket.gethostname", return_value="ariel-k2"))
                stack.enter_context(patch("signal_study.run.environment", return_value={"kind": "MOCK_CPU_TEST"}))
                stack.enter_context(patch("signal_study.run.load", return_value=(mod, {"kind": "tiny random CPU fixture"})))
                stack.enter_context(patch("torch.autocast", side_effect=lambda *a, **kw: nullcontext()))
                for name, value in (("is_available", True), ("device_count", 1), ("is_bf16_supported", True),
                                    ("get_rng_state_all", []), ("max_memory_allocated", 0), ("max_memory_reserved", 0)):
                    stack.enter_context(patch(f"torch.cuda.{name}", return_value=value))
                for name in ("synchronize", "reset_peak_memory_stats"):
                    stack.enter_context(patch(f"torch.cuda.{name}"))
                code = run(args)
            self.assertEqual(code, 0)
            self.assertEqual(read_json(root / "out/reports/tests.json")["status"], "complete")
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

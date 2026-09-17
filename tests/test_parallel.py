import csv
from pathlib import Path
import tempfile
import unittest

from signal_study.common import write_json, read_json
from signal_study.parallel import merge_outputs, partition_rows, worker_devices


class ParallelTests(unittest.TestCase):
    def test_partition_preserves_pairs_and_covers_exactly_once(self):
        natural = [{"prompt_id": f"n-{i}"} for i in range(16)]
        binding = [{"prompt_id": f"b-{i}-{side}", "base_problem_id": f"b-{i}"} for i in range(8) for side in ("A", "B")]
        a, b = partition_rows(natural, binding, 0, 2), partition_rows(natural, binding, 1, 2)
        self.assertEqual([len(x) for x in a], [8, 8])
        self.assertEqual([len(x) for x in b], [8, 8])
        self.assertFalse({r["prompt_id"] for r in a[0]} & {r["prompt_id"] for r in b[0]})
        self.assertFalse({r["base_problem_id"] for r in a[1]} & {r["base_problem_id"] for r in b[1]})
        self.assertEqual(len({r["prompt_id"] for r in a[1] + b[1]}), 16)

    def test_preserves_slurm_mask_without_guessing_gpu_ids(self):
        self.assertEqual(worker_devices("3,7", 2), ["3", "7"])
        for mask, count in ((None, 2), ("0,0", 2), ("0,1", 1), ("0,1,2", 3)):
            with self.assertRaises(ValueError):
                worker_devices(mask, count)

    def test_invalid_shards_rejected(self):
        for index, count in ((2, 2), (-1, 2), (0, 3)):
            with self.assertRaises(ValueError):
                partition_rows([], [], index, count)

    def build_workers(self, root, duplicate=False, mismatch=False):
        cfg = {"smoke_per_domain": 1, "binding_pairs": 1}
        for i in range(2):
            worker = root / f"worker-{i}"
            write_json(worker / "reports/tests.json", {"status": "complete"})
            write_json(worker / "manifests/run.json", cfg)
            write_json(worker / "manifests/data.json", {"fixed": 2 if mismatch and i else 1})
            write_json(worker / "manifests/models.json", {"revision": "same"})
            write_json(worker / "manifests/calibration.json", {"prompt_ids": ["c"]})
            write_json(worker / "manifests/environment.json", {"kind": "MOCK_CPU_TEST"})
            write_json(worker / "tensor_map.json", {"port": "G"})
            ids = [f"{0 if duplicate else i}-{j}" for j in range(3)]
            write_json(worker / "results/cases.json", [{"prompt_id": name} for name in ids])
            with (worker / "results/S1_endpoint.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["prompt_id", "condition", "run_valid"])
                writer.writeheader()
                writer.writerows({"prompt_id": name, "condition": "original", "run_valid": True} for name in ids)
            (worker / "results/S1_cases.md").write_text("Offline fixture", encoding="utf-8")

    def test_merge_verifies_counts_and_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.build_workers(root)
            self.assertEqual(merge_outputs(root, 2, [0, 0]), 0)
            self.assertEqual(read_json(root / "reports/tests.json")["completed_cases"], 6)

    def test_merge_rejects_duplicates_or_manifest_mismatch(self):
        for duplicate, mismatch in ((True, False), (False, True)):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.build_workers(root, duplicate, mismatch)
                with self.assertRaises(ValueError):
                    merge_outputs(root, 2, [0, 0])

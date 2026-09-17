"""Run offline unit tests and record actual results; never masquerade as a GPU smoke."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signal_study.common import write_json
from signal_study.runtime import environment

suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
write_json(ROOT / "reports/local_tests.json", {
    "kind": "offline_unit_and_tiny_random_CPU_models_NOT_real_checkpoint_results",
    "tests_run": result.testsRun, "passed": result.wasSuccessful(),
    "failures": [{"test": str(test), "details": trace} for test, trace in result.failures],
    "errors": [{"test": str(test), "details": trace} for test, trace in result.errors],
    "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
    "environment": environment(),
})
raise SystemExit(not result.wasSuccessful())

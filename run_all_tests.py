"""Unified Test Runner for MaleCNS Mario 2D Prototype.

Runs Tests A through F sequentially and produces a machine-readable test report.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_environment import run_environment_test
from tests.test_data_pipeline import run_data_pipeline_test
from tests.test_model_step import run_model_step_test
from tests.test_environment_integration import run_environment_integration_test
from tests.test_short_training import run_short_training_test
from tests.test_local_inference import run_local_inference_test
from tests.test_demonstration_dataset import run_demonstration_tests
from tests.test_comparative_training import run_comparative_training_tests
from tests.test_graph_baselines import run_graph_baseline_tests
from tests.test_ppo_trainer import run_ppo_trainer_tests


TEST_SUITE = [
    ("TEST_A_ENVIRONMENT", "Mario Gymnasium Environment Verification", run_environment_test),
    ("TEST_B_DATA_PIPELINE", "Real MaleCNS Connectome Pipeline Verification", run_data_pipeline_test),
    ("TEST_C_MODEL_STEP", "Model Forward & Backward Step with Fixed Connectome", run_model_step_test),
    ("TEST_D_INTEGRATION", "End-to-End Observation-to-Action 100 Steps", lambda: run_environment_integration_test(100)),
    ("TEST_E_TRAINING", "Short Behavioral Cloning Smoke Test", lambda: run_short_training_test(50, 10)),
    ("TEST_F_INFERENCE", "Local Inference Benchmark on Apple Silicon (MPS)", lambda: run_local_inference_test(50)),
    ("TEST_G_DEMONSTRATION", "Demonstration Collector & Dataset Validation", run_demonstration_tests),
    ("TEST_H_COMPARATIVE", "Comparative Training Pipeline & Parity Verification", run_comparative_training_tests),
    ("TEST_I_GRAPH_BASELINES", "Graph Baselines (Rewired & Random Sparse) Invariants & Parity", run_graph_baseline_tests),
    ("TEST_J_PPO_TRAINER", "PPO Reinforcement Learning Rollout, GAE & Gradient Step", run_ppo_trainer_tests),
]


def main():
    print("#" * 70)
    print("RUNNING COMPLETE MALECNS MARIO 2D VERIFICATION SUITE")
    print("#" * 70)

    results = {}
    all_passed = True
    start_total = time.time()

    for test_id, desc, fn in TEST_SUITE:
        t0 = time.time()
        print(f"\n>>> Running {test_id}: {desc}...")
        try:
            success = fn()
            status = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"!!! Error in {test_id}: {type(e).__name__}: {e}")
            status = "FAIL"
            all_passed = False
        duration = round(time.time() - t0, 3)

        results[test_id] = {
            "description": desc,
            "status": status,
            "duration_seconds": duration,
        }
        if status != "PASS":
            all_passed = False

    total_time = round(time.time() - start_total, 3)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_status": "PASS" if all_passed else "FAIL",
        "total_duration_seconds": total_time,
        "tests": results,
        "hardware": {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
        },
    }

    report_path = REPO_ROOT / "test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("VERIFICATION SUITE SUMMARY")
    print("=" * 70)
    for tid, r in results.items():
        print(f"[{r['status']}] {tid:<24} ({r['duration_seconds']:.2f}s) - {r['description']}")
    print(f"\nOverall: {report['overall_status']} in {total_time:.2f}s")
    print(f"Saved machine-readable test report to: {report_path.name}")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()


"""Unit tests for scientific ablation study and telemetry metrics.

Verifies:
1. L1/L2 Sparsity ratio computation correctness.
2. Welch's t-test and Cohen's d calculation.
3. Telemetry canvas composite rendering.
4. Parity of Model A, B, C architectures.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_scientific_ablation import compute_welch_t_test, load_model_family
from src.environment import WORLD_1_1_LANDMARKS
from src.visualization import (
    MarioTelemetryFrame,
    compute_sparsity_ratio,
    render_telemetry_canvas,
)


def test_sparsity_ratio():
    # Dense vector with equal entries
    dense = np.ones(100, dtype=np.float32)
    dense_ratio = compute_sparsity_ratio(dense)
    # L1 = 100, L2 = 10 -> ratio = 10.0
    assert math.isclose(dense_ratio, 10.0, rel_tol=1e-3)

    # Sparse vector with only 1 active entry
    sparse = np.zeros(100, dtype=np.float32)
    sparse[0] = 1.0
    sparse_ratio = compute_sparsity_ratio(sparse)
    # L1 = 1, L2 = 1 -> ratio = 1.0
    assert math.isclose(sparse_ratio, 1.0, rel_tol=1e-3)

    # Sparse ratio must be strictly lower for more sparse vector
    assert sparse_ratio < dense_ratio


def test_welch_t_test_and_cohens_d():
    # Two identical distributions -> t = 0, p = 1.0, d = 0
    samples_a = [10.0, 11.0, 9.0, 10.5, 9.5]
    samples_b = [10.0, 11.0, 9.0, 10.5, 9.5]
    res_ident = compute_welch_t_test(samples_a, samples_b)
    assert math.isclose(res_ident["t_statistic"], 0.0, abs_tol=1e-3)
    assert math.isclose(res_ident["p_value"], 1.0, abs_tol=1e-3)
    assert math.isclose(res_ident["cohens_d"], 0.0, abs_tol=1e-3)
    assert not res_ident["significant_p05"]

    # Significantly different distributions
    samples_high = [100.0, 102.0, 99.0, 101.0, 100.5]
    samples_low = [10.0, 12.0, 9.0, 11.0, 10.5]
    res_diff = compute_welch_t_test(samples_high, samples_low)
    assert res_diff["t_statistic"] > 50.0
    assert res_diff["p_value"] < 0.001
    assert res_diff["significant_p05"]
    assert res_diff["cohens_d"] > 10.0


def test_telemetry_canvas_rendering():
    dummy_rgb = np.zeros((240, 256, 3), dtype=np.uint8)
    dummy_frame = MarioTelemetryFrame(
        step=42,
        rgb_frame=dummy_rgb,
        x_pos=350,
        y_pos=100,
        action_idx=1,
        action_probs=np.array([0.05, 0.80, 0.10, 0.02, 0.03], dtype=np.float32),
        value=15.5,
        reward=2.0,
        sensory_act=np.random.randn(9201).astype(np.float32),
        core_act=np.random.randn(30000).astype(np.float32) * 0.1,
        motor_act=np.random.randn(1314).astype(np.float32),
    )

    canvas = render_telemetry_canvas(dummy_frame, dpi=60, model_name="Test Model")
    assert isinstance(canvas, np.ndarray)
    assert canvas.ndim == 3
    assert canvas.shape[2] == 3
    assert canvas.shape[0] > 100
    assert canvas.shape[1] > 100


def test_model_families_loading():
    device = torch.device("cpu")
    ckpt_path = REPO_ROOT / "checkpoints" / "connectome_ppo_v3.pt"

    for fam in ["Model_A_Biological", "Model_B_Shuffled_Null", "Model_C_Random_Sparse"]:
        model, in_channels = load_model_family(fam, ckpt_path, device)
        assert model.num_nodes == 30000
        assert model.num_sensory == 9201
        assert model.num_motor == 1314
        assert not model.rnn.W_rec.requires_grad


def test_landmark_consistency():
    """Ensure landmark clearance coordinates match gold-standard level geometry."""
    assert WORLD_1_1_LANDMARKS["goomba1"] == 300
    assert WORLD_1_1_LANDMARKS["pipe1"] == 481
    assert WORLD_1_1_LANDMARKS["pipe2"] == 641
    assert WORLD_1_1_LANDMARKS["pipe3"] == 769
    assert WORLD_1_1_LANDMARKS["pipe4"] == 945
    assert WORLD_1_1_LANDMARKS["pit1"] == 1137


def run_ablation_metrics_tests() -> bool:
    test_sparsity_ratio()
    test_welch_t_test_and_cohens_d()
    test_telemetry_canvas_rendering()
    test_model_families_loading()
    test_landmark_consistency()
    return True


if __name__ == "__main__":
    run_ablation_metrics_tests()
    print("All ablation metrics tests passed!")

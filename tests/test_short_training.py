"""Test E: Short Reproducible Training Smoke Test (Imitation Learning).

Requirements:
- Collect a small demonstration trajectory from the Mario environment.
- Split into train (80%) and held-out validation (20%) sets.
- Train the ConnectomeMarioPolicy via Behavioral Cloning for 10 epochs.
- Record initial vs final train loss and validation loss.
- Verify measurable improvement on held-out validation loss.
- Save training checkpoint to check that model serialization works.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from src.environment import Mario2DEnv
from src.connectome_model import ConnectomeMarioPolicy


def run_short_training_test(num_samples: int = 50, epochs: int = 10) -> bool:
    print("=" * 60)
    print("TEST E: SHORT TRAINING SMOKE TEST (BEHAVIORAL CLONING)")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cpu")

    processed_dir = REPO_ROOT / "data" / "processed"
    sub_graph_path = processed_dir / "subgraph_graph.npz"
    node_map_path = processed_dir / "subgraph_node_mapping.csv"
    node_meta_path = processed_dir / "node_metadata.csv"

    print("1. Collecting demonstration transitions from Mario environment...")
    env = Mario2DEnv()
    obs, info = env.reset()

    observations = []
    actions = []

    # Heuristic expert: alternate right (1) and right+jump (4) to make forward progress
    for t in range(num_samples):
        expert_action = 4 if (t % 5 == 0) else 1  # jump periodically
        observations.append(obs.clone())
        actions.append(expert_action)
        obs, reward, terminated, truncated, info = env.step(expert_action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    obs_tensor = torch.stack(observations)  # (N, 1, 84, 84)
    act_tensor = torch.tensor(actions, dtype=torch.long)  # (N,)

    # Train / validation split (80 / 20)
    n_train = int(0.8 * num_samples)
    train_obs, val_obs = obs_tensor[:n_train], obs_tensor[n_train:]
    train_act, val_act = act_tensor[:n_train], act_tensor[n_train:]
    print(f"   Collected {num_samples} frames: {n_train} train, {len(val_obs)} validation.")

    print("2. Initializing ConnectomeMarioPolicy...")
    model = ConnectomeMarioPolicy(
        subgraph_npz_path=sub_graph_path,
        node_mapping_csv_path=node_map_path,
        node_metadata_csv_path=node_meta_path,
    ).to(device)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-5,
    )
    criterion = nn.CrossEntropyLoss()

    # Initial validation evaluation
    model.eval()
    with torch.no_grad():
        h_val = model.init_hidden(len(val_obs), device)
        val_logits, _, _ = model(val_obs, h_val, num_ticks=1)
        initial_val_loss = criterion(val_logits, val_act).item()

    print(f"   Initial validation loss: {initial_val_loss:.4f}")

    print(f"3. Training for {epochs} epochs...")
    train_losses = []
    val_losses = []

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()

        h_train = model.init_hidden(len(train_obs), device)
        logits, _, _ = model(train_obs, h_train, num_ticks=1)
        loss = criterion(logits, train_act)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Evaluate validation
        model.eval()
        with torch.no_grad():
            h_val = model.init_hidden(len(val_obs), device)
            val_logits, _, _ = model(val_obs, h_val, num_ticks=1)
            val_loss = criterion(val_logits, val_act).item()
            val_losses.append(val_loss)

        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"   Epoch {ep+1:2d}/{epochs:2d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    total_time = time.time() - t0
    final_val_loss = val_losses[-1]
    improvement = initial_val_loss - final_val_loss

    print("\n--- Training Results Summary ---")
    print(f"Total training time: {total_time:.2f}s ({(total_time/epochs):.2f}s/epoch)")
    print(f"Initial Val Loss:    {initial_val_loss:.4f}")
    print(f"Final Val Loss:      {final_val_loss:.4f}")
    print(f"Absolute Reduction:  {improvement:.4f} ({(improvement/initial_val_loss)*100:.1f}%)")

    # 4. Save checkpoint & verify serializability
    checkpoints_dir = REPO_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoints_dir / "connectome_mario_smoke_ckpt.pt"
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "train_loss": train_losses[-1],
                "val_loss": final_val_loss,
                "epochs": epochs,
            },
            ckpt_path,
        )
        print(f"   [Saved] Checkpoint to {ckpt_path.name} ({ckpt_path.stat().st_size / 1024 / 1024:.2f} MB)")
        assert ckpt_path.is_file(), "Checkpoint file was not created"

        # Assert verifiable learning (loss must strictly decrease)
        assert final_val_loss < initial_val_loss, (
            f"Validation loss did not improve: initial={initial_val_loss:.4f}, final={final_val_loss:.4f}"
        )
        assert torch.isfinite(torch.tensor(final_val_loss))
    finally:
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"   [Cleaned] Removed temporary smoke checkpoint {ckpt_path.name}")

    print("TEST E RESULT: PASS\n")
    return True


if __name__ == "__main__":
    success = run_short_training_test()
    sys.exit(0 if success else 1)


"""Test C: Model Step & Gradient Propagation Verification.

Requirements:
- Instantiate ConnectomeMarioPolicy using the real processed 30,000-neuron graph.
- Run one forward pass with synthetic observation batch [B, 1, 84, 84].
- Verify output shapes: logits [B, 5], value [B, 1], recurrent state [B, 30000].
- Compute a cross-entropy loss against dummy target action.
- Run backward pass.
- Verify finite loss and nonzero gradients in:
  - CNN encoder
  - Input adapter
  - Recurrent layer bias
  - Action decoder (actor and critic)
- Verify that the fixed connectome topology (W_rec buffer) receives NO gradients
  and is strictly non-trainable.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from src.connectome_model import ConnectomeMarioPolicy


def run_model_step_test() -> bool:
    print("=" * 60)
    print("TEST C: MODEL STEP & GRADIENT PROPAGATION")
    print("=" * 60)

    processed_dir = REPO_ROOT / "data" / "processed"
    sub_graph_path = processed_dir / "subgraph_graph.npz"
    node_map_path = processed_dir / "subgraph_node_mapping.csv"
    node_meta_path = processed_dir / "node_metadata.csv"

    print("1. Instantiating ConnectomeMarioPolicy with real 30,000-neuron subgraph...")
    model = ConnectomeMarioPolicy(
        subgraph_npz_path=sub_graph_path,
        node_mapping_csv_path=node_map_path,
        node_metadata_csv_path=node_meta_path,
        num_actions=5,
        cnn_dim=512,
    )
    print(f"   Total nodes: {model.num_nodes:,}")
    print(f"   Sensory (Visual Projection) neurons: {model.num_sensory:,}")
    print(f"   Motor (Descending) neurons: {model.num_motor:,}")

    # Check non-trainable fixed topology
    assert not model.rnn.W_rec.requires_grad, "W_rec must NOT require gradients!"
    print(f"   Fixed connectome buffer W_rec nnz: {model.rnn.W_rec._nnz():,}")

    batch_size = 2
    device = torch.device("cpu")
    model.to(device)

    print(f"2. Running forward pass with batch_size={batch_size}...")
    obs = torch.randn(batch_size, 1, 84, 84, device=device)
    h0 = model.init_hidden(batch_size, device=device)

    logits, value, h1 = model(obs, h0)
    print(f"   Logits shape: {logits.shape}, range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"   Value shape: {value.shape}, range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    print(f"   Hidden state shape: {h1.shape}, range: [{h1.min().item():.3f}, {h1.max().item():.3f}]")

    assert logits.shape == (batch_size, 5), f"Expected ({batch_size}, 5), got {logits.shape}"
    assert value.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {value.shape}"
    assert h1.shape == (batch_size, model.num_nodes), f"Expected ({batch_size}, {model.num_nodes}), got {h1.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf!"
    assert torch.isfinite(value).all(), "Value contains NaN or Inf!"
    assert torch.isfinite(h1).all(), "Hidden state contains NaN or Inf!"

    print("3. Computing loss and backward pass...")
    target_actions = torch.tensor([1, 4], device=device, dtype=torch.long)
    target_values = torch.tensor([[1.0], [2.0]], device=device, dtype=torch.float32)

    loss_actor = F.cross_entropy(logits, target_actions)
    loss_critic = F.mse_loss(value, target_values)
    total_loss = loss_actor + 0.5 * loss_critic

    print(f"   Loss: total={total_loss.item():.4f} (actor={loss_actor.item():.4f}, critic={loss_critic.item():.4f})")
    assert torch.isfinite(total_loss), "Loss is not finite!"

    total_loss.backward()

    print("4. Verifying gradient propagation across trainable modules...")
    # Encoder
    assert model.encoder.conv1.weight.grad is not None
    assert torch.isfinite(model.encoder.conv1.weight.grad).all()
    assert model.encoder.conv1.weight.grad.abs().sum().item() > 0.0
    print("   [Verified] CNN Encoder has nonzero gradients.")

    # Adapter
    assert model.adapter.weight.grad is not None
    assert torch.isfinite(model.adapter.weight.grad).all()
    assert model.adapter.weight.grad.abs().sum().item() > 0.0
    print("   [Verified] Input Adapter has nonzero gradients.")

    # Recurrent layer bias
    assert model.rnn.bias.grad is not None
    assert torch.isfinite(model.rnn.bias.grad).all()
    assert model.rnn.bias.grad.abs().sum().item() > 0.0
    print("   [Verified] Recurrent bias parameter has nonzero gradients.")

    # Actor & Critic
    assert model.actor.weight.grad is not None
    assert model.actor.weight.grad.abs().sum().item() > 0.0
    assert model.critic.weight.grad is not None
    assert model.critic.weight.grad.abs().sum().item() > 0.0
    print("   [Verified] Action & Value Decoders have nonzero gradients.")

    # Connectome buffer check
    assert model.rnn.W_rec.grad is None, "W_rec must NEVER have gradients!"
    print("   [Verified] MaleCNS W_rec topology is fixed and non-trainable.")

    print("TEST C RESULT: PASS\n")
    return True


if __name__ == "__main__":
    success = run_model_step_test()
    sys.exit(0 if success else 1)


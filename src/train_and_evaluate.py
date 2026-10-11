"""Controlled Comparative Training & Evaluation Pipeline for MaleCNS & Baselines.

Trains and evaluates 4 parameter-matched architectures:
1. Model A: ConnectomeMarioPolicy (Fixed 30,000-neuron MaleCNS biological topology, 6.44M params)
2. Model B: Rewired MaleCNS Policy (Degree-preserving directed shuffled topology, 6.44M params)
3. Model C: Random Sparse ER Policy (Uniform Erdős–Rényi random sparse topology, 6.44M params)
4. Model D: BaselineMarioPolicy (Unconstrained dense GRU-1024, 6.41M params)

Protocol:
- Identical training data (whole-episode partitioned scaled demonstration dataset).
- Identical optimizer, learning rate, loss function, epochs, and batch size.
- Held-out validation split for checkpoint selection.
- Closed-loop evaluation on held-out test seeds in Super Mario Bros 1-1.
- Metrics: Cross-entropy loss, action accuracy, maximum horizontal progress, total reward,
  survival time, obstacle clearance rate, action distribution, and action entropy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.demonstration_collector import (
    ACTION_NAMES,
    MARIO_5_ACTIONS,
    load_demonstration_dataset,
)
from src.environment import Mario2DEnv


MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "connectome": {
        "name": "MaleCNS Connectome Policy (30k Neurons)",
        "slug": "connectome_model",
        "type": "connectome",
        "graph_rel_path": "data/processed/subgraph_graph.npz",
        "topology": "Fixed MaleCNS 30,000-neuron CSR (6,001,050 edges)",
        "checkpoint": "checkpoints/connectome_model_best.pt",
    },
    "rewired": {
        "name": "Rewired MaleCNS Policy (Degree-Preserving)",
        "slug": "rewired_connectome",
        "type": "connectome",
        "graph_rel_path": "data/processed/shuffled_connectome_graph.npz",
        "topology": "Degree-Preserving Shuffled 30,000-neuron CSR (6,001,050 edges, 98.1% rewired)",
        "checkpoint": "checkpoints/rewired_connectome_best.pt",
    },
    "random_sparse": {
        "name": "Random Sparse Erdős-Rényi Policy (30k Neurons)",
        "slug": "random_sparse",
        "type": "connectome",
        "graph_rel_path": "data/processed/random_sparse_graph.npz",
        "topology": "Erdős-Rényi Random Sparse 30,000-neuron CSR (6,001,050 edges)",
        "checkpoint": "checkpoints/random_sparse_best.pt",
    },
    "baseline_gru": {
        "name": "Baseline GRU-1024 Policy (Dense)",
        "slug": "baseline_gru",
        "type": "gru",
        "graph_rel_path": None,
        "topology": "Unconstrained PyTorch GRUCell (dense 1024)",
        "checkpoint": "checkpoints/baseline_gru_best.pt",
    },
}


class TransitionDataset(Dataset):
    """PyTorch Dataset wrapping uint8 demonstration frames."""

    def __init__(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        self.obs = obs  # uint8 (N, 1, 84, 84)
        self.actions = actions  # int64 (N,)
        self.rewards = rewards  # float32 (N,)

    def __len__(self) -> int:
        return self.obs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize uint8 [0, 255] to float32 [0.0, 1.0]
        obs_float = self.obs[idx].float() / 255.0
        return obs_float, self.actions[idx], self.rewards[idx]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for obs, actions, _ in loader:
        obs, actions = obs.to(device), actions.to(device)
        optimizer.zero_grad()

        # Forward pass (h initialized to 0 inside model)
        logits, _, _ = model(obs, None)
        loss = criterion(logits, actions)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item() * obs.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == actions).sum().item()
        total += obs.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute validation loss and action accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for obs, actions, _ in loader:
        obs, actions = obs.to(device), actions.to(device)
        logits, _, _ = model(obs, None)
        loss = criterion(logits, actions)

        total_loss += loss.item() * obs.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == actions).sum().item()
        total += obs.size(0)

    val_loss = total_loss / total
    val_acc = correct / total
    return val_loss, val_acc


@torch.no_grad()
def evaluate_closed_loop(
    env: Mario2DEnv,
    model: nn.Module,
    seeds: List[int],
    device: torch.device,
    max_steps: int = 2500,
) -> Dict[str, Any]:
    """Evaluate closed-loop gameplay in Super Mario Bros 1-1."""
    model.eval()
    ep_results = []
    action_totals = {name: 0 for name in ACTION_NAMES}

    for s in seeds:
        obs_tensor, info = env.reset(seed=s)
        h: Optional[torch.Tensor] = None
        total_reward = 0.0
        max_x = info.get("x_pos", 40)
        steps = 0
        flag_get = False
        died_on_first_obstacle = False

        for step in range(max_steps):
            inp = obs_tensor.unsqueeze(0).to(device)  # (1, 1, 84, 84)
            logits, _, h = model(inp, h)
            action = torch.argmax(logits, dim=-1).item()
            action_totals[ACTION_NAMES[action]] += 1

            next_obs, r, term, trunc, info = env.step(action)
            total_reward += r
            current_x = info.get("x_pos", 40)
            max_x = max(max_x, current_x)
            flag_get = bool(info.get("flag_get", False))
            steps = step + 1

            obs_tensor = next_obs

            # Check if died on first Goomba (step ~159, x=296)
            if (term or trunc) and (max_x <= 296):
                died_on_first_obstacle = True

            if term or trunc:
                break

        ep_results.append({
            "seed": s,
            "steps": steps,
            "final_x": max_x,
            "total_reward": round(total_reward, 2),
            "flag_get": flag_get,
            "cleared_first_obstacle": max_x > 296,
        })

    xs = [e["final_x"] for e in ep_results]
    rewards = [e["total_reward"] for e in ep_results]
    steps_list = [e["steps"] for e in ep_results]
    cleared_first = [e["cleared_first_obstacle"] for e in ep_results]
    cleared_level = [e["flag_get"] for e in ep_results]

    total_actions = sum(action_totals.values())
    action_props = (
        {k: round(v / total_actions, 4) for k, v in action_totals.items()}
        if total_actions > 0
        else {}
    )

    # Compute Shannon entropy over action distribution: -sum(p * log(p))
    entropy = 0.0
    for p in action_props.values():
        if p > 0:
            entropy -= p * math.log(p)

    return {
        "mean_final_x": round(float(np.mean(xs)), 2),
        "std_final_x": round(float(np.std(xs)), 2),
        "max_final_x": int(np.max(xs)),
        "min_final_x": int(np.min(xs)),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_steps": round(float(np.mean(steps_list)), 2),
        "first_obstacle_clear_rate": round(float(np.mean(cleared_first)), 2),
        "level_clear_rate": round(float(np.mean(cleared_level)), 2),
        "action_entropy": round(entropy, 4),
        "action_proportions": action_props,
        "action_counts": action_totals,
        "episodes": ep_results,
    }


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    checkpoint_path: Path,
    class_weights: Optional[torch.Tensor] = None,
    log_dir: Optional[Path] = None,
    model_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a single model and track best validation checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr}")
    print(f"{'=' * 60}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize CSV logger if log_dir provided
    csv_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        slug = model_slug or ("connectome_model" if "Connectome" in model_name else "baseline_gru")
        csv_file = log_dir / f"{slug}_training_metrics.csv"
        with open(csv_file, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,epoch_duration_sec,samples_per_sec,is_best,timestamp\n")

    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t_start = time.time()
    n_train_samples = len(train_loader.dataset)

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = evaluate_loss(model, val_loader, criterion, device)
        ep_duration = time.time() - ep_t0
        samples_per_sec = round(n_train_samples / ep_duration, 2) if ep_duration > 0 else 0.0

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))

        is_best = v_loss < best_val_loss
        if is_best:
            best_val_loss = v_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                    "model_name": model_name,
                    "history": history,
                },
                checkpoint_path,
            )

        if csv_file is not None:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(csv_file, "a") as f:
                f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{v_loss:.4f},{v_acc:.4f},{lr},{ep_duration:.2f},{samples_per_sec},{str(is_best).lower()},{ts}\n")

        best_mark = " [*BEST*]" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{epochs} ({ep_duration:4.1f}s) | "
            f"Train Loss: {tr_loss:.4f} (Acc: {tr_acc * 100:5.1f}%) | "
            f"Val Loss: {v_loss:.4f} (Acc: {v_acc * 100:5.1f}%){best_mark}"
        )

    total_training_time = round(time.time() - t_start, 2)
    print(f"Training completed in {total_training_time}s. Best Val Loss: {best_val_loss:.4f}")

    # Reload best weights for evaluation
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    return {
        "model_name": model_name,
        "trainable_parameters": trainable_params,
        "total_training_time_seconds": total_training_time,
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": round(best_ckpt["val_loss"], 4),
        "best_val_acc": round(best_ckpt["val_acc"], 4),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }


def instantiate_model(model_key: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Factory function to instantiate any of the 4 models."""
    cfg = MODELS_CONFIG[model_key]
    processed_dir = Path("data/processed")

    if cfg["type"] == "connectome":
        model = ConnectomeMarioPolicy(
            subgraph_npz_path=processed_dir.parent.parent / cfg["graph_rel_path"] if not Path(cfg["graph_rel_path"]).exists() else Path(cfg["graph_rel_path"]),
            node_mapping_csv_path=processed_dir / "subgraph_node_mapping.csv",
            node_metadata_csv_path=processed_dir / "node_metadata.csv",
        ).to(device)
    elif cfg["type"] == "gru":
        model = BaselineMarioPolicy(cnn_dim=512, hidden_dim=1024).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg['type']}")

    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Comparative Training & Evaluation Suite")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/demonstrations/scaled_demonstrations.pt",
        help="Path to scaled demonstrations",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["connectome", "rewired", "random_sparse", "baseline_gru"],
        help="Models to train/evaluate: connectome, rewired, random_sparse, baseline_gru, or all",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--output-report",
        type=str,
        default="baseline_ablation_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save per-epoch CSV metrics, logs, and summaries",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate existing checkpoints on test split and closed-loop environment",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if model checkpoint already exists on disk",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss",
    )
    args = parser.parse_args()

    # Expand 'all'
    selected_keys = args.models
    if "all" in selected_keys:
        selected_keys = ["connectome", "rewired", "random_sparse", "baseline_gru"]
    for k in selected_keys:
        if k not in MODELS_CONFIG:
            raise ValueError(f"Unknown model key '{k}'. Available: {list(MODELS_CONFIG.keys())}")

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Target execution device: {device}")

    # 1. Load scaled demonstration dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = Path("data/demonstrations/pilot_demonstrations.pt")
        print(f"Scaled dataset not found. Falling back to pilot dataset: {dataset_path}")

    ds = load_demonstration_dataset(dataset_path)
    meta = ds["metadata"]
    print(f"Loaded dataset from {dataset_path} ({meta['num_transitions']:,} transitions, {meta['num_episodes']} episodes)")

    # 2. Extract split masks
    train_seeds = set(meta["splits"]["train_seeds"])
    val_seeds = set(meta["splits"]["val_seeds"])
    test_seeds = set(meta["splits"]["test_seeds"])
    print(f"Splits -> Train: {sorted(train_seeds)} | Val: {sorted(val_seeds)} | Test: {sorted(test_seeds)}")

    seeds_tensor = ds["seeds"]
    train_mask = torch.tensor([s.item() in train_seeds for s in seeds_tensor], dtype=torch.bool)
    val_mask = torch.tensor([s.item() in val_seeds for s in seeds_tensor], dtype=torch.bool)
    test_mask = torch.tensor([s.item() in test_seeds for s in seeds_tensor], dtype=torch.bool)

    train_ds = TransitionDataset(ds["obs"][train_mask], ds["actions"][train_mask], ds["rewards"][train_mask])
    val_ds = TransitionDataset(ds["obs"][val_mask], ds["actions"][val_mask], ds["rewards"][val_mask])
    test_ds = TransitionDataset(ds["obs"][test_mask], ds["actions"][test_mask], ds["rewards"][test_mask])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Partition sizes -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # Class weighting
    weights = None
    if args.use_class_weights:
        train_acts = ds["actions"][train_mask]
        counts = torch.bincount(train_acts, minlength=5).float()
        weights = torch.zeros(5, dtype=torch.float32, device=device)
        for c in range(5):
            if counts[c] > 0:
                weights[c] = float(len(train_acts)) / (5.0 * counts[c])
            else:
                weights[c] = 0.0
        active = weights > 0
        weights[active] = weights[active] / weights[active].mean()
        print(f"Normalized class weights: {weights.tolist()}")
    else:
        print("Using standard unweighted CrossEntropyLoss")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. Train or Load Models
    trained_models: Dict[str, nn.Module] = {}
    training_stats: Dict[str, Dict[str, Any]] = {}

    for key in selected_keys:
        model, cfg = instantiate_model(key, device)
        ckpt_path = Path(cfg["checkpoint"])
        should_train = True

        if args.eval_only:
            should_train = False
        elif args.skip_existing and ckpt_path.exists():
            print(f"\nFound existing checkpoint for {cfg['name']} (--skip-existing). Skipping training.")
            should_train = False

        if should_train:
            stats = train_model(
                model_name=cfg["name"],
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                checkpoint_path=ckpt_path,
                class_weights=weights,
                log_dir=log_dir,
                model_slug=cfg["slug"],
            )
            training_stats[key] = stats
        else:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Requested eval-only / skip-existing for {key}, but checkpoint missing: {ckpt_path}")
            print(f"Loading checkpoint for {cfg['name']} from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            training_stats[key] = {
                "model_name": cfg["name"],
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_training_time_seconds": ckpt.get("total_training_time_seconds", 0.0),
                "best_epoch": ckpt.get("epoch", 5),
                "best_val_loss": round(ckpt.get("val_loss", 0.0), 4),
                "best_val_acc": round(ckpt.get("val_acc", 0.0), 4),
                "history": ckpt.get("history", {}),
                "checkpoint_path": str(ckpt_path),
            }

        trained_models[key] = model

    # 4. Evaluate on Held-Out Test Set (Loss & Accuracy)
    criterion_eval = nn.CrossEntropyLoss()
    test_eval_results: Dict[str, Dict[str, float]] = {}
    for key, model in trained_models.items():
        t_loss, t_acc = evaluate_loss(model, test_loader, criterion_eval, device)
        test_eval_results[key] = {
            "test_loss": round(t_loss, 4),
            "test_acc": round(t_acc, 4),
        }
        print(f"Test Split Eval -> {MODELS_CONFIG[key]['name']}: Loss = {t_loss:.4f}, Accuracy = {t_acc * 100:.2f}%")

    # 5. Closed-Loop Gameplay Evaluation on Held-Out Test Seeds
    env = Mario2DEnv()
    eval_seeds = sorted(test_seeds)
    print(f"\nEvaluating closed-loop gameplay on {len(eval_seeds)} test seeds: {eval_seeds}...")
    closed_loop_results: Dict[str, Dict[str, Any]] = {}

    for key, model in trained_models.items():
        print(f"Benchmarking closed-loop: {MODELS_CONFIG[key]['name']}...")
        cl_res = evaluate_closed_loop(env, model, eval_seeds, device)
        closed_loop_results[key] = cl_res
        print(f"  -> Mean Final X: {cl_res['mean_final_x']} | Mean Reward: {cl_res['mean_reward']} | Obstacle 1 Rate: {cl_res['first_obstacle_clear_rate']*100:.1f}%")

    env.close()

    # 6. Assemble Comprehensive Report
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "Phase 5: Baselines & Ablation Study Block",
        "training_configuration": {
            "dataset": str(dataset_path),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": str(device),
            "class_weights": weights.tolist() if weights is not None else None,
        },
        "models": {},
    }

    for key in selected_keys:
        cfg = MODELS_CONFIG[key]
        report["models"][key] = {
            "name": cfg["name"],
            "slug": cfg["slug"],
            "topology": cfg["topology"],
            "trainable_parameters": training_stats[key]["trainable_parameters"],
            "best_epoch": training_stats[key]["best_epoch"],
            "best_val_loss": training_stats[key]["best_val_loss"],
            "best_val_acc": training_stats[key]["best_val_acc"],
            "test_loss": test_eval_results[key]["test_loss"],
            "test_acc": test_eval_results[key]["test_acc"],
            "training_time_seconds": training_stats[key]["total_training_time_seconds"],
            "checkpoint_path": training_stats[key]["checkpoint_path"],
            "closed_loop_gameplay": closed_loop_results[key],
        }

    out_p = Path(args.output_report)
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)

    summary_log_p = log_dir / "baseline_ablation_summary.json"
    with open(summary_log_p, "w") as f:
        json.dump(report, f, indent=2)

    # 7. Print Master Comparison Table
    print("\n" + "=" * 105)
    print("PHASE 5: 4-MODEL BASELINES & ABLATION STUDY COMPARATIVE RESULTS")
    print("=" * 105)
    header = f"{'Metric':<28} | " + " | ".join(f"{MODELS_CONFIG[k]['slug']:<15}" for k in selected_keys)
    print(header)
    print("-" * 105)

    p_row = f"{'Trainable Parameters':<28} | " + " | ".join(f"{training_stats[k]['trainable_parameters']:<15,d}" for k in selected_keys)
    print(p_row)

    vl_row = f"{'Best Val Loss':<28} | " + " | ".join(f"{training_stats[k]['best_val_loss']:<15.4f}" for k in selected_keys)
    print(vl_row)

    va_row = f"{'Best Val Accuracy':<28} | " + " | ".join(f"{training_stats[k]['best_val_acc']*100:<14.1f}%" for k in selected_keys)
    print(va_row)

    tl_row = f"{'Test Split Loss':<28} | " + " | ".join(f"{test_eval_results[k]['test_loss']:<15.4f}" for k in selected_keys)
    print(tl_row)

    ta_row = f"{'Test Split Accuracy':<28} | " + " | ".join(f"{test_eval_results[k]['test_acc']*100:<14.1f}%" for k in selected_keys)
    print(ta_row)

    print("-" * 105)
    print("Closed-Loop Gameplay on Held-Out Test Seeds (Seeds 13, 14, 15):")

    mx_row = f"{'Mean Final X':<28} | " + " | ".join(f"{closed_loop_results[k]['mean_final_x']:<15.1f}" for k in selected_keys)
    print(mx_row)

    mr_row = f"{'Mean Reward':<28} | " + " | ".join(f"{closed_loop_results[k]['mean_reward']:<15.1f}" for k in selected_keys)
    print(mr_row)

    ms_row = f"{'Mean Steps':<28} | " + " | ".join(f"{closed_loop_results[k]['mean_steps']:<15.1f}" for k in selected_keys)
    print(ms_row)

    ob_row = f"{'Obstacle 1 Clear Rate':<28} | " + " | ".join(f"{closed_loop_results[k]['first_obstacle_clear_rate']*100:<14.1f}%" for k in selected_keys)
    print(ob_row)

    ae_row = f"{'Action Entropy':<28} | " + " | ".join(f"{closed_loop_results[k]['action_entropy']:<15.4f}" for k in selected_keys)
    print(ae_row)

    print("=" * 105)
    print(f"Report saved to: {out_p}")
    print(f"Summary saved to: {summary_log_p}")


if __name__ == "__main__":
    main()

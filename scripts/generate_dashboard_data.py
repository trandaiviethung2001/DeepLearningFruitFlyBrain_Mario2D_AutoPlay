"""Extracts step-by-step MaleCNS neuron body activations and metadata for the interactive dashboard.

Runs an episode with ConnectomeMarioPolicy, records activations across 30k nodes,
merges with FlyWire / MaleCNS body annotations, and saves data/telemetry_dashboard_data.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.connectome_model import ConnectomeMarioPolicy
from src.environment import Mario2DEnv
from src.visualization import ACTION_NAMES, compute_sparsity_ratio


def main():
    device = torch.device("cpu")
    processed_dir = REPO_ROOT / "data" / "processed"
    ckpt_path = REPO_ROOT / "checkpoints" / "connectome_ppo_v3.pt"

    print("1. Loading node metadata and subgraph mapping...")
    sub_map = pd.read_csv(processed_dir / "subgraph_node_mapping.csv")
    node_meta = pd.read_csv(processed_dir / "node_metadata.csv", low_memory=False)

    merged = sub_map.merge(node_meta, on="body_id", how="left")
    body_ids = merged["body_id"].values
    types = merged["type"].fillna("Unknown").values
    superclasses = merged["superclass"].fillna("cb_intrinsic").values
    soma_sides = merged["somaSide"].fillna("Unknown").values
    neuromeres = merged["somaNeuromere"].fillna("Brain").values

    print("2. Loading ConnectomeMarioPolicy...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("frame_stack", 4)
    if "model_state_dict" in ckpt and "encoder.conv1.weight" in ckpt["model_state_dict"]:
        in_channels = ckpt["model_state_dict"]["encoder.conv1.weight"].shape[1]

    policy = ConnectomeMarioPolicy(
        subgraph_npz_path=processed_dir / "subgraph_graph.npz",
        node_mapping_csv_path=processed_dir / "subgraph_node_mapping.csv",
        node_metadata_csv_path=processed_dir / "node_metadata.csv",
        in_channels=in_channels,
        cnn_dim=512,
    )
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()

    # Precompute descending motor weights (which DNs connect strongly to which action)
    actor_weight = policy.actor.weight.detach().cpu().numpy()  # (5, 1314)
    dn_indices = policy.motor_indices.cpu().numpy()  # (1314,)

    print("3. Executing Mario episode and capturing milestone decisions...")
    env = Mario2DEnv(
        frame_stack=in_channels,
        frame_skip=4,
        auto_unlatch=True,
        render_mode="rgb_array",
        max_episode_frames=5000,
        stall_frames=600,
    )

    obs, info = env.reset(seed=1000)
    h = None

    steps_data: List[Dict[str, Any]] = []
    step = 0

    # Key milestones to sample densely, plus regular stride
    milestone_targets = [0, 20, 40, 60, 80, 100, 120, 140, 170, 200, 240, 280, 320, 360, 400, 440, 480, 487]

    while step <= 488:
        if isinstance(obs, torch.Tensor):
            obs_t = obs.unsqueeze(0).float().to(device)
        else:
            obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits, val, h_next, tele = policy(obs_t, h, num_ticks=2, return_telemetry=True)
            core_act = tele["h_core"].cpu().numpy().squeeze(0)  # (30000,)
            sensory_drive = tele["sensory_drive"].cpu().numpy().squeeze(0)  # (9201,)
            motor_act = tele["motor_activity"].cpu().numpy().squeeze(0)  # (1314,)

            probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
            action = int(np.argmax(probs))

        h = h_next
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        step += 1

        x_pos = int(next_info.get("x_pos", 0))
        y_pos = int(next_info.get("y_pos", 0))
        flag = bool(next_info.get("flag_get", False))

        # Sample frame if it matches a milestone or step % 25 == 0 or flag
        if (step in milestone_targets) or (step % 25 == 0) or flag or (step == 1):
            # Extract top 25 active neurons overall
            abs_core = np.abs(core_act)
            top_overall_idx = np.argsort(abs_core)[-25:][::-1]

            # Extract top 15 active descending neurons (motor)
            abs_motor = np.abs(motor_act)
            top_dn_sub_idx = np.argsort(abs_motor)[-15:][::-1]
            top_dn_global_idx = dn_indices[top_dn_sub_idx]

            # Extract top 15 active visual projection neurons
            abs_vp = np.abs(sensory_drive)
            vp_indices = policy.sensory_indices.cpu().numpy()
            top_vp_sub_idx = np.argsort(abs_vp)[-15:][::-1]
            top_vp_global_idx = vp_indices[top_vp_sub_idx]

            def serialize_neurons(indices, act_array):
                items = []
                for idx in indices:
                    b_id = int(body_ids[idx])
                    c_type = str(types[idx])
                    s_class = str(superclasses[idx])
                    side = str(soma_sides[idx])
                    neuromere = str(neuromeres[idx])
                    act_val = float(core_act[idx])
                    items.append({
                        "subgraph_index": int(idx),
                        "body_id": b_id,
                        "type": c_type,
                        "superclass": s_class,
                        "side": side,
                        "neuromere": neuromere,
                        "activation": round(act_val, 4),
                        "abs_activation": round(abs(act_val), 4),
                    })
                return items

            # Superclass mean activity
            vp_mean_act = float(np.mean(abs_vp))
            dn_mean_act = float(np.mean(abs_motor))
            cb_mask = (superclasses == "cb_intrinsic")
            cb_mean_act = float(np.mean(np.abs(core_act[cb_mask])))
            sparsity = compute_sparsity_ratio(core_act)

            # Determine human-friendly milestone description
            desc = "Traversing Ground"
            if x_pos < 100:
                desc = "Game Spawn & Visual Intake"
            elif 250 <= x_pos <= 320:
                desc = "Approaching Goomba 1 (Jump Decided)"
            elif 350 <= x_pos <= 420:
                desc = "Clearing Pipe 1 (32px Height)"
            elif 500 <= x_pos <= 600:
                desc = "Clearing Pipe 2 (48px Height)"
            elif 700 <= x_pos <= 780:
                desc = "Clearing Pipe 3 (64px Height)"
            elif 900 <= x_pos <= 1000:
                desc = "Clearing Pipe 4 (High Pipe)"
            elif 1300 <= x_pos <= 1450:
                desc = "Leaping over Pit 1 (Abyss Crossing)"
            elif x_pos >= 3100:
                desc = "🏆 Flagpole Capture & Victory Leap!"

            frame_entry = {
                "step": step,
                "x_pos": x_pos,
                "y_pos": y_pos,
                "milestone": desc,
                "action_idx": action,
                "action_name": ACTION_NAMES[action],
                "action_probs": [round(float(p), 4) for p in probs],
                "value": round(float(val.item()), 3),
                "sparsity_l1_l2": round(sparsity, 2),
                "summary": {
                    "vp_mean_abs": round(vp_mean_act, 4),
                    "dn_mean_abs": round(dn_mean_act, 4),
                    "cb_mean_abs": round(cb_mean_act, 4),
                },
                "top_active_overall": serialize_neurons(top_overall_idx, core_act),
                "top_active_dn": serialize_neurons(top_dn_global_idx, core_act),
                "top_active_vp": serialize_neurons(top_vp_global_idx, core_act),
            }
            steps_data.append(frame_entry)
            print(f"  Captured Step {step:3d} (X={x_pos:4d}): {desc} | Action: {ACTION_NAMES[action]}")

        obs = next_obs
        if flag or terminated or truncated:
            break

    env.close()

    out_json = REPO_ROOT / "data" / "telemetry_dashboard_data.json"
    with open(out_json, "w") as f:
        json.dump({
            "total_nodes": 30000,
            "total_edges": 6001050,
            "sensory_nodes": 9201,
            "motor_nodes": 1314,
            "intrinsic_nodes": 19485,
            "total_steps": step,
            "final_x": x_pos,
            "frames": steps_data,
        }, f, indent=2)

    print(f"\nSaved dashboard telemetry data ({len(steps_data)} frames) to: {out_json}")


if __name__ == "__main__":
    main()


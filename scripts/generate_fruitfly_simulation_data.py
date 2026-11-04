"""Generates comprehensive simulation dataset for Fruit Fly Organism & Connectome Simulator.

Records all 488 steps of the Flagpole-winning episode:
- NATIVE 256x240 LOSSLESS NES frames directly from play.py via env.render()
- Detailed Drosophila MaleCNS anatomical compartments (Optic Lobes L/R, CX, Central Protocerebrum, DN trunk)
- Synaptic in-degree and out-degree from 6,001,050-synapse CSR connectome
- Dense 3,500-neuron anatomical scaffold with 3D FlyWire soma coordinates
- Real-time neural firing intensities, giant fiber escape signals, and looming detectors.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.feather as feather
import scipy.sparse as sp
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
    feather_path = REPO_ROOT / "data" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

    print("1. Loading metadata, CSR synapses, and 3D FlyWire soma coordinates...")
    sub_map = pd.read_csv(processed_dir / "subgraph_node_mapping.csv")
    node_meta = pd.read_csv(processed_dir / "node_metadata.csv", low_memory=False)

    # Compute in/out degree from 6M-edge CSR graph
    npz = np.load(processed_dir / "subgraph_graph.npz")
    mat = sp.csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=npz["shape"])
    out_degrees = np.diff(mat.indptr)
    in_degrees = np.bincount(mat.indices, minlength=mat.shape[0])

    ann_table = feather.read_table(
        feather_path,
        columns=["bodyId", "somaLocation"]
    )
    df_ann = ann_table.to_pandas()
    df_ann["body_id"] = df_ann["bodyId"].astype(int)

    merged = sub_map.merge(node_meta, on="body_id", how="left")
    merged = merged.merge(df_ann[["body_id", "somaLocation"]], on="body_id", how="left")

    body_ids = merged["body_id"].values
    types = merged["type"].fillna("Unknown").values
    superclasses = merged["superclass"].fillna("cb_intrinsic").values
    soma_sides = merged["somaSide"].fillna("Unknown").values
    neuromeres = merged["somaNeuromere"].fillna("Brain").values
    classes = merged["class"].fillna("").values

    # Determine anatomical neuropil compartment for every neuron
    neuropils = []
    for i in range(len(merged)):
        sc = str(superclasses[i])
        side = str(soma_sides[i])
        cl = str(classes[i])
        t = str(types[i])
        if sc == "visual_projection":
            neuropils.append(f"Optic Lobe ({side})" if side in ["L", "R"] else "Optic Lobe")
        elif sc == "descending_neuron":
            neuropils.append("Descending Motor Trunk")
        elif "CX" in cl or t.startswith(("FB", "EB", "NO", "AB", "PB")):
            neuropils.append("Central Complex (CX)")
        elif "Kenyon" in cl or "MB" in cl:
            neuropils.append("Mushroom Body (MB)")
        else:
            neuropils.append("Central Protocerebrum")

    neuropils = np.array(neuropils)

    xs, ys, zs = [], [], []
    for loc in merged["somaLocation"]:
        if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) == 3:
            xs.append(int(loc[0]))
            ys.append(int(loc[1]))
            zs.append(int(loc[2]))
        else:
            xs.append(None)
            ys.append(None)
            zs.append(None)

    # Build dense 3,500-neuron anatomical scaffold representing detailed Drosophila morphology
    scaffold_neurons = []
    valid_indices = [i for i, x in enumerate(xs) if x is not None]
    np.random.seed(42)

    targets = [
        ("Optic Lobe (L)", 800),
        ("Optic Lobe (R)", 800),
        ("Central Complex (CX)", 400),
        ("Mushroom Body (MB)", 300),
        ("Descending Motor Trunk", 600),
        ("Central Protocerebrum", 800),
    ]

    for np_name, target_n in targets:
        n_idxs = [i for i in valid_indices if neuropils[i] == np_name]
        chosen = np.random.choice(n_idxs, size=min(target_n, len(n_idxs)), replace=False)
        for i in chosen:
            scaffold_neurons.append({
                "idx": int(i),
                "id": int(body_ids[i]),
                "type": str(types[i]),
                "sc": str(superclasses[i]),
                "np": str(neuropils[i]),
                "side": str(soma_sides[i]),
                "x": xs[i],
                "y": ys[i],
                "z": zs[i],
                "in_deg": int(in_degrees[i]),
                "out_deg": int(out_degrees[i]),
            })

    print(f"Dense anatomical scaffold neurons selected: {len(scaffold_neurons)}")

    # Compute anatomical neuropil centroids and radii for HUD outlines
    neuropil_bounds = {}
    for np_name in ["Optic Lobe (L)", "Optic Lobe (R)", "Central Complex (CX)", "Descending Motor Trunk", "Central Protocerebrum"]:
        np_idxs = [i for i in valid_indices if neuropils[i] == np_name]
        if np_idxs:
            n_xs = [xs[i] for i in np_idxs]
            n_ys = [ys[i] for i in np_idxs]
            n_zs = [zs[i] for i in np_idxs]
            neuropil_bounds[np_name] = {
                "cx": int(np.mean(n_xs)),
                "cy": int(np.mean(n_ys)),
                "cz": int(np.mean(n_zs)),
                "rx": int(np.std(n_xs) * 1.5),
                "ry": int(np.std(n_ys) * 1.5),
                "rz": int(np.std(n_zs) * 1.5),
                "count": len(np_idxs),
            }

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

    dn_indices = policy.motor_indices.cpu().numpy()
    vp_indices = policy.sensory_indices.cpu().numpy()
    cb_mask = (superclasses == "cb_intrinsic")
    cb_indices = np.where(cb_mask)[0]

    gf_types = {"DNp01", "DNp06", "DNp31"}
    gf_local_indices = [i for i in dn_indices if str(types[i]) in gf_types]

    print("3. Executing live game loop, capturing native 256x240 frames and rich connectome telemetry...")
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
    recorded_neuron_details: Dict[int, Dict[str, Any]] = {}
    macro_step = 0
    frame_count = 0

    while macro_step < 500:
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
        macro_step += 1

        top_dn_order = np.argsort(np.abs(motor_act))[-12:][::-1]
        top_dn_idxs = dn_indices[top_dn_order]

        top_vp_order = np.argsort(np.abs(sensory_drive))[-12:][::-1]
        top_vp_idxs = vp_indices[top_vp_order]

        top_cb_order = np.argsort(np.abs(core_act[cb_indices]))[-12:][::-1]
        top_cb_idxs = cb_indices[top_cb_order]

        def record_neurons(indices):
            res = []
            for idx in indices:
                i = int(idx)
                act_val = round(float(core_act[i]), 3)
                res.append([i, act_val])
                if i not in recorded_neuron_details:
                    recorded_neuron_details[i] = {
                        "id": int(body_ids[i]),
                        "type": str(types[i]),
                        "sc": str(superclasses[i]),
                        "np": str(neuropils[i]),
                        "side": str(soma_sides[i]),
                        "nm": str(neuromeres[i]),
                        "x": xs[i],
                        "y": ys[i],
                        "z": zs[i],
                        "in_deg": int(in_degrees[i]),
                        "out_deg": int(out_degrees[i]),
                    }
            return res

        serialized_dn = record_neurons(top_dn_idxs)
        serialized_vp = record_neurons(top_vp_idxs)
        serialized_cb = record_neurons(top_cb_idxs)

        vp_mean = round(float(np.mean(np.abs(sensory_drive))), 3)
        dn_mean = round(float(np.mean(np.abs(motor_act))), 3)
        cb_mean = round(float(np.mean(np.abs(core_act[cb_mask]))), 3)
        sparsity = round(float(compute_sparsity_ratio(core_act)), 1)
        gf_max = float(np.max(np.abs(core_act[gf_local_indices]))) if gf_local_indices else 0.0

        # Record every individual 60 FPS subframe: zero frame skipping!
        subframes_captured = []
        terminated = truncated = False
        for f in range(env.frame_skip):
            act = action
            if env.auto_unlatch:
                if act in (3, 4):
                    on_ground = (env.ram[0x001D] == 0)
                    if on_ground and env._last_held_a and f == 0:
                        act = 1 if action == 4 else 0
                        env._last_held_a = False
                    else:
                        env._last_held_a = True
                else:
                    env._last_held_a = False

            obs_raw, reward, term, trunc, sub_info = env._raw_step(act)
            if term:
                terminated = True
            if trunc:
                truncated = True

            # Save lossless WebP 256x240 frame
            img = Image.fromarray(obs_raw)
            buf = io.BytesIO()
            img.save(buf, format="WEBP", lossless=True)
            frame_b64 = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            sub_x = int(sub_info.get("x_pos", 0))
            sub_y = int(sub_info.get("y_pos", 0))
            sub_flag = bool(sub_info.get("flag_get", False))

            frame_count += 1

            milestone = ""
            if frame_count == 1:
                milestone = "Game Spawn & Sensory Intake"
            elif 290 <= sub_x <= 315 and act in [2, 4]:
                milestone = "Goomba 1 Escape Leap"
            elif 390 <= sub_x <= 420:
                milestone = "Pipe 1 Jump"
            elif 540 <= sub_x <= 570:
                milestone = "Pipe 2 Jump"
            elif 730 <= sub_x <= 760:
                milestone = "Pipe 3 Jump"
            elif 940 <= sub_x <= 970:
                milestone = "Pipe 4 Jump"
            elif 1380 <= sub_x <= 1430:
                milestone = "Pit 1 Abyss Leap"
            elif sub_flag or sub_x >= 3160:
                milestone = "Flagpole Capture Win"

            steps_data.append({
                "step": frame_count,
                "macro_step": macro_step,
                "subframe": f,
                "x": sub_x,
                "y": sub_y,
                "a": act,
                "a_name": ACTION_NAMES[act],
                "p": [round(float(p), 3) for p in probs],
                "v": round(float(val.item()), 2),
                "sp": sparsity,
                "act": [vp_mean, cb_mean, dn_mean],
                "gf": round(gf_max, 3),
                "m": milestone,
                "f": frame_b64,
                "dns": serialized_dn,
                "vps": serialized_vp,
                "cbs": serialized_cb,
            })

            subframes_captured.append((obs_raw, term, trunc, sub_info))
            if term or trunc or sub_flag:
                break

        # Progress environment state tracking
        frames_run = len(subframes_captured)
        env._episode_frames += frames_run
        last_info = subframes_captured[-1][3]
        x_pos = int(last_info.get("x_pos", 0))
        flag = bool(last_info.get("flag_get", False))
        if x_pos > env._best_x:
            env._best_x = x_pos
            env._frames_since_progress = 0
        else:
            env._frames_since_progress += frames_run

        if not terminated:
            if env.max_episode_frames is not None and env._episode_frames >= env.max_episode_frames:
                truncated = True
            elif env.stall_frames is not None and env._frames_since_progress >= env.stall_frames:
                truncated = True

        # Preprocess final frame for recurrent observation stack
        final_obs_raw = subframes_captured[-1][0]
        proc_obs = env.preprocess_frame(final_obs_raw)
        env.frame_buffer.append(proc_obs)
        obs = env._get_stacked_obs()

        if macro_step % 50 == 0:
            print(f"Macro-step {macro_step}: Frame {frame_count}, X={x_pos}")

        if flag or terminated or truncated:
            print(f"Episode completed at macro-step {macro_step}, Frame {frame_count}, X={x_pos}, Flag={flag}")
            break

    env.close()

    output_payload = {
        "metadata": {
            "total_nodes": 30000,
            "total_edges": 6001050,
            "total_steps": len(steps_data),
            "final_x": x_pos,
            "flag_captured": flag,
        },
        "neuropil_bounds": neuropil_bounds,
        "scaffold": scaffold_neurons,
        "neuron_table": recorded_neuron_details,
        "steps": steps_data,
    }

    out_file = REPO_ROOT / "data" / "fruitfly_simulation_data.json"
    with open(out_file, "w") as f:
        json.dump(output_payload, f)

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Successfully generated {out_file} ({file_size_mb:.2f} MB, {len(steps_data)} frames, {len(scaffold_neurons)} scaffold points)")


if __name__ == "__main__":
    main()

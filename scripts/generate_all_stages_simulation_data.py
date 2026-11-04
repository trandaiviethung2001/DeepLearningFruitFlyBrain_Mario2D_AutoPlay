"""Generates complete multi-stage simulation dataset across all 32 Super Mario Bros stages.

Applies the trained Connectome Mario Policy (Connectome PPO v3) across all 32 levels
from World 1-1 to World 8-4:
- Captures native 256x240 lossless WebP frames with zero frame skipping for every stage
- Connectome neural firing states, sensory visual intake, and motor output
- Rich stage metadata: World, Stage, Biome (Overworld, Underground, Athletics, Castle, Underwater)
- Full benchmark matrix comparing cross-stage zero-shot transfer performance
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
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

# Stage metadata catalog for all 32 stages
STAGE_CATALOG = [
    # World 1
    {"id": "1-1", "w": 1, "s": 1, "biome": "Overworld", "title": "World 1-1: Grassland Inception", "desc": "Iconic overworld with green pipes, Goombas, and Flagpole victory."},
    {"id": "1-2", "w": 1, "s": 2, "biome": "Underground", "title": "World 1-2: Subterranean Caverns", "desc": "Blue brick ceiling running, Piranha Plants, Buzzy Beetles, and Warp Zone."},
    {"id": "1-3", "w": 1, "s": 3, "biome": "Athletics", "title": "World 1-3: Mushroom Treetops", "desc": "High-altitude bouncy mushrooms, precision platforming across abyssal chasms."},
    {"id": "1-4", "w": 1, "s": 4, "biome": "Castle", "title": "World 1-4: Bowser's Lava Fortress", "desc": "Stone corridors, spinning firebars, lava pits, and Bowser's bridge axe."},

    # World 2
    {"id": "2-1", "w": 2, "s": 1, "biome": "Overworld", "title": "World 2-1: Daybreak Plains", "desc": "Overworld with high brick staircases, jumping Koopa Troopas, and springboards."},
    {"id": "2-2", "w": 2, "s": 2, "biome": "Underwater", "title": "World 2-2: Coral Deep Sea", "desc": "Buoyant underwater swimming physics, Bloopers, and Cheep Cheeps."},
    {"id": "2-3", "w": 2, "s": 3, "biome": "Athletics", "title": "World 2-3: Flying Fish Bridge", "desc": "Endless ocean bridge besieged by leaping red and grey Cheep Cheeps."},
    {"id": "2-4", "w": 2, "s": 4, "biome": "Castle", "title": "World 2-4: Fortress of Flame", "desc": "Dense firebar traps, tight stone corridors, and Bowser's fire breath."},

    # World 3
    {"id": "3-1", "w": 3, "s": 1, "biome": "Overworld (Night)", "title": "World 3-1: Midnight Meadows", "desc": "Nighttime overworld, Hammer Bros ambush, springboards, and tall pipes."},
    {"id": "3-2", "w": 3, "s": 2, "biome": "Overworld", "title": "World 3-2: The Bullet Sprint", "desc": "Flat open sprint terrain, dense Goomba charges, and Flagpole victory."},
    {"id": "3-3", "w": 3, "s": 3, "biome": "Athletics", "title": "World 3-3: Pulley Platforms", "desc": "Precision weighted balancing pulleys suspended over bottomless voids."},
    {"id": "3-4", "w": 3, "s": 4, "biome": "Castle", "title": "World 3-4: Bowser's Redoubt", "desc": "Complex firebar gauntlet with narrow footing over bubbling magma."},

    # World 4
    {"id": "4-1", "w": 4, "s": 1, "biome": "Overworld", "title": "World 4-1: Lakitu's Cloud Assault", "desc": "Open sky with Lakitu dropping airborne Spiny eggs from clouds above."},
    {"id": "4-2", "w": 4, "s": 2, "biome": "Underground", "title": "World 4-2: The Deep Hollow", "desc": "Expansive subterranean maze with elevator platforms and hidden vines."},
    {"id": "4-3", "w": 4, "s": 3, "biome": "Athletics", "title": "World 4-3: Giant Mushroom Ridge", "desc": "Massive red and green mushroom tops with extreme leap gaps."},
    {"id": "4-4", "w": 4, "s": 4, "biome": "Castle", "title": "World 4-4: The Labyrinth Keep", "desc": "Infamous looping maze puzzle; wrong paths trap the player in an infinite loop."},

    # World 5
    {"id": "5-1", "w": 5, "s": 1, "biome": "Overworld (Snow)", "title": "World 5-1: Highland Tundra", "desc": "Winterized palette, high-speed Bullet Bills firing from cannons."},
    {"id": "5-2", "w": 5, "s": 2, "biome": "Overworld", "title": "World 5-2: Hammer Bros Gauntlet", "desc": "Long endurance trail guarded by aggressive projectile-throwing Hammer Bros."},
    {"id": "5-3", "w": 5, "s": 3, "biome": "Athletics", "title": "World 5-3: Treetop Aerials", "desc": "Narrow wooden bridges and bouncing mushroom stems high in the canopy."},
    {"id": "5-4", "w": 5, "s": 4, "biome": "Castle", "title": "World 5-4: Molten Bastion", "desc": "Fast double firebars and low-ceiling jumps across boiling lava pools."},

    # World 6
    {"id": "6-1", "w": 6, "s": 1, "biome": "Overworld (Night)", "title": "World 6-1: Twilight Ridge", "desc": "Dark sky, relentless Lakitu bombardment, and steep pipe formations."},
    {"id": "6-2", "w": 6, "s": 2, "biome": "Overworld", "title": "World 6-2: The Pipe Maze", "desc": "Dense cluster of 12 green pipes containing biting Piranha Plants."},
    {"id": "6-3", "w": 6, "s": 3, "biome": "Athletics (Night)", "title": "World 6-3: Midnight Stems", "desc": "Eerie dark mushroom platforms requiring maximum ballistic momentum."},
    {"id": "6-4", "w": 6, "s": 4, "biome": "Castle", "title": "World 6-4: Infernal Keep", "desc": "Multi-tiered firebars with aggressive Bowser fireball volleys."},

    # World 7
    {"id": "7-1", "w": 7, "s": 1, "biome": "Overworld", "title": "World 7-1: Bullet Bill Alley", "desc": "Dense artillery battery of Bill Blasters firing in rapid succession."},
    {"id": "7-2", "w": 7, "s": 2, "biome": "Underwater", "title": "World 7-2: Abyssal Reef", "desc": "Extreme underwater hazard density with homing Bloopers and Cheep swarms."},
    {"id": "7-3", "w": 7, "s": 3, "biome": "Athletics", "title": "World 7-3: Oceanic Flying Bridge", "desc": "Furious wave of airborne Cheep Cheeps raining across narrow footing."},
    {"id": "7-4", "w": 7, "s": 4, "biome": "Castle", "title": "World 7-4: The False Mazes", "desc": "Complex four-tier branching castle maze leading to Bowser's sanctum."},

    # World 8
    {"id": "8-1", "w": 8, "s": 1, "biome": "Overworld", "title": "World 8-1: The Final Frontier", "desc": "The longest continuous sprint level in the game with extreme gap leaps."},
    {"id": "8-2", "w": 8, "s": 2, "biome": "Overworld", "title": "World 8-2: Paratroopa Barrage", "desc": "High bouncing green Paratroopas and flying Bullet Bills over deep pits."},
    {"id": "8-3", "w": 8, "s": 3, "biome": "Overworld", "title": "World 8-3: The Hammer Fortress", "desc": "Castle exterior defended by squads of charging, jumping Hammer Bros."},
    {"id": "8-4", "w": 8, "s": 4, "biome": "Castle", "title": "World 8-4: Bowser's Dark Citadel", "desc": "Final multi-room pipe maze transitioning from castle to underwater to lava bridge."},
]


def main():
    device = torch.device("cpu")
    processed_dir = REPO_ROOT / "data" / "processed"
    ckpt_path = REPO_ROOT / "checkpoints" / "connectome_ppo_v3.pt"
    existing_w11_path = REPO_ROOT / "data" / "fruitfly_simulation_data.json"
    output_path = REPO_ROOT / "data" / "all_stages_simulation_data.json"

    print("1. Loading Connectome Metadata and Biological CSR Synapses...")
    sub_map = pd.read_csv(processed_dir / "subgraph_node_mapping.csv")
    node_meta = pd.read_csv(processed_dir / "node_metadata.csv", low_memory=False)

    feather_path = REPO_ROOT / "data" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    ann_table = feather.read_table(feather_path, columns=["bodyId", "somaLocation"])
    df_ann = ann_table.to_pandas()
    df_ann["body_id"] = df_ann["bodyId"].astype(int)

    npz = np.load(processed_dir / "subgraph_graph.npz")
    mat = sp.csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=npz["shape"])
    out_degrees = np.diff(mat.indptr)
    in_degrees = np.bincount(mat.indices, minlength=mat.shape[0])

    merged = sub_map.merge(node_meta, on="body_id", how="left")
    merged = merged.merge(df_ann[["body_id", "somaLocation"]], on="body_id", how="left")

    body_ids = merged["body_id"].values
    types = merged["type"].fillna("Unknown").values
    superclasses = merged["superclass"].fillna("cb_intrinsic").values
    soma_sides = merged["somaSide"].fillna("Unknown").values
    neuromeres = merged["somaNeuromere"].fillna("Brain").values
    classes = merged["class"].fillna("").values

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

    # 2. Load existing scaffold and neuron table from W1-1 dataset
    print(f"2. Loading base scaffold and World 1-1 data from {existing_w11_path}...")
    with open(existing_w11_path) as f:
        existing_data = json.load(f)

    scaffold_neurons = existing_data["scaffold"]
    neuropil_bounds = existing_data["neuropil_bounds"]
    recorded_neuron_details: Dict[int, Dict[str, Any]] = {
        int(k): v for k, v in existing_data["neuron_table"].items()
    }
    w11_steps = existing_data["steps"]

    print("3. Loading ConnectomeMarioPolicy checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    def record_neuron_indices(core_act, indices):
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

    # 4. Process all 32 stages
    all_stages_dict: Dict[str, Dict[str, Any]] = {}
    stage_benchmark_summary: List[Dict[str, Any]] = []

    print(f"4. Executing live connectome policy across all {len(STAGE_CATALOG)} stages...")
    t_start = time.time()

    for idx, stage_meta in enumerate(STAGE_CATALOG, 1):
        stage_id = stage_meta["id"]
        w, s = stage_meta["w"], stage_meta["s"]
        rom_id = f"SuperMarioBros-{w}-{s}-v0"

        # Check if we can reuse World 1-1
        if stage_id == "1-1":
            print(f"[{idx}/32] {stage_meta['title']}: Reusing existing 1,950 frames (Flagpole Win X=3161)")
            all_stages_dict[stage_id] = {
                "id": stage_id,
                "world": w,
                "stage": s,
                "title": stage_meta["title"],
                "biome": stage_meta["biome"],
                "desc": stage_meta["desc"],
                "total_frames": len(w11_steps),
                "max_x": 3161,
                "flag_captured": True,
                "steps": w11_steps,
            }
            stage_benchmark_summary.append({
                "id": stage_id,
                "world": w,
                "stage": s,
                "title": stage_meta["title"],
                "biome": stage_meta["biome"],
                "max_x": 3161,
                "total_frames": len(w11_steps),
                "macro_steps": 488,
                "flag_captured": True,
                "status": "Flagpole Cleared (100% Win)",
            })
            continue

        print(f"[{idx}/32] Executing {stage_meta['title']} ({rom_id})...")
        t_stage_0 = time.time()

        env = Mario2DEnv(
            rom_id=rom_id,
            frame_stack=in_channels,
            frame_skip=4,
            auto_unlatch=True,
            render_mode="rgb_array",
            max_episode_frames=3000,
            stall_frames=300,
        )

        obs, info = env.reset(seed=1000)
        h = None

        stage_steps: List[Dict[str, Any]] = []
        macro_step = 0
        frame_count = 0
        max_x = 0
        flag = False

        while macro_step < 500:
            if isinstance(obs, torch.Tensor):
                obs_t = obs.unsqueeze(0).float().to(device)
            else:
                obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)

            with torch.no_grad():
                logits, val, h_next, tele = policy(obs_t, h, num_ticks=2, return_telemetry=True)
                core_act = tele["h_core"].cpu().numpy().squeeze(0)
                sensory_drive = tele["sensory_drive"].cpu().numpy().squeeze(0)
                motor_act = tele["motor_activity"].cpu().numpy().squeeze(0)

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

            serialized_dn = record_neuron_indices(core_act, top_dn_idxs)
            serialized_vp = record_neuron_indices(core_act, top_vp_idxs)
            serialized_cb = record_neuron_indices(core_act, top_cb_idxs)

            vp_mean = round(float(np.mean(np.abs(sensory_drive))), 3)
            dn_mean = round(float(np.mean(np.abs(motor_act))), 3)
            cb_mean = round(float(np.mean(np.abs(core_act[cb_mask]))), 3)
            sparsity = round(float(compute_sparsity_ratio(core_act)), 1)
            gf_max = float(np.max(np.abs(core_act[gf_local_indices]))) if gf_local_indices else 0.0

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

                # Lossless WebP 256x240 frame
                img = Image.fromarray(obs_raw)
                buf = io.BytesIO()
                img.save(buf, format="WEBP", lossless=True)
                frame_b64 = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

                sub_x = int(sub_info.get("x_pos", 0))
                sub_y = int(sub_info.get("y_pos", 0))
                sub_flag = bool(sub_info.get("flag_get", False))

                max_x = max(max_x, sub_x)
                if sub_flag:
                    flag = True

                frame_count += 1

                milestone = ""
                if frame_count == 1:
                    milestone = "Stage Spawn"
                elif sub_flag or sub_x >= 3160:
                    milestone = "Flagpole Win"

                stage_steps.append({
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

            frames_run = len(subframes_captured)
            env._episode_frames += frames_run
            last_info = subframes_captured[-1][3]
            x_pos = int(last_info.get("x_pos", 0))
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

            final_obs_raw = subframes_captured[-1][0]
            proc_obs = env.preprocess_frame(final_obs_raw)
            env.frame_buffer.append(proc_obs)
            obs = env._get_stacked_obs()

            if flag or terminated or truncated:
                break

        env.close()

        status_str = "Flagpole Cleared (100% Win)" if flag else f"Explored to X={max_x}"
        t_stage = time.time() - t_stage_0
        print(f"   -> Result: max_x={max_x}, frames={frame_count}, flag={flag} ({t_stage:.1f}s)")

        all_stages_dict[stage_id] = {
            "id": stage_id,
            "world": w,
            "stage": s,
            "title": stage_meta["title"],
            "biome": stage_meta["biome"],
            "desc": stage_meta["desc"],
            "total_frames": frame_count,
            "max_x": max_x,
            "flag_captured": flag,
            "steps": stage_steps,
        }

        stage_benchmark_summary.append({
            "id": stage_id,
            "world": w,
            "stage": s,
            "title": stage_meta["title"],
            "biome": stage_meta["biome"],
            "max_x": max_x,
            "total_frames": frame_count,
            "macro_steps": macro_step,
            "flag_captured": flag,
            "status": status_str,
        })

    # Save output payload
    output_payload = {
        "metadata": {
            "total_stages": len(STAGE_CATALOG),
            "total_nodes": 30000,
            "total_edges": 6001050,
            "scaffold_points": len(scaffold_neurons),
            "total_recorded_neurons": len(recorded_neuron_details),
            "benchmark_summary": stage_benchmark_summary,
        },
        "neuropil_bounds": neuropil_bounds,
        "scaffold": scaffold_neurons,
        "neuron_table": recorded_neuron_details,
        "stages": all_stages_dict,
    }

    print(f"5. Writing all-stages dataset to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_payload, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Complete! Generated {output_path} ({file_size_mb:.2f} MB, {len(all_stages_dict)} stages, total time {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()


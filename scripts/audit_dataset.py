"""Phase-4 audit: behavioral-cloning demonstration dataset quality.

Run: .venv/bin/python scripts/audit_dataset.py [path]
Writes logs/audit_dataset.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

NAMES = ["NOOP", "RIGHT", "LEFT", "JUMP", "RIGHT+JUMP"]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data/demonstrations/scaled_demonstrations.pt"
    d = torch.load(path, weights_only=False)
    print("keys:", list(d.keys()) if isinstance(d, dict) else type(d))
    out = {"path": str(path)}
    obs, act = d["obs"], d["actions"]
    ep = d.get("episode_ids"); seeds = d.get("seeds")
    out["n_frames"] = int(obs.shape[0]); out["obs_shape"] = list(obs.shape[1:])
    out["n_episodes"] = int(ep.unique().numel()) if ep is not None else None
    cnt = np.bincount(act.numpy(), minlength=5)
    out["action_distribution"] = {NAMES[i]: round(float(c / cnt.sum()), 4) for i, c in enumerate(cnt)}
    print("frames", out["n_frames"], "episodes", out["n_episodes"], "actions", out["action_distribution"])

    # Splits
    for k in ("splits", "split_indices", "train_idx", "train_indices"):
        if k in d:
            out["split_key"] = k
    meta = d.get("metadata", {})
    out["metadata_keys"] = list(meta.keys()) if isinstance(meta, dict) else None
    if isinstance(meta, dict) and "splits" in meta:
        out["splits"] = {k: (v if not torch.is_tensor(v) else v.tolist()) for k, v in meta["splits"].items()} \
            if isinstance(meta["splits"], dict) else str(meta["splits"])[:300]

    # Near-duplicates: consecutive frames within an episode
    flat = obs.reshape(obs.shape[0], -1).float()
    diffs = (flat[1:] - flat[:-1]).abs().mean(1)
    same_ep = (ep[1:] == ep[:-1]) if ep is not None else torch.ones_like(diffs, dtype=torch.bool)
    d_in = diffs[same_ep]
    out["consecutive_mean_abs_pixel_diff"] = {"median": float(d_in.median()), "p10": float(d_in.quantile(0.1))}
    out["consecutive_exact_duplicates_frac"] = float((d_in == 0).float().mean())
    # Exact duplicate frames dataset-wide (hash)
    hashes = [hash(o.numpy().tobytes()) for o in obs]
    out["unique_frames_frac"] = round(len(set(hashes)) / len(hashes), 4)

    # Same observation, different label (label ambiguity)
    from collections import defaultdict
    lab = defaultdict(set)
    for h, a in zip(hashes, act.tolist()):
        lab[h].add(a)
    out["frames_with_conflicting_labels_frac"] = round(sum(len(v) > 1 for v in lab.values()) / len(lab), 4)

    # Jump phase structure: jumps are runs of action 4
    a = act.numpy(); runs = []
    i = 0
    while i < len(a):
        if a[i] == 4:
            j = i
            while j < len(a) and a[j] == 4 and (ep is None or ep[j] == ep[i]):
                j += 1
            runs.append(j - i); i = j
        else:
            i += 1
    out["jump_runs"] = {"count": len(runs), "mean_len": float(np.mean(runs)) if runs else 0,
                        "min": int(min(runs)) if runs else 0, "max": int(max(runs)) if runs else 0}
    # Fraction of RIGHT+JUMP frames that are jump *onsets* (the decisive timing frames)
    out["jump_onset_frames_frac_of_all"] = round(len(runs) / len(a), 4)

    # Action by x region (obstacle approach) if x available
    if "x_positions" in d:
        x = d["x_positions"].numpy()
        regions = {"pre_goomba_200_296": (200, 296), "pipe1_380_440": (380, 440), "flat_100_200": (100, 200)}
        out["action_by_region"] = {}
        for name, (lo, hi) in regions.items():
            m = (x >= lo) & (x < hi)
            c = np.bincount(a[m], minlength=5)
            out["action_by_region"][name] = {NAMES[i]: int(v) for i, v in enumerate(c)}
        out["episode_max_x"] = {}
        for e in ep.unique().tolist():
            out["episode_max_x"][int(e)] = int(x[(ep == e).numpy()].max())
    print(json.dumps({k: v for k, v in out.items() if k != "episode_max_x"}, indent=1, default=str))
    (REPO_ROOT / "logs/audit_dataset.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()


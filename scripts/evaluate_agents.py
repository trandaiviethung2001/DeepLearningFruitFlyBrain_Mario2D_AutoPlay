"""Evaluate agents under the shared protocol and print a comparison table.

Examples:
  # Reproduce the pre-fix state (legacy checkpoints, random, untrained, rule-based)
  .venv/bin/python scripts/evaluate_agents.py --suite legacy --out logs/eval_before.json
  # Evaluate new checkpoints
  .venv/bin/python scripts/evaluate_agents.py --ckpt checkpoints/ppo_v2/connectome_s1.pt --out logs/eval_x.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore")

import torch

from src.evaluation import (DEFAULT_EVAL_SEEDS, ConstantAgent, EvalProtocol, PolicyAgent, RandomAgent,
                            ScriptedAgent, build_model, evaluate_agent, load_policy_agent, save_results)


def table(results):
    hdr = (f"{'agent':<44} {'skip':>4} {'ret':>14} {'max_x':>14} {'goomba1':>7} {'pipe1':>6} "
           f"{'pipe2':>6} {'death':>6} {'trunc':>6} {'flag':>5}  actions[N,R,L,J,RJ]")
    print(hdr); print("-" * len(hdr))
    for r in results:
        s = r["summary"]; ad = s["action_dist"]
        print(f"{r['agent'][:44]:<44} {r['frame_skip']:>4} {s['return_mean']:>7.1f}±{s['return_std']:<6.1f}"
              f"{s['max_x_mean']:>7.1f}±{s['max_x_std']:<6.1f} {s['clear_rate_goomba1']:>7.2f} "
              f"{s['clear_rate_pipe1']:>6.2f} {s['clear_rate_pipe2']:>6.2f} {s['death_rate']:>6.2f} "
              f"{s['truncation_rate']:>6.2f} {s['completion_rate']:>5.2f}  "
              f"{[ad[k] for k in ['NOOP','RIGHT','LEFT','JUMP','RIGHT+JUMP']]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["legacy", "controls", "none"], default="none")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--modes", nargs="*", default=["sample", "greedy"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", type=str, default="logs/eval_results.json")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    proto = EvalProtocol(seeds=DEFAULT_EVAL_SEEDS[: args.episodes] if args.episodes <= 10
                         else list(range(1000, 1000 + args.episodes)))
    agents = []
    if args.suite in ("legacy", "controls"):
        agents += [RandomAgent(frame_skip=1), RandomAgent(frame_skip=4), ConstantAgent(1, 1), ScriptedAgent()]
        torch.manual_seed(0)
        agents.append(PolicyAgent(build_model("baseline", 4, dev), dev, 4, 4, "sample", "untrained GRU[sample]"))
        torch.manual_seed(0)
        agents.append(PolicyAgent(build_model("connectome", 4, dev), dev, 4, 4, "sample", "untrained MaleCNS[sample]"))
    if args.suite == "legacy":
        for c in ["connectome_model_best", "baseline_gru_best", "connectome_ppo_best", "baseline_ppo_best"]:
            p = REPO_ROOT / "checkpoints" / f"{c}.pt"
            if p.exists():
                for m in args.modes:
                    agents.append(load_policy_agent(p, dev, m, name=f"legacy {c}[{m}]"))
    for c in args.ckpt:
        for m in args.modes:
            agents.append(load_policy_agent(c, dev, m))

    results = []
    for a in agents:
        t = time.time()
        r = evaluate_agent(a, proto)
        results.append(r)
        print(f"  evaluated {a.name} in {time.time()-t:.1f}s", flush=True)
    save_results(results, REPO_ROOT / args.out)
    print()
    table(results)
    print(f"\nProtocol: seeds={proto.seeds} max_frames={proto.max_episode_frames} "
          f"stall_frames={proto.stall_frames} noop_max={proto.noop_max}. Saved {args.out}")


if __name__ == "__main__":
    main()


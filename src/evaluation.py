"""Unified evaluation protocol for all Mario agents.

Every agent is evaluated with identical settings:
- World 1-1 from the true level start (no curriculum start states)
- seeded random no-op starts (0..noop_max NES frames) -> the seed actually changes the episode
- the same budget in NES frames (max_episode_frames) and stall truncation (stall_frames)
- the same raw task reward (gym_super_mario_bros reward, summed over frames)

Metrics: return, max x, landmark clearance (Goomba-1, Pipe-1, ...), death rate,
level-completion rate, truncation rate, action distribution.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.environment import Mario2DEnv, WORLD_1_1_LANDMARKS

ACTION_NAMES = ["NOOP", "RIGHT", "LEFT", "JUMP", "RIGHT+JUMP"]
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"


@dataclass
class EvalProtocol:
    seeds: List[int]
    max_episode_frames: int = 5000
    stall_frames: int = 600
    noop_max: int = 30


DEFAULT_EVAL_SEEDS = list(range(1000, 1010))


# --------------------------------------------------------------------------- agents
class Agent:
    """Interface: reset(seed) at episode start, act(obs, info, env) -> action int."""
    frame_stack: int = 4
    frame_skip: int = 4
    name: str = "agent"

    def reset(self, seed: int) -> None:  # pragma: no cover - trivial
        pass

    def act(self, obs: torch.Tensor, info: dict, env: Mario2DEnv) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, frame_skip: int = 4):
        self.frame_skip, self.frame_stack, self.name = frame_skip, 1, f"random(skip={frame_skip})"

    def reset(self, seed):
        self.rng = np.random.default_rng(seed + 7)

    def act(self, obs, info, env):
        return int(self.rng.integers(0, 5))


class ConstantAgent(Agent):
    def __init__(self, action: int, frame_skip: int = 1):
        self.action, self.frame_skip, self.frame_stack = action, frame_skip, 1
        self.name = f"always-{ACTION_NAMES[action]}"

    def act(self, obs, info, env):
        return self.action


class ScriptedAgent(Agent):
    """RAM-privileged rule-based controller (ScriptedDemonstrator), native frame_skip=1."""

    def __init__(self):
        from src.demonstration_collector import ScriptedDemonstrator
        self.d = ScriptedDemonstrator()
        self.frame_skip, self.frame_stack, self.name = 1, 1, "rule-based (RAM, privileged)"

    def reset(self, seed):
        self.d.reset(seed)

    def act(self, obs, info, env):
        return self.d.get_action(obs, info, env.ram)


class PolicyAgent(Agent):
    """Neural policy (connectome or GRU). mode: 'sample' or 'greedy'."""

    def __init__(self, model: torch.nn.Module, device: torch.device, frame_stack: int, frame_skip: int,
                 mode: str = "sample", name: str = "policy", jump_hold: int = 0, auto_unlatch: bool = False):
        self.model, self.device = model.eval(), device
        self.frame_stack, self.frame_skip, self.mode, self.name = frame_stack, frame_skip, mode, name
        self.jump_hold = jump_hold
        self.auto_unlatch = auto_unlatch
        self.h: Optional[torch.Tensor] = None
        self.gen = torch.Generator()

    def reset(self, seed):
        self.h = None
        self.gen.manual_seed(seed)
        self._hold = 0

    @torch.no_grad()
    def act(self, obs, info, env):
        logits, _, self.h = self.model(obs.unsqueeze(0).to(self.device), self.h)
        if self.mode == "greedy":
            a = int(logits.argmax(-1).item())
        else:
            p = torch.softmax(logits.float().cpu(), -1)[0]
            a = int(torch.multinomial(p, 1, generator=self.gen).item())
        # Legacy viewer hack (eval-only jump sustain); disabled by default for fair evaluation.
        if self.jump_hold > 0:
            if self._hold > 0:
                self._hold -= 1
                return 4
            if a in (3, 4):
                self._hold = self.jump_hold
                return 4
        return a


def build_model(model_type: str, frame_stack: int, device: torch.device, graph: str = "subgraph_graph.npz"):
    if model_type == "connectome":
        from src.connectome_model import ConnectomeMarioPolicy
        return ConnectomeMarioPolicy(
            subgraph_npz_path=PROCESSED / graph,
            node_mapping_csv_path=PROCESSED / "subgraph_node_mapping.csv",
            node_metadata_csv_path=PROCESSED / "node_metadata.csv",
            in_channels=frame_stack,
        ).to(device)
    from src.baseline_model import BaselineMarioPolicy
    return BaselineMarioPolicy(in_channels=frame_stack, hidden_dim=1024).to(device)


def load_policy_agent(ckpt_path: str | Path, device: torch.device, mode: str = "sample",
                      model_type: Optional[str] = None, frame_skip: Optional[int] = None,
                      name: Optional[str] = None, auto_unlatch: Optional[bool] = None) -> PolicyAgent:
    """Restore a checkpoint. New checkpoints carry a 'config'; legacy ones are inferred."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("config", {})
    mtype = model_type or cfg.get("model_type") or ("connectome" if "rnn.bias" in sd else "baseline")
    fstack = cfg.get("frame_stack") or int(sd["encoder.conv1.weight"].shape[1])
    fskip = frame_skip or cfg.get("frame_skip") or 1  # legacy checkpoints were trained at 1 frame/step
    unlatch = auto_unlatch if auto_unlatch is not None else cfg.get("auto_unlatch", False)
    model = build_model(mtype, fstack, device)
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    return PolicyAgent(model, device, fstack, fskip, mode, name or f"{Path(ckpt_path).stem}[{mode}]", auto_unlatch=unlatch)


# --------------------------------------------------------------------------- evaluation
def run_episode(agent: Agent, env: Mario2DEnv, seed: int) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed, use_start_states=False)
    agent.reset(seed)
    ret, max_x, steps = 0.0, int(info.get("x_pos", 40)), 0
    counts = np.zeros(5, dtype=np.int64)
    term = trunc = False
    while True:
        a = agent.act(obs, info, env)
        counts[a] += 1
        obs, r, term, trunc, info = env.step(a)
        ret += r
        steps += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
        if term or trunc:
            break
    flag = bool(info.get("flag_get", False))
    res = {
        "seed": seed, "return": round(ret, 2), "max_x": max_x, "agent_steps": steps,
        "frames": int(info.get("episode_frames", steps)), "flag_get": flag,
        "death": bool(term and not flag), "truncated": bool(trunc),
        "truncation_reason": info.get("truncation_reason"), "action_counts": counts.tolist(),
    }
    for k, x in WORLD_1_1_LANDMARKS.items():
        res[f"cleared_{k}"] = max_x >= x
    return res


def summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    R = np.array([e["return"] for e in episodes]); X = np.array([e["max_x"] for e in episodes])
    counts = np.sum([e["action_counts"] for e in episodes], axis=0)
    p = counts / max(1, counts.sum())
    s = {
        "n_episodes": len(episodes),
        "return_mean": round(float(R.mean()), 2), "return_std": round(float(R.std()), 2),
        "max_x_mean": round(float(X.mean()), 1), "max_x_std": round(float(X.std()), 1),
        "max_x_best": int(X.max()),
        "death_rate": round(float(np.mean([e["death"] for e in episodes])), 3),
        "truncation_rate": round(float(np.mean([e["truncated"] for e in episodes])), 3),
        "completion_rate": round(float(np.mean([e["flag_get"] for e in episodes])), 3),
        "action_dist": {ACTION_NAMES[i]: round(float(p[i]), 3) for i in range(5)},
        "action_entropy_nats": round(float(-sum(q * math.log(q) for q in p if q > 0)), 3),
    }
    for k in WORLD_1_1_LANDMARKS:
        s[f"clear_rate_{k}"] = round(float(np.mean([e[f"cleared_{k}"] for e in episodes])), 3)
    return s


def evaluate_agent(agent: Agent, protocol: EvalProtocol) -> Dict[str, Any]:
    auto_unlatch = getattr(agent, "auto_unlatch", False)
    env = Mario2DEnv(frame_stack=agent.frame_stack, frame_skip=agent.frame_skip,
                     noop_max=protocol.noop_max, max_episode_frames=protocol.max_episode_frames,
                     stall_frames=protocol.stall_frames, auto_unlatch=auto_unlatch)
    try:
        eps = [run_episode(agent, env, s) for s in protocol.seeds]
    finally:
        env.close()
    return {"agent": agent.name, "frame_skip": agent.frame_skip, "frame_stack": agent.frame_stack,
            "protocol": asdict(protocol), "summary": summarize(eps), "episodes": eps}


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


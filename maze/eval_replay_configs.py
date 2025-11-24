#!/usr/bin/env python3
"""
Evaluate alternative replay-buffer style training loops for DQN agents
when performing dream-environment updates. We instantiate a maze, build
the debug world model (so imagined transitions are accurate), and then
compare several strategies for how to use trajectories collected from
the dream environment.

Variants:
  • on_policy: current behaviour – run the agent directly in DreamEnv
               for N × max_steps transitions (pure online TD)
  • uniform_replay: collect N trajectories in DreamEnv and replay them
                    uniformly without oversampling beyond the buffer size
  • success_replay: replay only trajectories that contain a goal hit
                    (falls back to uniform when no successes exist)

For each variant we report the real-environment validation reward before
and after dream training, aggregated across multiple seeds.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

if __package__ is None:
    # Allow running the script directly from the project root.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in os.sys.path:
        os.sys.path.append(SCRIPT_DIR)

from config import Config
from dqn_agent import BaseDQNAgent, DQNConfig, make_dqn_agent
from environment import DreamEnv, MazeEnv
from rl_valuator import compute_validation_reward
from seed_utils import seed_all
from utils import collect_episodes
from world_model import WorldModel


def deterministic_next_idx(start_idx: int, action: int, maze_layout: np.ndarray, grid: int) -> int:
    """Mirror MazeEnv's deterministic transition for cache-based world models."""
    row, col = divmod(start_idx, grid)
    if action == 0 and row > 0 and maze_layout[row - 1, col] == 0:
        row -= 1
    elif action == 1 and row < grid - 1 and maze_layout[row + 1, col] == 0:
        row += 1
    elif action == 2 and col > 0 and maze_layout[row, col - 1] == 0:
        col -= 1
    elif action == 3 and col < grid - 1 and maze_layout[row, col + 1] == 0:
        col += 1
    return row * grid + col


def build_debug_world_model(real_env: MazeEnv) -> WorldModel:
    """Populate a debug/cache world model with exact transitions from the real maze."""
    obs_dim = real_env.observation_space.shape[0]
    action_dim = real_env.action_space.n
    world_model = WorldModel(obs_dim, action_dim, model_type="debug")
    maze_layout = np.array(real_env.maze, copy=True)
    grid = real_env.grid_size

    num_cells = grid * grid
    for start_idx in range(num_cells):
        for action in range(action_dim):
            next_idx = deterministic_next_idx(start_idx, action, maze_layout, grid)
            world_model.set_transition_index(start_idx, action, next_idx)

    # Let DreamEnv know about barriers so it samples only free cells.
    world_model.cache  # touch to satisfy mypy-type linters
    return world_model


def free_cell_indices(maze_layout: np.ndarray) -> List[int]:
    rows, cols = np.where(maze_layout == 0)
    return [int(r * maze_layout.shape[0] + c) for r, c in zip(rows, cols)]


def sample_allowed_pairs(maze_layout: np.ndarray, grid: int, top_n: int) -> List[Tuple[int, int]]:
    free_cells = free_cell_indices(maze_layout)
    if len(free_cells) < 2:
        return []
    rng = np.random.default_rng()
    pairs: List[Tuple[int, int]] = []
    attempts = 0
    max_pairs = min(top_n, len(free_cells) * (len(free_cells) - 1))
    while len(pairs) < max_pairs and attempts < max_pairs * 3:
        start, goal = rng.choice(free_cells, size=2, replace=False)
        pair = (int(start), int(goal))
        if pair not in pairs:
            pairs.append(pair)
        attempts += 1
    return pairs


def has_success(episode: Sequence[Tuple[np.ndarray, int, float, np.ndarray, bool, dict]]) -> bool:
    return any(transition[2] >= 0.99 for transition in episode)


def train_in_dream(
    agent: BaseDQNAgent,
    dream_env: DreamEnv,
    strategy: str,
    episodes_per_burst: int,
    episode_max_steps: int,
) -> Dict[str, float]:
    """
    Apply one dream-environment training update using the requested replay strategy.
    Returns simple bookkeeping metrics for diagnostic logging.
    """
    original_env = agent.get_env()
    agent.set_env(dream_env)

    stats: Dict[str, float] = {}

    if strategy == "on_policy":
        total_timesteps = episodes_per_burst * episode_max_steps
        agent.learn(total_timesteps=total_timesteps)
        stats["replay_episodes"] = 0
        stats["dream_steps"] = float(total_timesteps)

    else:
        episodes = collect_episodes(
            dream_env,
            agent,
            num_episodes=episodes_per_burst,
            max_steps=episode_max_steps,
        )
        stats["collected_episodes"] = len(episodes)
        stats["success_episodes"] = float(sum(1 for ep in episodes if has_success(ep)))

        if strategy == "success_replay":
            filtered = [ep for ep in episodes if has_success(ep)]
            if filtered:
                episodes = filtered
                stats["filtered_for_success"] = 1.0
            else:
                stats["filtered_for_success"] = 0.0

        elif strategy == "success_mix":
            successes = [ep for ep in episodes if has_success(ep)]
            failures = [ep for ep in episodes if not has_success(ep)]
            if successes:
                stats["success_pool"] = float(len(successes))
                stats["failure_pool"] = float(len(failures))
                target = len(episodes)
                chosen = list(successes)
                if len(chosen) < target and failures:
                    needed = target - len(chosen)
                    chosen.extend(random.sample(failures, min(needed, len(failures))))
                episodes = chosen
                stats["filtered_for_success"] = 1.0
            else:
                stats["success_pool"] = 0.0
                stats["failure_pool"] = float(len(failures))
                stats["filtered_for_success"] = 0.0

        total_episodes = len(episodes)
        if total_episodes == 0:
            stats["replay_episodes"] = 0.0
        else:
            agent.learn_from_episodes(episodes, total_episodes=total_episodes)
            stats["replay_episodes"] = float(total_episodes)

    agent.set_env(original_env)
    return stats


def evaluate_strategies(
    seeds: Iterable[int],
    initial_training_steps: int,
    episodes_per_burst: int,
    episode_max_steps: int,
    eval_episodes: int,
    strategies: Sequence[str],
) -> Dict[str, Dict[str, List[float]]]:
    """
    For each seed and strategy, evaluate the real-environment reward before and after
    a single dream-environment training burst. Returns per-strategy collections of results.
    """
    outputs: Dict[str, Dict[str, List[float]]] = {
        strat: {"before": [], "after": [], "delta": []} for strat in strategies
    }

    for seed in seeds:
        seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        real_env = MazeEnv(grid_size=Config.GRID_SIZE)
        dqn_cfg = DQNConfig(
            learning_rate=float(Config.DQN_LEARNING_RATE),
            discount=float(Config.Q_DISCOUNT),
            epsilon=float(Config.Q_EPSILON),
            epsilon_decay=float(Config.Q_EPSILON_DECAY),
            min_epsilon=float(Config.Q_MIN_EPSILON),
            hidden_size=int(Config.DQN_HIDDEN_SIZE),
            sb3_kwargs=getattr(Config, "SB3_DQN_KWARGS", None),
        )

        base_agent = make_dqn_agent(
            real_env,
            cfg=dqn_cfg,
            agent_type=getattr(Config, "AGENT_TYPE", "dqn"),
        )
        if initial_training_steps > 0:
            base_agent.learn(total_timesteps=initial_training_steps)

        base_params = base_agent.get_parameters()
        maze_layout = np.array(real_env.maze, copy=True)
        world_model = build_debug_world_model(real_env)
        allowed_pairs = sample_allowed_pairs(maze_layout, real_env.grid_size, episodes_per_burst)

        dream_env = DreamEnv(
            world_model=world_model,
            obs_dim=real_env.observation_space.shape[0],
            action_dim=real_env.action_space.n,
            barriers=maze_layout,
            allowed_pairs=allowed_pairs,
        )

        for strategy in strategies:
            # Fresh agent clone per strategy to keep comparisons fair.
            agent = make_dqn_agent(
                real_env,
                cfg=dqn_cfg,
                agent_type=getattr(Config, "AGENT_TYPE", "dqn"),
            )
            agent.set_parameters(base_params)

            reward_before = compute_validation_reward(
                agent,
                real_env,
                episodes=eval_episodes,
                max_steps=25,
            )

            stats = train_in_dream(agent, dream_env, strategy, episodes_per_burst, episode_max_steps)
            reward_after = compute_validation_reward(
                agent,
                real_env,
                episodes=eval_episodes,
                max_steps=25,
            )

            outputs[strategy]["before"].append(reward_before)
            outputs[strategy]["after"].append(reward_after)
            outputs[strategy]["delta"].append(reward_after - reward_before)

            print(
                f"[seed={seed}][{strategy}] before={reward_before:.4f} "
                f"after={reward_after:.4f} delta={reward_after - reward_before:.4f} stats={stats}"
            )

    return outputs


def summarise(results: Dict[str, Dict[str, List[float]]]) -> None:
    for strategy, values in results.items():
        before = np.array(values["before"], dtype=np.float64)
        after = np.array(values["after"], dtype=np.float64)
        delta = np.array(values["delta"], dtype=np.float64)

        def mean_sem(x: np.ndarray) -> Tuple[float, float]:
            if x.size <= 1:
                return float(x.mean() if x.size else 0.0), 0.0
            sem = float(np.std(x, ddof=1) / math.sqrt(x.size))
            return float(x.mean()), sem

        before_mean, before_sem = mean_sem(before)
        after_mean, after_sem = mean_sem(after)
        delta_mean, delta_sem = mean_sem(delta)

        print(
            f"\nStrategy: {strategy}"
            f"\n  before = {before_mean:.4f} ± {before_sem:.4f}"
            f"\n  after  = {after_mean:.4f} ± {after_sem:.4f}"
            f"\n  delta  = {delta_mean:+.4f} ± {delta_sem:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="Seeds to evaluate")
    parser.add_argument("--initial-training-steps", type=int, default=10000)
    parser.add_argument("--episodes-per-burst", type=int, default=Config.EPISODES_PER_BURST)
    parser.add_argument("--episode-max-steps", type=int, default=Config.EPISODE_MAX_STEPS)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="*",
        default=["on_policy", "uniform_replay", "success_replay", "success_mix"],
        choices=["on_policy", "uniform_replay", "success_replay", "success_mix"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Config.WORLD_MODEL_TYPE = "debug"
    Config.MAZE_MODE = getattr(Config, "MAZE_MODE", "extra_links")

    results = evaluate_strategies(
        seeds=args.seeds,
        initial_training_steps=int(args.initial_training_steps),
        episodes_per_burst=int(args.episodes_per_burst),
        episode_max_steps=int(args.episode_max_steps),
        eval_episodes=int(args.eval_episodes),
        strategies=args.strategies,
    )
    summarise(results)


if __name__ == "__main__":
    main()

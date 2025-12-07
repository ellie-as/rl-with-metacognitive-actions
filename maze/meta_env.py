import random
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

from world_model import WorldModel
from environment import MazeEnv, DreamEnv
from config import Config
from rl_valuator import *
from rl_valuator import _shortest_path_length
from rl_valuator_shapley import compute_start_goal_shapley_value, compute_start_goal_direct_impact
from utils import *
from dqn_agent import BaseDQNAgent, DQNConfig, make_dqn_agent
from seed_utils import derive_seed

import matplotlib.pyplot as plt
import logging


def _build_agent(params, env, q_params, sb3_overrides=None, seed: Optional[int] = None) -> BaseDQNAgent:
    """
    Build a temporary agent for pair valuation using the configured backend.
    """
    base_kwargs = getattr(Config, "SB3_DQN_KWARGS", None) or {}
    sb3_kwargs = dict(base_kwargs)
    if sb3_overrides:
        sb3_kwargs.update(sb3_overrides)
    if not sb3_kwargs:
        sb3_kwargs = None
    dqn_cfg = DQNConfig(
        learning_rate=float(getattr(Config, "DQN_LEARNING_RATE", 1e-3)),
        discount=float(q_params.get("discount", 0.95)),
        epsilon=float(q_params.get("epsilon", 0.2)),
        epsilon_decay=float(q_params.get("epsilon_decay", 0.995)),
        min_epsilon=float(q_params.get("min_epsilon", 0.05)),
        hidden_size=getattr(Config, "DQN_HIDDEN_SIZE", 128),
        sb3_kwargs=sb3_kwargs,
        seed=seed,
    )
    agent = make_dqn_agent(env, cfg=dqn_cfg, agent_type=getattr(Config, "AGENT_TYPE", "dqn"))
    if params is not None:
        agent.set_parameters(params)
    return agent

def _eval_pair(args):
    (pair, valuation_type, base_params, dream_env,
     mini_steps, n_eps, n_perm, sampled, q_params, coalition_size, free_cells_eval, worker_seed) = args
    overrides = {"buffer_size": 1000} if str(valuation_type).lower() == "shapley" else None
    agent = _build_agent(base_params, dream_env, q_params, sb3_overrides=overrides, seed=worker_seed)
    try:
        s, g = pair
        if valuation_type == "shapley":
            val, trajs = compute_start_goal_shapley_value(
                pair, sampled, agent, dream_env,
                mini_train_steps=mini_steps,
                num_episodes=n_eps,
                num_permutations=getattr(Config, "NUM_PERMUTATIONS", 3),
                free_cells=free_cells_eval,
            )
        elif valuation_type == "fixed_size":
            raise ValueError("'fixed_size' valuation has been removed.")
        else:
            raise ValueError(f"Unsupported valuation_type '{valuation_type}'. Use 'shapley' or 'fixed_size'.")
        print(f"Gave pair {pair} value {val}")
        return val, (trajs[0] if trajs else [])
    finally:
        try:
            env = getattr(agent, "get_env", None)
            if callable(env):
                bound_env = env()
                if hasattr(bound_env, "close"):
                    bound_env.close()
        except Exception:
            pass
        del agent
        gc.collect()

def _eval_pair_direct(args):
    (pair, valuation_type, base_params, dream_env,
     mini_steps, n_eps, n_perm, sampled, q_params, coalition_size, free_cells_eval, worker_seed) = args
    agent = _build_agent(base_params, dream_env, q_params, seed=worker_seed)
    try:
        val, trajs = compute_start_goal_direct_impact(
            pair, sampled, agent, dream_env,
            mini_train_steps=mini_steps,
            num_episodes=n_eps,
            free_cells=free_cells_eval,
        )
        print(f"Gave pair {pair} value {val} (direct impact)")
        return val, (trajs[0] if trajs else [])
    finally:
        try:
            env = getattr(agent, "get_env", None)
            if callable(env):
                bound_env = env()
                if hasattr(bound_env, "close"):
                    bound_env.close()
        except Exception:
            pass
        del agent
        gc.collect()

class MetaEnv(gym.Env):
    """
    Meta environment:
      - Action space = Discrete(3): [0=train on real episodes, 1=train on dream episodes, 2=update world model]
      - base_agent = an SR policy updated via:
           * "Train on real env": run additional Q-learning steps inside the real maze.
           * "Train on dream env": focus on selected start-goal pairs inside the DreamEnv.
      - After each meta action, the mean reward (on the validation set) is recorded.
    """
    def __init__(
                 self,
                 base_learning_rate=getattr(Config, "Q_LEARNING_RATE", 0.2),
                 evaluation_episodes=Config.EVALUATION_EPISODES,
                 reset_interval=Config.RESET_INTERVAL,
                 grid_size=Config.GRID_SIZE,
                 constant_ep_length=Config.CONSTANT_EP_LENGTH,
                 p_change=Config.P_CHANGE,
                 initial_training_steps=Config.INITIAL_TRAINING_STEPS,
                 mode=Config.MODE,
                 valuation_type=Config.VALUATION_TYPE,
                 show_plots: bool = False,
                 base_seed: Optional[int] = None):
        super(MetaEnv, self).__init__()
        self.evaluation_episodes = evaluation_episodes

        self.reset_interval = reset_interval
        self.grid_size = grid_size
        self.constant_ep_length = constant_ep_length
        self.p_change = p_change
        self.initial_training_steps = initial_training_steps
        self.mini_train_steps = Config.MINI_TRAIN_STEPS
        self.mode = mode
        self.world_model_type = getattr(Config, "WORLD_MODEL_TYPE", "nn")
        if self.world_model_type not in {"nn", "cache", "debug"}:
            raise ValueError("WORLD_MODEL_TYPE must be 'nn', 'cache', or 'debug'")
        allowed_valuations = ["shapley", "direct_impact", "approx_shapley"]
        if valuation_type not in allowed_valuations:
            raise ValueError(f"Unsupported valuation_type '{valuation_type}'. Use one of {allowed_valuations}.")
        self.valuation_type = valuation_type
        self.base_learning_rate = base_learning_rate

        self._did_initial_training = False
        self.meta_episode_index = 0
        self.full_reset_period = max(1, getattr(Config, "FULL_MAZE_RESET_PERIOD", 10))

        self.real_env = MazeEnv(grid_size=self.grid_size)
        obs_dim = self.real_env.observation_space.shape[0]
        act_dim = self.real_env.action_space.n

        # Initially, DreamEnv is instantiated without a world model and without allowed pairs.
        self.dream_env = DreamEnv(None, obs_dim, act_dim)

        epsilon = float(getattr(Config, "Q_EPSILON", 0.2))
        epsilon_decay = float(getattr(Config, "Q_EPSILON_DECAY", 0.995))
        min_epsilon = float(getattr(Config, "Q_MIN_EPSILON", 0.05))
        if getattr(Config, "FIX_EPSILON", False):
            epsilon_decay = 1.0
            min_epsilon = epsilon
        self.q_params = {
            "learning_rate": base_learning_rate,
            "discount": getattr(Config, "Q_DISCOUNT", 0.95),
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "min_epsilon": min_epsilon,
        }

        if base_seed is None:
            base_seed = getattr(Config, "SEED", None)
        self.base_seed = None if base_seed is None else int(base_seed)

        dqn_cfg = DQNConfig(
            learning_rate=float(getattr(Config, "DQN_LEARNING_RATE", 1e-3)),
            discount=float(self.q_params["discount"]),
            epsilon=float(self.q_params["epsilon"]),
            epsilon_decay=float(self.q_params["epsilon_decay"]),
            min_epsilon=float(self.q_params["min_epsilon"]),
            hidden_size=getattr(Config, "DQN_HIDDEN_SIZE", 128),
            sb3_kwargs=getattr(Config, "SB3_DQN_KWARGS", None),
            seed=self.base_seed,
        )
        self.base_agent = make_dqn_agent(
            self.real_env,
            cfg=dqn_cfg,
            agent_type=getattr(Config, "AGENT_TYPE", "dqn"),
            seed=self.base_seed,
        )

        self.world_model = None
        self.world_model_accuracy = 0.0

        self.meta_action_count = 0
        self._valuation_round = 0

        # To store a summary of meta-steps; each entry is the mean validation reward after the action.
        self.val_history = []
        self.val_success_history = []
        self.last_eval_success = 0.0

        # Store evaluation rollouts for reusing during world-model training.
        self._last_eval_transitions = []

        # Hippocampus: encoded real episodes
        self.hippocampus_episodes = []

        # MetaEnv observation = [meta_step_count, world_model_accuracy, last_action]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.show_plots = show_plots

        # Start-goal value estimator (for dream env training)
        self.sg_value_estimator = None
        self._pre_incremental_maze = None
        self._candidates_maze_snapshot = None
        self._last_candidates_maze_snapshot = None
        self.last_pair_trajectories = {}
        self._valuation_round = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.base_seed = int(seed)
        self.meta_action_count = 0
        self.world_model_accuracy = 0.0
        self.dream_env.world_model = None
        self._pre_incremental_maze = None
        self._candidates_maze_snapshot = None

        prev_maze = None if self.real_env.maze is None else self.real_env.maze.copy()
        is_full_reset = (self.meta_episode_index % self.full_reset_period == 0)
        if is_full_reset:
            print("Resetting barriers: full reset")
            self.real_env.reset_barriers(incremental=False)
            try:
                self._curr_maze = None if self.real_env.maze is None else self.real_env.maze.copy()
            except Exception:
                self._curr_maze = self.real_env.maze

            # Reinitialise the base DQN so it doesn't carry over stale weights to a new maze.
            print("Reinitialising base DQN after full reset…")
            self.base_agent = make_dqn_agent(
                self.real_env,
                cfg=DQNConfig(
                    learning_rate=float(getattr(Config, "DQN_LEARNING_RATE", 1e-3)),
                    discount=float(self.q_params["discount"]),
                    epsilon=float(self.q_params["epsilon"]),
                    epsilon_decay=float(self.q_params["epsilon_decay"]),
                    min_epsilon=float(self.q_params["min_epsilon"]),
                    hidden_size=getattr(Config, "DQN_HIDDEN_SIZE", 128),
                    sb3_kwargs=getattr(Config, "SB3_DQN_KWARGS", None),
                    seed=self.base_seed,
                ),
                agent_type=getattr(Config, "AGENT_TYPE", "dqn"),
                seed=self.base_seed,
            )

        else:
            print("Resetting barriers: incremental update")
            self._apply_incremental_barrier_update()

        # Initial training
        post_pretrain_update_needed = False
        if not self._did_initial_training:
            # First ever call: run initial training once
            if self.initial_training_steps > 0:
                print(f"Performing initial training for {self.initial_training_steps} steps…")
                self.base_agent.learn(total_timesteps=self.initial_training_steps)
            self._did_initial_training = True
            post_pretrain_update_needed = True
        elif is_full_reset and self.initial_training_steps > 0:
            # Repeat initial training on every subsequent full reset
            print(f"Repeating initial training for {self.initial_training_steps} steps after full reset…")
            self.base_agent.learn(total_timesteps=self.initial_training_steps)

        # Initialise world model only after a full reset; do not update after incremental resets
        if is_full_reset:
            try:
                print("Initialising world model after full reset…")
                self.world_model, _ = self.train_world_model(self.real_env, self.base_agent)
                self.dream_env.world_model = self.world_model
            except Exception as e:
                print(f"World model initialisation failed: {e}")

        if post_pretrain_update_needed:
            try:
                print("Applying incremental maze update after pretraining…")
                self._apply_incremental_barrier_update()
            except Exception as e:
                print(f"Post-pretraining incremental update failed: {e}")

        # Ensure hippocampus has encoded real episodes
        self._ensure_hippocampus()

        # Save snapshots for later diagnostics/plots
        self._prev_maze = prev_maze
        try:
            self._curr_maze = None if self.real_env.maze is None else self.real_env.maze.copy()
        except Exception:
            self._curr_maze = self.real_env.maze

        if prev_maze is not None:
            changed_cells = getattr(self.real_env, "last_changed_cells", [])
            try:
                self._plot_maze_transition(prev_maze, self.real_env.maze, changed_cells)
            except Exception as exc:
                print(f"Maze transition plot failed: {exc}")
        
        # Emit a zeroed “current” 3-vector
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        print(self.real_env.maze)
        self.meta_action_count += 1
        mean_reward_before, success_before, eval_rollouts = compute_validation_reward(
            self.base_agent,
            self.real_env,
            episodes=self.evaluation_episodes,
            max_steps=25,
            return_episodes=True,
            return_success=True,
        )
        self._last_eval_transitions = eval_rollouts
        self.last_eval_success = float(success_before)

        if action == 0:
            print("Meta action 0: train base_agent directly from replay buffer.")
            self._train_from_replay_buffer()

        elif action == 1:
            if self.dream_env.world_model is None:
                print("World model not yet trained => skipping dream training.")
            else:
                print("Meta action 1: training on high-value start-goal pairs in dream environment.")
                self._train_on_valued_dream_pairs(online=True)

        elif action == 2:
            if self.world_model is None:
                print("Meta action 2: train new world model.")
                self.world_model, _ = self.train_world_model(self.real_env, self.base_agent)
            else:
                print("Meta action 2: update existing world model.")
                self.world_model, _ = self.update_world_model(self.real_env, self.base_agent, self.world_model)
            self.dream_env.world_model = self.world_model

        mean_reward_after, success_after, eval_rollouts_after = compute_validation_reward(
            self.base_agent,
            self.real_env,
            episodes=self.evaluation_episodes,
            max_steps=25,
            return_episodes=True,
            return_success=True,
        )
        self._last_eval_transitions = eval_rollouts_after
        self.last_eval_success = float(success_after)

        # Record the mean validation reward after this meta action.
        self.val_history.append(mean_reward_after)
        self.val_success_history.append(self.last_eval_success)
        
        if self.constant_ep_length:
            done = (self.meta_action_count % self.reset_interval == 0)
        else:
            done = (random.random() < self.p_change)
        
        if done:
            print("Meta-episode ended, resetting barriers.")
            if self.show_plots:
                self.plot_val_history()
            meta_reward = mean_reward_after
            print(f"Reward at end of episode: {meta_reward}")
            self.meta_episode_index += 1
            # Clear hippocampus buffer between episodes to enforce buffer-only memory replay
            self.hippocampus_episodes = []
        else:
            meta_reward = 0

        clear_buffer = getattr(self.base_agent, "clear_replay_buffer", None)
        if callable(clear_buffer):
            try:
                clear_buffer()
            except Exception as exc:
                print(f"Replay buffer clear failed: {exc}")

        obs = np.array(
            [
                float(self.meta_action_count),
                self.world_model_accuracy,
                float(action),
            ],
            dtype=np.float32,
        )

        print(f"[Meta step] obs={obs.tolist()} reward={float(meta_reward)}")

        return obs, meta_reward, done, False, {}

    def plot_val_history(self):
        """
        Plot the mean validation reward recorded after each meta action.
        """
        plt.figure(figsize=(8, 4))
        plt.plot(self.val_history, marker='o', linestyle='-', color='blue')
        plt.xlabel("Meta Action Step")
        plt.ylabel("Mean Validation Reward")
        plt.title("Validation Reward over Meta Actions")
        plt.grid(True)
        plt.show(block=False)

    def _apply_incremental_barrier_update(self):
        self._pre_incremental_maze = None if self.real_env.maze is None else self.real_env.maze.copy()
        self.real_env.reset_barriers(incremental=True)
        try:
            self._curr_maze = None if self.real_env.maze is None else self.real_env.maze.copy()
        except Exception:
            self._curr_maze = self.real_env.maze

    def plot_random_rollout(self, max_steps: int = 25):
        """
        Roll out the current base agent policy on the real environment from a random
        start/goal and plot the visited path over the current maze.
        """
        try:
            original_env = self.base_agent.get_env()
        except Exception:
            original_env = None

        # Ensure we are acting in the real environment
        self.base_agent.set_env(self.real_env)

        obs, _ = self.real_env.reset()
        grid = np.array(self.real_env.maze)
        g = self.grid_size
        half = obs.shape[0] // 2
        start_idx = int(np.argmax(obs[:half]))
        goal_idx = int(np.argmax(obs[half:]))
        sr, sc = divmod(start_idx, g)
        gr, gc = divmod(goal_idx, g)

        path_rc = [(sr, sc)]
        done = False
        steps = 0
        while not done and steps < max_steps:
            act, _ = self.base_agent.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.real_env.step(int(act))
            a_idx = int(np.argmax(obs[:half]))
            r, c = divmod(a_idx, g)
            path_rc.append((r, c))
            steps += 1

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, interpolation="none")
        ax.set_title("Random rollout on real maze")
        ax.set_xlabel("col"); ax.set_ylabel("row")
        ax.set_xticks(range(g)); ax.set_yticks(range(g))
        ax.grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)

        # Overlay start/goal and path
        ax.scatter([sc], [sr], c="lime", s=80, marker="o", label="start")
        ax.scatter([gc], [gr], c="red", s=80, marker="*", label="goal")
        if len(path_rc) >= 2:
            ys = [r for r, _ in path_rc]
            xs = [c for _, c in path_rc]
            ax.plot(xs, ys, color="dodgerblue", linewidth=2, alpha=0.9, label="path")
            ax.scatter(xs, ys, color="dodgerblue", s=12)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show(block=False)

        # Restore original env binding
        if original_env is not None:
            self.base_agent.set_env(original_env)

    def _plot_maze_transition(self, old_maze, new_maze, changed_cells):
        old_maze = np.array(old_maze)
        new_maze = np.array(new_maze)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, grid, title in zip(axes, (old_maze, new_maze), ("Previous maze", "New maze")):
            im = ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, interpolation="none")
            ax.set_title(title)
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            ax.set_xticks(range(grid.shape[1]))
            ax.set_yticks(range(grid.shape[0]))
            ax.grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)
            if changed_cells:
                rows, cols = zip(*changed_cells)
                ax.scatter(cols, rows, marker='x', s=120, color='red', linewidths=2)
        fig.suptitle("Maze reset comparison")
        plt.tight_layout()
        plt.show(block=False)

    def _compute_length_weights(self, pairs, maze_arr):
        return None

    @staticmethod
    def _weighted_sample_pairs(pairs, weights, k):
        if k <= 0 or not pairs:
            return []
        return random.sample(pairs, k=min(k, len(pairs)))

    def _prepare_training_pairs(self):
        grid_size = self.grid_size

        def _maze_snapshot():
            if getattr(self, "_curr_maze", None) is not None:
                return np.array(self._curr_maze)
            maze = getattr(self.real_env, "maze", None)
            return None if maze is None else np.array(maze)

        maze_arr = _maze_snapshot()
        self.last_pair_trajectories = {}
        # Store snapshot of maze at time of candidate generation
        self._candidates_maze_snapshot = maze_arr.copy() if maze_arr is not None else None
        if self._candidates_maze_snapshot is not None:
            self._last_candidates_maze_snapshot = self._candidates_maze_snapshot
        if maze_arr is not None:
            free_start_cells = [
                r * grid_size + c
                for r in range(grid_size)
                for c in range(grid_size)
                if maze_arr[r, c] == 0
            ]
            free_goal_cells = free_start_cells[:]
        else:
            free_start_cells = list(range(grid_size * grid_size))
            free_goal_cells = free_start_cells[:]

        if len(free_start_cells) < 1:
            print("Not enough free cells to form start-goal pairs.")
            self.last_sampled_pairs = []
            self.last_pair_values = []
            self.last_all_pairs = []
            self.last_all_pair_values = []
            self.last_selected_pairs = []
            self.last_pair_trajectories = {}
            return {"top_pairs": [], "all_pairs": [], "all_pair_values": None, "valued": False}

        allow_blocked_goals = bool(getattr(Config, "INCLUDE_IMPOSSIBLE_GOALS", False))
        if allow_blocked_goals:
            goal_candidates_all = list(range(grid_size * grid_size))
        else:
            goal_candidates_all = list(free_goal_cells)
        all_pairs_full = [(s, g) for s in free_start_cells for g in goal_candidates_all]
        if not all_pairs_full:
            print("Unable to build start-goal pair set; skipping targeted training.")
            self.last_sampled_pairs = []
            self.last_pair_values = []
            self.last_all_pairs = []
            self.last_all_pair_values = []
            self.last_selected_pairs = []
            self.last_pair_trajectories = {}
            return {"top_pairs": [], "all_pairs": [], "all_pair_values": None, "valued": False}

        valuation_modes = {"top", "bottom", "top_with_mmr", "bottom_with_mmr", "direct_impact", "approx_shapley"}
        has_world_model = self.dream_env.world_model is not None
        needs_valuation = self.mode in valuation_modes and has_world_model

        if not needs_valuation:
            if self.mode in valuation_modes and not has_world_model:
                print("World model unavailable; defaulting to random start-goal selection for this action.")
            if self.mode == "longest_paths":
                return self._select_longest_path_pairs(all_pairs_full)
            top_n = min(Config.EPISODES_PER_BURST, len(all_pairs_full))
            top_pairs = random.sample(all_pairs_full, k=top_n) if top_n else []
            self.last_sampled_pairs = []
            self.last_pair_values = []
            self.last_all_pairs = list(all_pairs_full)
            self.last_all_pair_values = []
            self.last_selected_pairs = list(top_pairs)
            self.last_pair_trajectories = {}
            if self.mode == "random" or self.dream_env.world_model is None:
                print(f"Selected {len(top_pairs)} start-goal pairs uniformly at random.")
            return {
                "top_pairs": top_pairs,
                "all_pairs": all_pairs_full,
                "all_pair_values": None,
                "valued": False,
            }

        # Ensure all modes (including baselines) consider the same candidate pool:
        # every reachable start paired with every grid cell (optionally including blocked goals),
        # while still excluding starts that are walls.
        valuation_pairs = [(s, g) for s in free_start_cells for g in goal_candidates_all]

        py_random_state = random.getstate()
        np_random_state = np.random.get_state()
        try:
            if not valuation_pairs:
                print("No reachable start-goal pairs available for valuation; falling back to random selection.")
                top_n = min(Config.EPISODES_PER_BURST, len(all_pairs_full))
                if top_n:
                    if self.mode == "random" and length_weights_full is not None:
                        top_pairs = self._weighted_sample_pairs(all_pairs_full, length_weights_full, top_n)
                    else:
                        top_pairs = random.sample(all_pairs_full, k=top_n)
                else:
                    top_pairs = []
                self.last_sampled_pairs = []
                self.last_pair_values = []
                self.last_all_pairs = list(all_pairs_full)
                self.last_all_pair_values = []
                self.last_selected_pairs = list(top_pairs)
                self.last_pair_trajectories = {}
                result = {
                    "top_pairs": top_pairs,
                    "all_pairs": all_pairs_full,
                    "all_pair_values": None,
                    "valued": False,
                }
            else:
                # Valuation path: estimate values for the full candidate set (including blocked goals) and pick top-N
                num_samples = min(getattr(Config, "NUM_TO_ESTIMATE", 1000), len(valuation_pairs))
                sampled_pairs = random.sample(valuation_pairs, k=min(num_samples, len(valuation_pairs))) if num_samples else []
                self.last_sampled_pairs = list(sampled_pairs)

                print(f"Evaluating {len(sampled_pairs)} start-goal pairs in dream environment...")
                pair_values = [0.0] * len(sampled_pairs)
                pair_trajs = [None] * len(sampled_pairs)
                base_params = self.base_agent.get_parameters()
                eval_free_cells = tuple(goal_candidates_all)
                self._valuation_round += 1
                round_id = self._valuation_round
                base_seed_value = self.base_seed if self.base_seed is not None else int(getattr(Config, "SEED", 0))
                args = [
                    (
                        pair,
                        self.valuation_type,
                        base_params,
                        self.dream_env,
                        self.mini_train_steps,
                        self.evaluation_episodes,
                        Config.NUM_PERMUTATIONS,
                        sampled_pairs,
                        self.q_params.copy(),
                        Config.TOP_TRAJECTORIES,
                        eval_free_cells,
                        derive_seed(base_seed_value, "valuation", round_id, idx, tuple(pair)),
                    )
                    for idx, pair in enumerate(sampled_pairs)
                ]

                if str(self.valuation_type).lower() == "direct_impact":
                    for idx, arg in enumerate(args):
                        val, traj = _eval_pair_direct(arg)
                        pair_values[idx] = val
                        pair_trajs[idx] = traj
                        if idx == 0:
                            self.last_pair_values_extra = traj
                elif str(self.valuation_type).lower() == "approx_shapley":
                    # Compute approximate Shapley for all sampled pairs in one pass
                    from rl_valuator_shapley import compute_start_goal_shapley_value_approx_all
                    original_env = self.base_agent.get_env()
                    try:
                        values_map = compute_start_goal_shapley_value_approx_all(
                            sampled_pairs,
                            self.base_agent,
                            self.dream_env,
                            mini_train_steps=self.mini_train_steps,
                            num_episodes=self.evaluation_episodes,
                            num_permutations=getattr(Config, "NUM_PERMUTATIONS", 3),
                            free_cells=eval_free_cells,
                        )
                    finally:
                        try:
                            if original_env is not None:
                                self.base_agent.set_env(original_env)
                        except Exception:
                            pass
                    for idx, pair in enumerate(sampled_pairs):
                        v = float(values_map.get(tuple(pair), 0.0))
                        pair_values[idx] = v
                        try:
                            print(f"Gave pair {tuple(pair)} value {v}")
                        except Exception:
                            pass
                else:
                    def _eval_indexed(payload):
                        idx, arg = payload
                        val, traj = _eval_pair(arg)
                        return idx, val, traj

                    worker_cfg = getattr(Config, "VALUATION_MAX_WORKERS", None)
                    # Deterministic valuation requires serial execution.
                    max_workers = 1

                    if max_workers <= 1:
                        for idx, arg in enumerate(args):
                            val, traj = _eval_pair(arg)
                            pair_values[idx] = val
                            pair_trajs[idx] = traj
                    else:
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            for idx, val, traj in executor.map(_eval_indexed, enumerate(args)):
                                pair_values[idx] = val
                                pair_trajs[idx] = traj

                self.last_pair_values = list(pair_values)
                self.last_pair_trajectories = {
                    tuple(sampled_pairs[i]): pair_trajs[i]
                    for i in range(len(sampled_pairs))
                    if pair_trajs[i]
                }

                # Report correlation between sampled Shapley values and path length (approximate ground truth)
                if maze_arr is not None and len(sampled_pairs) == len(pair_values) and len(sampled_pairs) > 0:
                    lengths = []
                    for (start_idx, goal_idx) in sampled_pairs:
                        L = _shortest_path_length(maze_arr, self.grid_size, start_idx, goal_idx)
                        lengths.append(np.nan if L is None else float(L))
                    val_arr = np.asarray(pair_values, dtype=float)
                    len_arr = np.asarray(lengths, dtype=float)
                    mask = np.isfinite(val_arr) & np.isfinite(len_arr)
                    if mask.any():
                        corr = np.corrcoef(val_arr[mask], len_arr[mask])[0, 1]
                        try:
                            from scipy.stats import spearmanr  # optional
                            spear = spearmanr(val_arr[mask], len_arr[mask]).correlation
                            print(f"Correlation (sampled Shapley vs path length): Pearson={corr:.3f}, Spearman={spear:.3f}")
                        except Exception:
                            print(f"Correlation (sampled Shapley vs path length): Pearson={corr:.3f}")

                if self.sg_value_estimator is None:
                    estimator_model = str(getattr(Config, "SG_ESTIMATOR_MODEL", "knn")).lower()
                    if estimator_model == "nn":
                        self.sg_value_estimator = StartGoalValueEstimator(grid_size, simple=False)
                    else:
                        self.sg_value_estimator = StartGoalValueEstimator(grid_size, simple=True, model_type=estimator_model)
                    
                    # Warm-start from cached historical data if enabled
                    if getattr(Config, "SG_WARM_START", False):
                        from rl_valuator import load_warm_start_data
                        cache_dir = os.path.join(os.path.dirname(__file__), "logs/pretrain_10000/warm_start_cache")
                        warm_pairs, warm_values = load_warm_start_data(cache_dir, mode="top_with_mmr", include_stages=True)
                        if warm_pairs:
                            warm_weight = float(getattr(Config, "SG_WARM_START_WEIGHT", 0.5))
                            self.sg_value_estimator.warm_start(warm_pairs, warm_values, weight=warm_weight)

                print(pair_values[:20])
                print("Training start-goal value estimator...")
                self.sg_value_estimator.train(sampled_pairs, pair_values, epochs=Config.SG_TRAIN_EPOCHS)

                print(f"Predicting values for all {len(valuation_pairs)} possible start-goal pairs...")
                all_pair_values = self.sg_value_estimator.predict(valuation_pairs)
                self.last_all_pairs = list(valuation_pairs)
                self.last_all_pair_values = list(all_pair_values)

                if maze_arr is not None and len(all_pair_values) == len(valuation_pairs):
                    lengths = []
                    for (start_idx, goal_idx) in valuation_pairs:
                        length = _shortest_path_length(maze_arr, self.grid_size, start_idx, goal_idx)
                        lengths.append(np.nan if length is None else float(length))
                    val_arr = np.asarray(all_pair_values, dtype=float)
                    len_arr = np.asarray(lengths, dtype=float)
                    mask = np.isfinite(val_arr) & np.isfinite(len_arr)
                    if mask.any():
                        corr = np.corrcoef(val_arr[mask], len_arr[mask])[0, 1]
                        print(f"Correlation between predicted value and path length: {corr:.3f}")

                paired_data = list(zip(valuation_pairs, all_pair_values))
                paired_data.sort(key=lambda x: x[1], reverse=True)
                top_n = min(Config.EPISODES_PER_BURST, len(paired_data))

                if self.mode == "top":
                    top_pairs = [pair for pair, _ in paired_data[:top_n]]
                    print(f"Selected {len(top_pairs)} highest-value start-goal pairs for training.")
                elif self.mode == "bottom":
                    top_pairs = [pair for pair, _ in paired_data[-top_n:]]
                    print(f"Selected {len(top_pairs)} lowest-value start-goal pairs for training.")
                elif self.mode == "bottom_with_mmr":
                    lambda_param = getattr(Config, "LAMBDA_PARAM", 0.5)
                    sel_idx = mmr_select_pairs(
                        valuation_pairs,
                        all_pair_values,
                        grid_size=self.grid_size,
                        top_n=top_n,
                        lambda_param=lambda_param,
                        prefer_low=True,
                    )
                    top_pairs = [valuation_pairs[i] for i in sel_idx]
                    print(f"Selected {len(top_pairs)} start-goal pairs with low-value MMR.")
                elif self.mode in {"top_with_mmr"}:
                    lambda_param = getattr(Config, "LAMBDA_PARAM", 0.5)
                    sel_idx = mmr_select_pairs(
                        valuation_pairs,
                        all_pair_values,
                        grid_size=self.grid_size,
                        top_n=top_n,
                        lambda_param=lambda_param,
                    )
                    top_pairs = [valuation_pairs[i] for i in sel_idx]
                    print(f"Selected {len(top_pairs)} start-goal pairs with normalised MMR.")
                else:
                    raise ValueError(f"Unsupported mode '{self.mode}' for valued selection.")

                self.last_selected_pairs = list(top_pairs)
                result = {
                    "top_pairs": top_pairs,
                    "all_pairs": valuation_pairs,
                    "all_pair_values": all_pair_values,
                    "valued": True,
                }
        finally:
            random.setstate(py_random_state)
            np.random.set_state(np_random_state)

        return result

    def _select_longest_path_pairs(self, all_pairs):
        """Select start-goal pairs with the longest shortest-path distances."""
        maze_layout = getattr(self.real_env, "maze", None)
        self.last_sampled_pairs = []
        self.last_pair_values = []
        self.last_all_pairs = list(all_pairs)

        if maze_layout is None:
            print("Maze layout unavailable; falling back to random selection for longest_paths mode.")
            self.last_all_pair_values = []
            top_n = min(Config.EPISODES_PER_BURST, len(all_pairs))
            top_pairs = random.sample(all_pairs, k=top_n) if top_n else []
            self.last_selected_pairs = list(top_pairs)
            self.last_pair_trajectories = {}
            return {
                "top_pairs": top_pairs,
                "all_pairs": all_pairs,
                "all_pair_values": None,
                "valued": False,
            }

        path_lengths = []
        for start_idx, goal_idx in all_pairs:
            length = _shortest_path_length(maze_layout, self.grid_size, start_idx, goal_idx)
            path_lengths.append(length)

        self.last_all_pair_values = list(path_lengths)

        finite_pairs = [
            (pair, length)
            for pair, length in zip(all_pairs, path_lengths)
            if np.isfinite(length)
        ]

        if not finite_pairs:
            print("No reachable start-goal pairs found; falling back to random selection for longest_paths mode.")
            top_n = min(Config.TOP_TRAJECTORIES, len(all_pairs))
            top_pairs = random.sample(all_pairs, k=top_n) if top_n else []
            self.last_selected_pairs = list(top_pairs)
            self.last_pair_trajectories = {}
            return {
                "top_pairs": top_pairs,
                "all_pairs": all_pairs,
                "all_pair_values": None,
                "valued": False,
            }

        finite_pairs.sort(key=lambda item: item[1], reverse=True)
        top_n = min(Config.TOP_TRAJECTORIES, len(finite_pairs))
        selected = [pair for pair, _ in finite_pairs[:top_n]]
        print(f"Selected {len(selected)} start-goal pairs with the longest shortest-path distances.")

        self.last_selected_pairs = list(selected)
        self.last_pair_trajectories = {}
        return {
            "top_pairs": selected,
            "all_pairs": all_pairs,
            "all_pair_values": None,
            "valued": False,
        }

    def _train_from_replay_buffer(self):
        """
        Train the base agent further using encoded real episodes (hippocampus).
        """
        if not self.hippocampus_episodes:
            print("Hippocampus empty; skipping memory replay (no online collection in this action).")
            return

        print(
            f"Training from hippocampus: oversampling {len(self.hippocampus_episodes)} -> {Config.EPISODES_PER_BURST} episodes."
        )
        original_env = self.base_agent.get_env()
        self.base_agent.set_env(self.real_env)
        try:
            self.base_agent.learn_from_episodes(
                self.hippocampus_episodes,
                total_episodes=Config.EPISODES_PER_BURST,
            )
        finally:
            self.base_agent.set_env(original_env)
        # Visualize behavior after real-env training
        if self.show_plots:
            try:
                self.plot_random_rollout()
            except Exception as e:
                print(f"Random rollout plot failed: {e}")

    def _train_on_valued_dream_pairs(self, online: bool = True):
        """
        For dream env training, we:
          1. Identify free cells from the real env.
          2. Sample random start-goal pairs from these free cells.
          3. Evaluate each pair to obtain training labels for the valuation model.
          4. Fit a start-goal value estimator to predict loss decrease.
          5. Predict value for all start-goal pairs (only from free cells).
          6. Select the top pairs.
          7. Instantiate a new DreamEnv using only these allowed pairs.
          8. Train the base agent on this dream environment via its normal .learn() routine.
        """
        if self.dream_env.world_model is None:
            print("World model not available. Cannot train in dream environment.")
            return

        selection = self._prepare_training_pairs()
        top_pairs = selection["top_pairs"]
        if not top_pairs:
            print("No start-goal pairs selected; skipping dream-environment training.")
            return

        if selection["valued"]:
            try:
                from rl_valuator import plot_start_goal_statistics
                removed_square = self.real_env.last_removed_square
                changed_cells = getattr(self.real_env, "last_changed_cells", None)
                plot_start_goal_statistics(
                    selection["all_pairs"],
                    selection["all_pair_values"],
                    top_pairs,
                    self.grid_size,
                    dream_env=self.dream_env,
                    base_agent=self.base_agent,
                    removed_square=removed_square,
                    changed_cells=changed_cells,
                    prev_maze=self._prev_maze,
                    curr_maze=self._curr_maze,
                    sampled_pairs=self.last_sampled_pairs,
                    sampled_pair_values=self.last_pair_values,
                    sampled_pair_trajs=getattr(self, "last_pair_trajectories", None),
                )
            except Exception as e:
                print(f"Plotting start-goal statistics failed: {e}")

        # 8. Instantiate a new DreamEnv with the allowed (high-value) pairs, and train via .learn()
        episode_repeats = max(1, int(getattr(Config, "DREAM_EPISODE_REPEATS", 1)))
        if episode_repeats > 1:
            repeated_pairs = []
            for pair in top_pairs:
                repeated_pairs.extend([pair] * episode_repeats)
        else:
            repeated_pairs = list(top_pairs)
        total_dream_episodes = len(repeated_pairs)

        allowed_dream_env = DreamEnv(
            world_model=self.dream_env.world_model,
            obs_dim=self.real_env.observation_space.shape[0],
            action_dim=self.real_env.action_space.n,
            allowed_pairs=repeated_pairs
        )
        print(f"Training base agent in dream environment for {total_dream_episodes} episodes (repeats={episode_repeats})...")
        # Perform online learning inside dream env (world model transitions)
        original_env = self.base_agent.get_env()
        self.base_agent.set_env(allowed_dream_env)
        try:
            learn_episodes = getattr(self.base_agent, "learn_episodes", None)
            if callable(learn_episodes):
                learn_episodes(total_dream_episodes, Config.EPISODE_MAX_STEPS)
            else:
                self.base_agent.learn(total_timesteps=total_dream_episodes * Config.EPISODE_MAX_STEPS)
        finally:
            self.base_agent.set_env(original_env)
        # Visualize behavior after dream-env training (on real maze)
        if self.show_plots:
            try:
                self.plot_random_rollout()
            except Exception as e:
                print(f"Random rollout plot failed: {e}")

    def evaluate_base_agent(self):
        mean_reward, success_rate, eval_rollouts = compute_validation_reward(
            self.base_agent,
            self.real_env,
            episodes=self.evaluation_episodes,
            max_steps=25,
            return_episodes=True,
            return_success=True,
        )
        self._last_eval_transitions = eval_rollouts
        self.last_eval_success = float(success_rate)
        return mean_reward, success_rate

    def _gather_eval_transitions(self):
        if not self._last_eval_transitions:
            return None
        states = []
        actions = []
        next_states = []
        for episode in self._last_eval_transitions:
            for transition in episode:
                if len(transition) < 4:
                    continue
                obs, action, _, next_obs, *_ = transition
                states.append(np.array(obs, copy=False))
                actions.append(int(action))
                next_states.append(np.array(next_obs, copy=False))
        if not states:
            return None
        return np.array(states), np.array(actions), np.array(next_states)

    def _gather_hippocampus_transitions(self):
        if not self.hippocampus_episodes:
            self._ensure_hippocampus()
        if not self.hippocampus_episodes:
            return None
        states = []
        actions = []
        next_states = []
        for episode in self.hippocampus_episodes:
            for transition in episode:
                if len(transition) < 4:
                    continue
                obs, action, _, next_obs, *_ = transition
                states.append(np.array(obs, copy=False))
                actions.append(int(action))
                next_states.append(np.array(next_obs, copy=False))
        if not states:
            return None
        return np.array(states), np.array(actions), np.array(next_states)

    def _deterministic_next_idx(self, start_idx, action, maze_layout):
        grid = self.grid_size
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

    def _populate_debug_cache(self, world_model):
        maze_layout = getattr(self.real_env, "maze", None)
        if maze_layout is None:
            return
        grid = self.grid_size
        free_cells = [r * grid + c for r in range(grid) for c in range(grid) if maze_layout[r, c] == 0]
        if not free_cells:
            return

        action_space = self.real_env.action_space.n

        for start_idx in free_cells:
            for action in range(action_space):
                next_idx = self._deterministic_next_idx(start_idx, action, maze_layout)
                world_model.set_transition_index(start_idx, action, next_idx)

        # Also attach the maze layout to DreamEnv so resets sample only free cells
        # when using debug/cache models.
        try:
            # Ensure dream_env exists
            if getattr(self, "dream_env", None) is not None:
                self.dream_env.barriers = maze_layout.copy()
        except Exception:
            pass

    def _train_world_model(self, states, actions, next_states, world_model=None, test_transitions=None):
        obs_dim = self.real_env.observation_space.shape[0]
        action_dim = self.real_env.action_space.n
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(-1)
        actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=action_dim).float().squeeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        if world_model is None or getattr(world_model, "model_type", "nn") != self.world_model_type:
            world_model = WorldModel(obs_dim, action_dim, model_type=self.world_model_type)

        loss_fn = nn.MSELoss()

        if world_model.model_type == "cache":
            world_model.cache_transitions(states, actions.squeeze(-1), next_states)
        elif world_model.model_type == "debug":
            world_model.clear_cache()
            self._populate_debug_cache(world_model)
            world_model.cache_transitions(states, actions.squeeze(-1), next_states)
        else:
            optimizer = optim.Adam(world_model.parameters(), lr=0.001)
            for epoch in range(500):
                optimizer.zero_grad()
                predictions = world_model(states, actions_one_hot)
                loss = loss_fn(predictions, next_states)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            predictions = world_model(states, actions_one_hot)
            mse = loss_fn(predictions, next_states).item()
            accuracy = 1.0 / (1.0 + mse)
            self.world_model_accuracy = accuracy
            print(f"World Model Accuracy (train): {accuracy:.4f}")

            if test_transitions is not None:
                try:
                    test_states, test_actions, test_next_states = test_transitions
                    test_states_t = torch.tensor(np.array(test_states), dtype=torch.float32)
                    test_actions_t = torch.tensor(np.array(test_actions), dtype=torch.long).unsqueeze(-1)
                    test_actions_one_hot = torch.nn.functional.one_hot(
                        test_actions_t, num_classes=action_dim
                    ).float().squeeze(1)
                    test_next_states_t = torch.tensor(np.array(test_next_states), dtype=torch.float32)
                    test_pred = world_model(test_states_t, test_actions_one_hot)
                    test_mse = loss_fn(test_pred, test_next_states_t).item()
                    test_accuracy = 1.0 / (1.0 + test_mse)
                    print(f"World Model Accuracy (eval):  {test_accuracy:.4f}")
                except Exception as exc:
                    print(f"World model eval accuracy failed: {exc}")
        return world_model, None

    def train_world_model(self, env, base_agent):
        print("Training a new world model from hippocampus data.")
        gathered = self._gather_hippocampus_transitions()
        eval_transitions = self._gather_eval_transitions()
        if gathered is None:
            print("No hippocampus data available; falling back to random data for world model training.")
            episodes = collect_random_episodes(env, num_episodes=200, max_steps=200)
            transitions = []
            for ep in episodes:
                transitions.extend(ep)
            states, actions, rewards, next_states, dones, _ = zip(*transitions)
            gathered = (np.array(states), np.array(actions), np.array(next_states))
        states, actions, next_states = gathered
        w, _ = self._train_world_model(states, actions, next_states, test_transitions=eval_transitions)
        return w, None

    def update_world_model(self, env, base_agent, w):
        print("Updating world model with hippocampus data.")
        gathered = self._gather_hippocampus_transitions()
        eval_transitions = self._gather_eval_transitions()
        if gathered is None:
            print("No hippocampus data available; falling back to random data for world model update.")
            episodes = collect_random_episodes(env, num_episodes=200, max_steps=200)
            transitions = []
            for ep in episodes:
                transitions.extend(ep)
            states, actions, rewards, next_states, dones, _ = zip(*transitions)
            gathered = (np.array(states), np.array(actions), np.array(next_states))
        states, actions, next_states = gathered
        w, _ = self._train_world_model(states, actions, next_states, w, test_transitions=eval_transitions)
        return w, None

    def _ensure_hippocampus(self):
        target = getattr(Config, "HIPPOCAMPUS_REAL_EPISODES", 50)
        if len(self.hippocampus_episodes) >= target:
            return
        # Encode real episodes from the current real environment using the base agent policy
        remaining = target - len(self.hippocampus_episodes)
        print(f"Encoding {remaining} real episodes into hippocampus…")
        episodes = collect_episodes(self.real_env, self.base_agent, num_episodes=remaining, max_steps=Config.EPISODE_MAX_STEPS)
        self.hippocampus_episodes.extend(episodes)

    def _encode_real_episodes(self):
        self.hippocampus_episodes = []
        self._ensure_hippocampus()

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy 
import gc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    from config import Config  # when running with maze/ on PYTHONPATH
except ImportError:  # pragma: no cover
    from maze.config import Config
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from environment import DreamEnv
from scipy.stats import pearsonr
import copy
import numpy as np
import matplotlib.pyplot as plt
from environment import DreamEnv


def _cleanup_agent(agent):
    if agent is None:
        return
    try:
        if hasattr(agent, "clear_replay_buffer"):
            agent.clear_replay_buffer()
        if hasattr(agent, "set_env"):
            agent.set_env(None)
    except Exception:
        pass
    gc.collect()


def compute_validation_reward(
    agent,
    env,
    episodes=10,
    max_steps=30,
    return_episodes: bool = False,
    return_success: bool = False,
):
    """
    Run the agent in the environment for some episodes and average the reward.

    When ``return_success`` is True, also return the fraction of episodes in which
    the agent reaches the goal (i.e. receives the positive terminal reward).
    """
    total_rewards = []
    success_count = 0
    collected = [] if return_episodes else None
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        step_count = 0
        episode_success = False
        episode_transitions = [] if return_episodes else None
        while not done and step_count < max_steps:
            action, _ = agent.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, info = env.step(action)
            if return_episodes:
                episode_transitions.append(
                    (
                        np.array(obs, copy=True),
                        int(action),
                        float(reward),
                        np.array(next_obs, copy=True),
                        bool(done or truncated),
                        copy.deepcopy(info),
                    )
                )
            obs = next_obs
            done = bool(done or truncated)
            ep_reward += reward
            step_count += 1
            if done and reward > 0:
                episode_success = True
        total_rewards.append(ep_reward)
        if episode_success:
            success_count += 1
        if return_episodes and episode_transitions is not None:
            collected.append(episode_transitions)
    mean_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    success_fraction = (
        float(success_count) / len(total_rewards) if total_rewards else 0.0
    )
    if return_episodes and return_success:
        return mean_reward, success_fraction, collected
    if return_episodes:
        return mean_reward, collected
    if return_success:
        return mean_reward, success_fraction
    return mean_reward


def compute_start_goal_value(start_idx, goal_idx, temp_agent, dream_env,
                             mini_train_steps=5, num_episodes=1):
    """
    For a given start-goal pair, returns the value (improvement in validation reward)
    by training the agent on a dream environment with only this start-goal pair allowed.

    Returns:
        improvement, the temporary agent, and collected trajectories
    """
    from environment import DreamEnv

    # single-pair environment
    single_pair_env = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim,
        allowed_pairs=[(start_idx, goal_idx)]
    )
    # all-pairs environment for evaluation
    all_pairs_env = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim
    )

    # baseline performance
    baseline_val = compute_validation_reward(
        temp_agent, all_pairs_env,
        episodes=num_episodes, max_steps=25
    )

    # train on single pair
    original_env = temp_agent.get_env()
    temp_agent.set_env(single_pair_env)
    temp_agent.learn(total_timesteps=mini_train_steps)

    # record one trajectory
    trajectories = []
    obs, _ = single_pair_env.reset()
    done = False
    episode = []
    step = 0
    while not done and step < single_pair_env.max_steps:
        action, _ = temp_agent.predict(obs, deterministic=False)
        next_obs, reward, done, _, info = single_pair_env.step(action)
        episode.append((obs, action, reward, next_obs, done, info))
        obs = next_obs
        step += 1
    trajectories.append(episode)

    # evaluate after training
    new_val = compute_validation_reward(
        temp_agent, all_pairs_env,
        episodes=num_episodes, max_steps=25
    )

    # restore original env
    temp_agent.set_env(original_env)

    improvement = new_val - baseline_val
    return improvement, temp_agent, trajectories

def _shortest_path_length(maze, grid_size, start_idx, goal_idx):
    """
    Compute shortest path length on a 4-neighbour grid with barriers=1, free=0.
    Returns np.inf if goal is unreachable.
    """
    from collections import deque
    sr, sc = divmod(start_idx, grid_size)
    gr, gc = divmod(goal_idx, grid_size)
    if maze[sr, sc] == 1 or maze[gr, gc] == 1:
        return np.inf
    q = deque()
    q.append((sr, sc, 0))
    seen = {(sr, sc)}
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r, c, d = q.popleft()
        if (r, c) == (gr, gc):
            return d
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and maze[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return np.inf


def _fractional_change_in_path(prev_maze, curr_maze, grid_size, start_idx, goal_idx):
    """
    Fractional change = (L_curr - L_prev) / max(L_prev, 1)
    If either path is unreachable, returns np.nan.
    """
    if prev_maze is None or curr_maze is None:
        return np.nan
    L_prev = _shortest_path_length(prev_maze, grid_size, start_idx, goal_idx)
    L_curr = _shortest_path_length(curr_maze, grid_size, start_idx, goal_idx)
    if not np.isfinite(L_prev) or not np.isfinite(L_curr):
        return np.nan
    denom = max(L_prev, 1.0)
    return (L_curr - L_prev) / denom


def plot_start_goal_statistics(all_pairs,
                               predictions,
                               top_pairs,
                               grid_size,
                               dream_env,
                               base_agent,
                               trajectories=None,
                               removed_square=None,
                               changed_cells=None,
                               prev_maze=None,
                               curr_maze=None,
                               sampled_pairs=None,
                               sampled_pair_values=None,
                               sampled_pair_trajs=None):
    """
    1) Heatmaps of start & goal locations for top_pairs
    2) Heatmap of all visited locations (agent + goal) from one rollout per top_pair
    3) Mean predicted-value heatmap per cell
    4) Quartile breakdowns of trajectory length, counts, and removed-square fraction
    """
    predictions = np.array(predictions)
    cell_cnt    = grid_size * grid_size

    # 1) Roll out one trajectory per top_pair in a fresh DreamEnv
    top_trajectories = []
    for s, g in top_pairs:
        env = DreamEnv(
            world_model  = dream_env.world_model,
            obs_dim      = dream_env.obs_dim,
            action_dim   = dream_env.action_dim,
            allowed_pairs=[(s, g)]
        )
        agent = copy.deepcopy(base_agent)
        agent.set_env(env)

        try:
            obs, _ = env.reset()
            done   = False
            ep     = []
            force_random = False
            prev_action = None
            stall_count = 0
            stall_limit = max(5, env.max_steps // 3)
            while not done and len(ep) < env.max_steps:
                if force_random:
                    legal = np.arange(env.action_space.n)
                    if prev_action is not None and legal.size > 1:
                        legal = legal[legal != prev_action]
                    if legal.size == 0:
                        legal = np.arange(env.action_space.n)
                    a = int(np.random.choice(legal))
                    force_random = False
                else:
                    chosen, _ = agent.predict(obs, deterministic=False)
                    a = int(chosen)

                nxt, r, done, _, _ = env.step(a)
                current_idx = int(np.argmax(obs[:cell_cnt]))
                next_idx = int(np.argmax(nxt[:cell_cnt]))
                ep.append((obs, a, r, nxt, done, {}))

                if not done and next_idx == current_idx:
                    force_random = True
                    stall_count += 1
                    if stall_count >= stall_limit:
                        # Abort trajectory early to avoid pathological loops in the diagnostic plots.
                        break
                else:
                    force_random = False
                    stall_count = 0
                prev_action = a
                obs = nxt

            top_trajectories.append(ep)
        finally:
            _cleanup_agent(agent)

    # Map each (s,g) → its rollout
    traj_map = { top_pairs[i]: top_trajectories[i]
                 for i in range(len(top_pairs)) }

    sampled_traj_map = None
    if sampled_pair_trajs:
        sampled_traj_map = {}
        for pair, traj in sampled_pair_trajs.items():
            if traj:
                # ensure key uses tuple[int, int]
                if not isinstance(pair, tuple):
                    try:
                        pair = tuple(pair)
                    except TypeError:
                        continue
                sampled_traj_map[pair] = traj

    # 2) Start & goal heatmaps
    start_heat = np.zeros((grid_size, grid_size))
    goal_heat  = np.zeros((grid_size, grid_size))
    for s, g in top_pairs:
        rs, cs = divmod(s, grid_size)
        rg, cg = divmod(g, grid_size)
        start_heat[rs, cs] += 1
        goal_heat [rg, cg] += 1

    plt.figure()
    plt.imshow(start_heat, cmap="viridis", interpolation="nearest")
    plt.title("Start locations (top_pairs)")
    plt.xlabel("col"); plt.ylabel("row"); plt.colorbar()
    plt.show(block=False)

    plt.figure()
    plt.imshow(goal_heat, cmap="viridis", interpolation="nearest")
    plt.title("Goal locations (top_pairs)")
    plt.xlabel("col"); plt.ylabel("row"); plt.colorbar()
    plt.show(block=False)

    # 3) Visited-location heatmap (agent only)
    # Accurately count agent state occupancy: count start once, then count each next state once per step.
    loc_heat = np.zeros((grid_size, grid_size))

    for (s, g), path in traj_map.items():
        if path:
            # count the initial agent position from the first observation in the path
            first_obs = path[0][0]
            a_idx = int(np.argmax(first_obs[:cell_cnt]))
            r, c = divmod(a_idx, grid_size)
            loc_heat[r, c] += 1

            # count each next state once per transition
            for _, _, _, nxt, _, _ in path:
                a_idx = int(np.argmax(nxt[:cell_cnt]))
                r, c = divmod(a_idx, grid_size)
                loc_heat[r, c] += 1
        else:
            # empty path: at least seed the provided start cell
            rs, cs = divmod(s, grid_size)
            loc_heat[rs, cs] += 1

    plt.figure()
    plt.imshow(loc_heat, cmap="plasma", interpolation="nearest")
    plt.title("Visited locations (agent only)")
    plt.xlabel("col"); plt.ylabel("row"); plt.colorbar(label="hit count")
    plt.show(block=False)

    # 4) Mean predicted value per cell
    #    Use predictions for every start/goal pair so the heatmap reflects the
    #    full estimator surface instead of only the handful of pairs we rolled out.
    val_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    val_cnt = np.zeros((grid_size, grid_size), dtype=np.float32)
    for (s, g), v in zip(all_pairs, predictions):
        rs, cs = divmod(s, grid_size)
        rg, cg = divmod(g, grid_size)
        val_sum[rs, cs] += v
        val_cnt[rs, cs] += 1
        val_sum[rg, cg] += v
        val_cnt[rg, cg] += 1

    mean_val = np.full_like(val_sum, np.nan, dtype=np.float32)
    np.divide(val_sum, val_cnt, out=mean_val, where=val_cnt > 0)

    plt.figure()
    finite_vals = mean_val[np.isfinite(mean_val)]
    if finite_vals.size:
        min_val, max_val = float(finite_vals.min()), float(finite_vals.max())
        if min_val < 0 < max_val:
            norm = TwoSlopeNorm(vcenter=0, vmin=min_val, vmax=max_val)
            heatmap = plt.imshow(mean_val, cmap="RdYlGn", norm=norm, interpolation="nearest")
        else:
            cmap = "viridis" if min_val >= 0 else "magma"
            heatmap = plt.imshow(mean_val, cmap=cmap, interpolation="nearest")
    else:
        heatmap = plt.imshow(mean_val, cmap="RdYlGn", interpolation="nearest")
    if changed_cells:
        unique_cells = {(int(r), int(c)) for r, c in changed_cells}
        if unique_cells:
            rows, cols = zip(*unique_cells)
            plt.scatter(cols, rows, marker='x', s=200, color='black', linewidths=2)
    plt.title("Mean predicted value by location")
    plt.xlabel("col"); plt.ylabel("row"); plt.colorbar(heatmap, label="value")
    plt.show(block=False)

    # 3b) Visualise the three highest and three lowest value start-goal pairs
    try:
            focus_pairs = []

            if sampled_pairs and sampled_pair_values and sampled_traj_map:
                spairs = [tuple(p) for p in sampled_pairs]
                paired_vals = []
                for pair, value in zip(spairs, sampled_pair_values):
                    traj = sampled_traj_map.get(pair)
                    if traj:
                        paired_vals.append((pair, value, traj))
                if paired_vals:
                    paired_vals.sort(key=lambda x: x[1], reverse=True)
                    top_k = paired_vals[:3]
                    bottom_k = paired_vals[-3:][::-1]
                    for pair, value, traj in top_k:
                        focus_pairs.append(("highest", pair, value, traj))
                    for pair, value, traj in bottom_k:
                        focus_pairs.append(("lowest", pair, value, traj))

            if not focus_pairs:
                paired = list(zip(all_pairs, predictions))
                if paired:
                    paired.sort(key=lambda x: x[1], reverse=True)
                    top_k = paired[:3]
                    bottom_k = paired[-3:][::-1]  # ensure ascending order for display
                    focus_pairs = [("highest", pair, value, traj_map.get(pair))
                                   for pair, value in top_k]
                    focus_pairs += [("lowest", pair, value, traj_map.get(pair))
                                    for pair, value in bottom_k]

            def _rollout_pair(pair):
                s, g = pair
                env = DreamEnv(
                    world_model=dream_env.world_model,
                    obs_dim=dream_env.obs_dim,
                    action_dim=dream_env.action_dim,
                    allowed_pairs=[(s, g)]
                )
                agent = copy.deepcopy(base_agent)
                agent.set_env(env)
                try:
                    obs, _ = env.reset()
                    path = []
                    done = False
                    steps = 0
                    while not done and steps < env.max_steps:
                        act, _ = agent.predict(obs, deterministic=False)
                        nxt, r, done, _, _ = env.step(int(act))
                        cell_idx = int(np.argmax(obs[:cell_cnt]))
                        path.append(cell_idx)
                        obs = nxt
                        steps += 1
                    # append final state
                    cell_idx = int(np.argmax(obs[:cell_cnt]))
                    path.append(cell_idx)
                    return path
                finally:
                    _cleanup_agent(agent)

            paths_to_plot = []
            for label, pair, value, cached_traj in focus_pairs:
                indices = None

                # Prefer actual valuation trajectory if provided.
                if cached_traj:
                    indices = []
                    first_obs = cached_traj[0][0]
                    indices.append(int(np.argmax(first_obs[:cell_cnt])))
                    for _, _, _, nxt, _, _ in cached_traj:
                        indices.append(int(np.argmax(nxt[:cell_cnt])))

                if not indices:
                    try:
                        indices = _rollout_pair(pair)
                    except Exception:
                        indices = None

                if not indices:
                    fallback = traj_map.get(pair)
                    if fallback:
                        indices = []
                        first_obs = fallback[0][0]
                        indices.append(int(np.argmax(first_obs[:cell_cnt])))
                        for _, _, _, nxt, _, _ in fallback:
                            indices.append(int(np.argmax(nxt[:cell_cnt])))
                    else:
                        indices = []

                paths_to_plot.append((label, pair, value, indices))

            if paths_to_plot:
                ncols = 3
                nrows = 2
                fig, axes = plt.subplots(nrows, ncols, figsize=(12, 7))
                axes = axes.flatten()
                for ax, (label, pair, value, indices) in zip(axes, paths_to_plot):
                    grid = np.zeros((grid_size, grid_size))
                    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
                    start_row, start_col = divmod(pair[0], grid_size)
                    goal_row, goal_col = divmod(pair[1], grid_size)
                    if indices:
                        coords = [divmod(idx, grid_size) for idx in indices]
                        ys = [r for r, _ in coords]
                        xs = [c for _, c in coords]
                        ax.plot(xs, ys, marker="o", color="dodgerblue", linewidth=2, alpha=0.85)
                    ax.scatter(start_col, start_row, c="lime", s=80, marker="o", label="start")
                    ax.scatter(goal_col, goal_row, c="red", s=80, marker="*", label="goal")
                    ax.set_title(f"{label.title()} value (v={value:.3f})\ns={pair[0]}, g={pair[1]}")
                    ax.set_xticks(range(grid_size))
                    ax.set_yticks(range(grid_size))
                    ax.set_xlabel("col")
                    ax.set_ylabel("row")
                    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
                    ax.invert_yaxis()
                # hide any unused subplots
                for ax in axes[len(paths_to_plot):]:
                    ax.axis("off")
                handles, labels = axes[0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="upper right")
                fig.suptitle("Example trajectories: highest vs lowest valued pairs")
                plt.tight_layout()
                plt.show(block=False)
    except Exception as exc:
        print(f"Failed to plot top/bottom valued paths: {exc}")

    # 6) New: Scatter of predicted pair value vs. Jaccard distance of optimal-path cells
    if prev_maze is not None and curr_maze is not None:
        def _optimal_path_cells(maze, grid_size, start_idx, goal_idx):
            from collections import deque
            sr, sc = divmod(start_idx, grid_size)
            gr, gc = divmod(goal_idx, grid_size)
            if maze[sr, sc] == 1 or maze[gr, gc] == 1:
                return None
            q = deque([(sr, sc)])
            parents = { (sr, sc): None }
            dirs = [(-1,0),(1,0),(0,-1),(0,1)]
            while q:
                r, c = q.popleft()
                if (r, c) == (gr, gc):
                    # reconstruct
                    path = []
                    cur = (r, c)
                    while cur is not None:
                        path.append(cur)
                        cur = parents[cur]
                    return set(path)
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size and maze[nr, nc] == 0:
                        if (nr, nc) not in parents:
                            parents[(nr, nc)] = (r, c)
                            q.append((nr, nc))
            return None

        def _jaccard_distance_path_cells(prev_maze, curr_maze, grid_size, start_idx, goal_idx):
            a = _optimal_path_cells(prev_maze, grid_size, start_idx, goal_idx)
            b = _optimal_path_cells(curr_maze, grid_size, start_idx, goal_idx)
            if a is None or b is None:
                return np.nan
            union = a | b
            if not union:
                return 0.0
            inter = a & b
            iou = len(inter) / len(union)
            return 1.0 - iou

        jacc_dists = []
        vals = []
        for (s, g), v in zip(all_pairs, predictions):
            d = _jaccard_distance_path_cells(prev_maze, curr_maze, grid_size, s, g)
            if np.isfinite(d):
                jacc_dists.append(d)
                vals.append(v)
        if len(jacc_dists) >= 3:
            try:
                r, p = pearsonr(jacc_dists, vals)
            except Exception:
                r, p = np.nan, np.nan
            plt.figure()
            plt.scatter(jacc_dists, vals, alpha=0.6)
            plt.axhline(0, color='gray', lw=1, alpha=0.5)
            plt.xlabel("Jaccard distance between optimal path cells (prev vs curr)")
            plt.ylabel("Estimated pair value")
            title = "Value vs. path-change (Jaccard)"
            if np.isfinite(r):
                title += f" (r={r:.3f}, p={p:.3g})"
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.show(block=False)

    # 5) Quartile statistics
    quart_bounds = np.percentile(predictions, [0,25,50,75,100])
    labels       = [f"{i*25}-{(i+1)*25}%" for i in range(4)]
    mean_lens, counts, removed_frac = [], [], []

    def traj_len(pair):
        path = traj_map.get(pair, [])
        if path:
            return len(path)
        s, g = pair
        rs, cs = divmod(s, grid_size)
        rg, cg = divmod(g, grid_size)
        return abs(rs-rs) + abs(cs-cg)

    for q in range(4):
        lo, hi = quart_bounds[q], quart_bounds[q+1]
        if q < 3:
            idxs = [i for i,p in enumerate(predictions) if lo <= p < hi]
        else:
            idxs = [i for i,p in enumerate(predictions) if lo <= p <= hi]
        counts.append(len(idxs))
        mean_lens.append(np.mean([traj_len(all_pairs[i]) for i in idxs]) if idxs else 0)

        if removed_square is not None:
            hit, tot = 0, 0
            for i in idxs:
                for obs, _, _, _, _, _ in traj_map.get(all_pairs[i], []):
                    a_idx = np.argmax(obs[:cell_cnt])
                    if divmod(a_idx, grid_size) == removed_square:
                        hit += 1; break
                tot += 1
            removed_frac.append(hit/tot if tot else 0)

    plt.figure(figsize=(8,3))
    plt.bar(labels, mean_lens); plt.title("Mean trajectory length by quartile"); plt.show(block=False)

    plt.figure(figsize=(8,3))
    plt.bar(labels, counts);      plt.title("Count by quartile");          plt.show(block=False)

    # if removed_square is not None:
    #     plt.figure(figsize=(8,3))
    #     plt.bar(labels, removed_frac)
    #     plt.title("Fraction through removed barrier")
    #     plt.ylim(0,1)
    #     plt.show(block=False)


def load_warm_start_data(cache_dir: str, mode: str = "top_with_mmr", include_stages: bool = True):
    """
    Load historical Shapley data from cache for warm-starting the value estimator.
    
    Args:
        cache_dir: Path to valuation cache directory
        mode: Which mode files to load (e.g., "top_with_mmr")
        include_stages: If True, also load stage files for more data
    
    Returns:
        Tuple of (pairs, z_scored_values) or ([], []) if no data found
    """
    import glob
    import re
    import json
    
    all_pairs = []
    all_values = []
    
    if not os.path.isdir(cache_dir):
        return [], []
    
    # Find files
    if include_stages:
        pattern = os.path.join(cache_dir, f"seed_*_{mode}*.json")
    else:
        pattern = os.path.join(cache_dir, f"seed_*_{mode}.json")
    
    files = glob.glob(pattern)
    if not files:
        return [], []
    
    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            sampled = data.get("sampled_pairs", [])
            values = data.get("sampled_pair_values", [])
            
            if sampled and values and len(sampled) == len(values):
                values_arr = np.array(values, dtype=float)
                
                # Robust normalize within this file (median + MAD)
                if len(values_arr) > 1:
                    median_v = np.median(values_arr)
                    mad_v = np.median(np.abs(values_arr - median_v))
                    # Scale MAD to be comparable to std (for normal dist, MAD ≈ 0.6745 * std)
                    mad_scaled = mad_v * 1.4826 if mad_v > 1e-8 else 1.0
                    if mad_scaled > 1e-8:
                        values_arr = (values_arr - median_v) / mad_scaled
                    else:
                        values_arr = values_arr - median_v
                
                all_pairs.extend([tuple(p) for p in sampled])
                all_values.extend(values_arr.tolist())
        except Exception:
            continue
    
    return all_pairs, all_values


class StartGoalValueEstimator:
    """
    A model to predict the value (loss decrease) for any start-goal pair.
    If simple=True, uses either a linear regression or a KNN over handcrafted spatial features.
    Otherwise, uses a small PyTorch network on one-hot encoding of start/goal.
    """
    def __init__(self, grid_size, simple=True, model_type: str = "knn"):
        self.grid_size = grid_size
        self.simple = simple
        self.model_type = str(model_type).lower() if simple else None
        self.scaler = None
        self.history_pairs = []
        self.history_values = []
        self.history_weights = []
        self._warm_started = False
        if self.simple:
            if self.model_type == "linear":
                self.model = LinearRegression()
            elif self.model_type in {"forest", "forest_with_history"}:
                n_estimators = int(getattr(Config, "SG_FOREST_TREES", 300))
                random_state = getattr(Config, "SEED", None)
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                )
                if self.model_type != "forest_with_history":
                    self.history_pairs = []
                    self.history_values = []
                    self.history_weights = []
            else:
                # Default to KNN with a small neighborhood to capture local variations.
                self.model = KNeighborsRegressor(n_neighbors=3, weights="distance")
                self.scaler = StandardScaler()
        else:
            # simple linear layer on the same 6-D handcrafted feature vector used by the simple estimators
            input_dim = 6  # [cs, rs, cg, rg, dx, dy]
            self.model = torch.nn.Linear(input_dim, 1)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = torch.nn.MSELoss()

    def warm_start(self, pairs, values, weight: float = 0.5):
        """
        Warm-start the estimator with historical data.
        
        Args:
            pairs: List of (start, goal) tuples
            values: List of z-scored values (should be normalized per-source before calling)
            weight: Weight to assign to warm-start data (default 0.5)
        """
        if not pairs or not values:
            return
        
        if self.model_type not in {"forest", "forest_with_history"}:
            print(f"Warm-start only supported for forest models, not {self.model_type}")
            return
        
        self.history_pairs.extend([tuple(p) for p in pairs])
        self.history_values.extend([float(v) for v in values])
        self.history_weights.extend([weight] * len(pairs))
        self._warm_started = True
        
        print(f"Warm-started estimator with {len(pairs)} historical pairs (weight={weight})")
    
    def _pairs_to_features(self, start_goal_pairs):
        feats = []
        for s, g in start_goal_pairs:
            s = int(s)
            g = int(g)
            rs, cs = divmod(s, self.grid_size)
            rg, cg = divmod(g, self.grid_size)
            dx = abs(cs - cg)
            dy = abs(rs - rg)
            feats.append([cs, rs, cg, rg, dx, dy])
        return np.array(feats, dtype=float)

    def train(self, start_goal_pairs, values, epochs=200):
        if not start_goal_pairs:
            return
        if self.simple:
            X_new = self._pairs_to_features(start_goal_pairs)
            y_new = np.array(values, dtype=float)

            if self.model_type == "linear":
                self.model.fit(X_new, y_new)
                X_used = X_new
            elif self.model_type in {"forest", "forest_with_history"}:
                X_train = X_new
                y_train = y_new
                w_hist = None
                if self.model_type == "forest_with_history" and self.history_pairs:
                    X_hist = self._pairs_to_features(self.history_pairs)
                    y_hist = np.array(self.history_values, dtype=float)
                    w_hist = np.array(self.history_weights, dtype=float) if self.history_weights else np.ones(len(y_hist), dtype=float)
                    if w_hist.size != len(y_hist):
                        w_hist = np.ones(len(y_hist), dtype=float)
                    X_train = np.vstack([X_hist, X_new])
                    y_train = np.concatenate([y_hist, y_new])
                fit_kwargs = {}
                weight_new = 1.0
                if self.model_type == "forest_with_history":
                    total_prev = float(w_hist.sum()) if (w_hist is not None and w_hist.size) else 0.0
                    if len(y_new) > 0:
                        min_new_frac = float(getattr(Config, "SG_HISTORY_MIN_NEW_FRACTION", 0.25))
                        min_new_frac = max(0.0, min(min_new_frac, 0.9))
                        if min_new_frac > 0.0 and total_prev > 0.0:
                            required_new_total = (min_new_frac / max(1e-8, 1.0 - min_new_frac)) * total_prev
                        else:
                            required_new_total = 0.0
                        weight_new = max(1.0, required_new_total / float(len(y_new)))
                        w_new = np.full(len(y_new), weight_new, dtype=float)
                    else:
                        w_new = np.array([], dtype=float)
                    if w_hist is not None and w_hist.size:
                        sample_weight = np.concatenate([w_hist, w_new]) if w_new.size else w_hist
                    else:
                        sample_weight = w_new if w_new.size else None
                    if sample_weight is not None and (not isinstance(sample_weight, np.ndarray) or sample_weight.size):
                        fit_kwargs["sample_weight"] = sample_weight
                self.model.fit(X_train, y_train, **fit_kwargs)
                X_used = X_train
            else:
                X_scaled = self.scaler.fit_transform(X_new)
                self.model.fit(X_scaled, y_new)
                X_used = X_scaled

            if self.model_type == "forest_with_history" and start_goal_pairs:
                weight_new_value = float(weight_new)
                self.history_pairs.extend([tuple(p) for p in start_goal_pairs])
                self.history_values.extend([float(v) for v in values])
                self.history_weights.extend([weight_new_value] * len(start_goal_pairs))

            try:
                r2 = float(self.model.score(X_used, y_train if 'y_train' in locals() else y_new))
                label = self.model_type.upper() if self.model_type else "MODEL"
                print(f"Start-goal {label} model R^2: {r2:.3f}")
            except Exception as exc:
                print(f"Failed to compute R^2 for start-goal model: {exc}")
        else:
            # original one-hot approach
            X = torch.FloatTensor(self._pairs_to_features(start_goal_pairs))
            Y = torch.FloatTensor(values).unsqueeze(1)

            for epoch in range(epochs):
                self.optimizer.zero_grad()
                out = self.model(X)
                loss = self.loss_fn(out, Y)
                loss.backward()
                self.optimizer.step()
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    def predict(self, start_goal_pairs):
        if self.simple:
            X = self._pairs_to_features(start_goal_pairs)
            if self.model_type == "linear" or self.model_type in {"forest", "forest_with_history"}:
                return self.model.predict(X)
            X_used = self.scaler.transform(X) if self.scaler is not None else X
            return self.model.predict(X_used)
        else:
            X = torch.FloatTensor(self._pairs_to_features(start_goal_pairs))
            with torch.no_grad():
                out = self.model(X)
            return out.numpy().flatten()

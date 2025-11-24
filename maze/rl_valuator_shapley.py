import random
import numpy as np
import torch
import copy
import gc
from environment import DreamEnv
from rl_valuator import compute_validation_reward


def compute_start_goal_shapley_value(
    target_pair,
    candidate_pairs,
    base_agent,
    dream_env,
    mini_train_steps=5,
    num_episodes=1,
    num_permutations=5,
    free_cells=None,
):
    """
    Approximate the Shapley value for a given start-goal pair (target_pair) with respect 
    to a set of candidate start-goal pairs.

    For each of num_permutations random orderings of candidate_pairs, we:
      1) Train on the prefix alone (without target_pair)
      2) Train on the prefix plus target_pair
      3) Take the difference in validation reward as the marginal contribution
    Return the average marginal contribution and one sample trajectory.
    """
    if target_pair not in candidate_pairs:
        raise ValueError("target_pair must be in candidate_pairs!")

    # Build a shared evaluation environment so both "with" and "without" agents
    # are scored on the same distribution of start-goal pairs.
    grid_size = dream_env.grid_size
    if free_cells:
        free_cells = [int(c) for c in free_cells]
        all_eval_pairs = [(s, g)
                          for s in free_cells
                          for g in free_cells
                          if s != g]
    else:
        total_cells = grid_size * grid_size
        all_eval_pairs = [(s, g)
                          for s in range(total_cells)
                          for g in range(total_cells)
                          if s != g]
    eval_count = min(len(all_eval_pairs), 100)
    eval_pairs = random.sample(all_eval_pairs, eval_count) if eval_count else []
    eval_env = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim,
        allowed_pairs=eval_pairs if eval_pairs else None
    )

    shapley_values = []
    for _ in range(num_permutations):
        # 1) sample a random permutation and split at target_pair
        perm = candidate_pairs.copy()
        random.shuffle(perm)
        prefix = []
        for p in perm:
            if p == target_pair:
                break
            prefix.append(p)

        # 2) build two dream environments
        env_with = DreamEnv(
            world_model=dream_env.world_model,
            obs_dim=dream_env.obs_dim,
            action_dim=dream_env.action_dim,
            allowed_pairs=prefix + [target_pair]
        )
        env_without = None
        if prefix:
            env_without = DreamEnv(
                world_model=dream_env.world_model,
                obs_dim=dream_env.obs_dim,
                action_dim=dream_env.action_dim,
                allowed_pairs=prefix
            )

        # 3) train & evaluate without target_pair
        agent_without = None
        agent_with = None
        perf_without = 0.0
        perf_with = 0.0
        try:
            agent_without = copy.deepcopy(base_agent)
            if env_without is not None:
                agent_without.set_env(env_without)
                agent_without.learn(total_timesteps=mini_train_steps)
            agent_without.set_env(eval_env)
            perf_without = compute_validation_reward(
                agent_without,
                eval_env,
                episodes=num_episodes,
                max_steps=25
            )

            # 4) train & evaluate with target_pair
            agent_with = copy.deepcopy(base_agent)
            agent_with.set_env(env_with)
            agent_with.learn(total_timesteps=mini_train_steps)
            agent_with.set_env(eval_env)
            perf_with = compute_validation_reward(
                agent_with,
                eval_env,
                episodes=num_episodes,
                max_steps=25
            )
        finally:
            for agent_ref in (agent_without, agent_with):
                if agent_ref is None:
                    continue
                try:
                    if hasattr(agent_ref, "clear_replay_buffer"):
                        agent_ref.clear_replay_buffer()
                    agent_ref.set_env(None)
                except Exception:
                    pass
            agent_without = None
            agent_with = None
            gc.collect()

        # 5) record marginal contribution
        shapley_values.append(perf_with - perf_without)

    # Collect one sample trajectory for analysis
    single_pair_env = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim,
        allowed_pairs=[target_pair]
    )
    trajectories = []
    temp_agent = None
    try:
        temp_agent = copy.deepcopy(base_agent)
        temp_agent.set_env(single_pair_env)

        obs, _ = single_pair_env.reset()
        done = False
        step_count = 0
        episode = []
        while not done and step_count < single_pair_env.max_steps:
            action, _ = temp_agent.predict(obs, deterministic=False)
            next_obs, reward, done, _, info = single_pair_env.step(action)
            episode.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            step_count += 1
        trajectories.append(episode)
    finally:
        if temp_agent is not None:
            try:
                if hasattr(temp_agent, "clear_replay_buffer"):
                    temp_agent.clear_replay_buffer()
                temp_agent.set_env(None)
            except Exception:
                pass
        temp_agent = None
        gc.collect()

    # Return mean Shapley and the trajectory
    return np.mean(shapley_values), trajectories


def compute_start_goal_shapley_value_approx_all(
    candidate_pairs,
    base_agent,
    dream_env,
    mini_train_steps=5,
    num_episodes=1,
    num_permutations=5,
    free_cells=None,
):
    """
    Approximate Shapley values for all candidate_pairs at once.

    For each permutation, start from a snapshot of the base agent, then iteratively
    add one pair to the training set, continue training, and log the delta in
    validation reward. The marginal for the just-added pair is the delta from
    the previous evaluation. Averaging deltas across permutations yields an
    efficient approximation to Shapley values with dramatically fewer resets.
    """
    if not candidate_pairs:
        return {}

    # Build shared evaluation environment
    grid_size = dream_env.grid_size
    if free_cells:
        free_cells = [int(c) for c in free_cells]
        all_eval_pairs = [(s, g) for s in free_cells for g in free_cells if s != g]
    else:
        total_cells = grid_size * grid_size
        all_eval_pairs = [(s, g) for s in range(total_cells) for g in range(total_cells) if s != g]
    eval_count = min(len(all_eval_pairs), 100)
    eval_pairs = random.sample(all_eval_pairs, eval_count) if eval_count else []
    eval_env = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim,
        allowed_pairs=eval_pairs if eval_pairs else None,
    )

    # Prepare accumulator
    contrib_sum = {tuple(p): 0.0 for p in candidate_pairs}

    base_params = None
    if hasattr(base_agent, "get_parameters"):
        base_params = base_agent.get_parameters()

    for perm_idx in range(int(num_permutations)):
        perm = candidate_pairs.copy()
        random.shuffle(perm)
        try:
            print(f"Approx Shapley permutation {perm_idx + 1}/{int(num_permutations)}", flush=True)
        except Exception:
            pass

        # Reset helper to snapshot
        if base_params is not None and hasattr(base_agent, "set_parameters"):
            base_agent.set_parameters(base_params)
        if hasattr(base_agent, "clear_replay_buffer"):
            base_agent.clear_replay_buffer()

        # Baseline performance (no training on specific pairs yet)
        base_agent.set_env(eval_env)
        prev_perf = compute_validation_reward(
            base_agent, eval_env, episodes=num_episodes, max_steps=25
        )

        for pair in perm:
            # Train only on the newly added pair (new_pair_only)
            env_with = DreamEnv(
                world_model=dream_env.world_model,
                obs_dim=dream_env.obs_dim,
                action_dim=dream_env.action_dim,
                allowed_pairs=[pair],
            )
            base_agent.set_env(env_with)
            base_agent.learn(total_timesteps=mini_train_steps)
            base_agent.set_env(eval_env)
            curr_perf = compute_validation_reward(
                base_agent, eval_env, episodes=num_episodes, max_steps=25
            )
            delta = curr_perf - prev_perf
            contrib_sum[tuple(pair)] += delta
            if perm_idx == 0:
                try:
                    print(f"Approx Shapley pass: pair {tuple(pair)} delta {delta}", flush=True)
                except Exception:
                    pass
            prev_perf = curr_perf

    try:
        print("Approx Shapley accumulation complete", flush=True)
    except Exception:
        pass

    # Average across permutations
    nperm = float(max(1, int(num_permutations)))
    result = {k: (v / nperm) for k, v in contrib_sum.items()}

    # Restore helper to base state and detach env
    try:
        if base_params is not None and hasattr(base_agent, "set_parameters"):
            base_agent.set_parameters(base_params)
        if hasattr(base_agent, "clear_replay_buffer"):
            base_agent.clear_replay_buffer()
        base_agent.set_env(None)
    except Exception:
        pass
    gc.collect()
    return result

def compute_start_goal_direct_impact(
    target_pair,
    candidate_pairs,
    base_agent,
    dream_env,
    mini_train_steps=5,
    num_episodes=1,
    free_cells=None,
):
    """
    Direct impact baseline: train agent on target_pair alone and measure validation improvement.
    """
    grid_size = dream_env.grid_size
    if free_cells:
        free_cells = [int(c) for c in free_cells]
        all_eval_pairs = [(s, g) for s in free_cells for g in free_cells if s != g]
    else:
        total_cells = grid_size * grid_size
        all_eval_pairs = [(s, g) for s in range(total_cells) for g in range(total_cells) if s != g]
    eval_count = min(len(all_eval_pairs), 100)
    eval_pairs = random.sample(all_eval_pairs, eval_count) if eval_count else []
    eval_env = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim,
        allowed_pairs=eval_pairs if eval_pairs else None
    )

    single_pair_env = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim,
        allowed_pairs=[target_pair]
    )

    agent_raw = None
    agent_trained = None
    perf_before = 0.0
    perf_after = 0.0
    try:
        agent_raw = copy.deepcopy(base_agent)
        agent_raw.set_env(eval_env)
        perf_before = compute_validation_reward(
            agent_raw,
            eval_env,
            episodes=num_episodes,
            max_steps=25
        )

        agent_trained = copy.deepcopy(base_agent)
        agent_trained.set_env(single_pair_env)
        agent_trained.learn(total_timesteps=mini_train_steps)
        agent_trained.set_env(eval_env)
        perf_after = compute_validation_reward(
            agent_trained,
            eval_env,
            episodes=num_episodes,
            max_steps=25
        )
    finally:
        for agent_ref in (agent_raw, agent_trained):
            if agent_ref is None:
                continue
            try:
                if hasattr(agent_ref, "clear_replay_buffer"):
                    agent_ref.clear_replay_buffer()
                agent_ref.set_env(None)
            except Exception:
                pass
        agent_raw = None
        agent_trained = None
        gc.collect()

    trajectories = []
    temp_agent = None
    try:
        temp_agent = copy.deepcopy(base_agent)
        temp_agent.set_env(single_pair_env)
        obs, _ = single_pair_env.reset()
        done = False
        step_count = 0
        episode = []
        while not done and step_count < single_pair_env.max_steps:
            action, _ = temp_agent.predict(obs, deterministic=False)
            next_obs, reward, done, _, info = single_pair_env.step(action)
            episode.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            step_count += 1
        trajectories.append(episode)
    finally:
        if temp_agent is not None:
            try:
                if hasattr(temp_agent, "clear_replay_buffer"):
                    temp_agent.clear_replay_buffer()
                temp_agent.set_env(None)
            except Exception:
                pass
        temp_agent = None
        gc.collect()

    return perf_after - perf_before, trajectories

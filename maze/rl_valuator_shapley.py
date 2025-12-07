import random
import numpy as np
import torch
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

    # Snapshot base agent parameters and original environment so we can
    # “clone” by resetting weights instead of using deepcopy (which may
    # fail for non-picklable SB3 internals).
    base_params = None
    if hasattr(base_agent, "get_parameters"):
        base_params = base_agent.get_parameters()
    original_env = None
    if hasattr(base_agent, "get_env"):
        try:
            original_env = base_agent.get_env()
        except Exception:
            original_env = None

    shapley_values = []
    try:
        for _ in range(num_permutations):
            # 1) sample a random permutation and split at target_pair
            perm = candidate_pairs.copy()
            random.shuffle(perm)
            prefix = []
            for p in perm:
                if p == target_pair:
                    break
                prefix.append(p)

            # 2) build dream environments for with / without target_pair
            env_with = DreamEnv(
                world_model=dream_env.world_model,
                obs_dim=dream_env.obs_dim,
                action_dim=dream_env.action_dim,
                allowed_pairs=prefix + [target_pair],
            )
            env_without = None
            if prefix:
                env_without = DreamEnv(
                    world_model=dream_env.world_model,
                    obs_dim=dream_env.obs_dim,
                    action_dim=dream_env.action_dim,
                    allowed_pairs=prefix,
                )

            # 3) train & evaluate without target_pair
            perf_without = 0.0
            if base_params is not None and hasattr(base_agent, "set_parameters"):
                base_agent.set_parameters(base_params)
            if hasattr(base_agent, "clear_replay_buffer"):
                try:
                    base_agent.clear_replay_buffer()
                except Exception:
                    pass
            if env_without is not None:
                base_agent.set_env(env_without)
                base_agent.learn(total_timesteps=mini_train_steps)
            base_agent.set_env(eval_env)
            perf_without = compute_validation_reward(
                base_agent,
                eval_env,
                episodes=num_episodes,
                max_steps=25,
            )

            # 4) train & evaluate with target_pair
            if base_params is not None and hasattr(base_agent, "set_parameters"):
                base_agent.set_parameters(base_params)
            if hasattr(base_agent, "clear_replay_buffer"):
                try:
                    base_agent.clear_replay_buffer()
                except Exception:
                    pass
            base_agent.set_env(env_with)
            base_agent.learn(total_timesteps=mini_train_steps)
            base_agent.set_env(eval_env)
            perf_with = compute_validation_reward(
                base_agent,
                eval_env,
                episodes=num_episodes,
                max_steps=25,
            )

            # 5) record marginal contribution
            shapley_values.append(perf_with - perf_without)

        # Collect one sample trajectory for analysis
        single_pair_env = DreamEnv(
            world_model=dream_env.world_model,
            obs_dim=dream_env.obs_dim,
            action_dim=dream_env.action_dim,
            allowed_pairs=[target_pair],
        )
        trajectories = []

        # Roll out using the base agent with reset parameters
        if base_params is not None and hasattr(base_agent, "set_parameters"):
            base_agent.set_parameters(base_params)
        if hasattr(base_agent, "clear_replay_buffer"):
            try:
                base_agent.clear_replay_buffer()
            except Exception:
                pass
        base_agent.set_env(single_pair_env)

        obs, _ = single_pair_env.reset()
        done = False
        step_count = 0
        episode = []
        while not done and step_count < single_pair_env.max_steps:
            action, _ = base_agent.predict(obs, deterministic=False)
            next_obs, reward, done, _, info = single_pair_env.step(action)
            episode.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            step_count += 1
        trajectories.append(episode)

        # Return mean Shapley and the trajectory
        return np.mean(shapley_values), trajectories
    finally:
        # Restore base agent to its original state as best we can.
        if base_params is not None and hasattr(base_agent, "set_parameters"):
            try:
                base_agent.set_parameters(base_params)
            except Exception:
                pass
        if hasattr(base_agent, "clear_replay_buffer"):
            try:
                base_agent.clear_replay_buffer()
            except Exception:
                pass
        if hasattr(base_agent, "set_env"):
            try:
                base_agent.set_env(original_env)
            except Exception:
                pass
        gc.collect()


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
    Leave-one-out style direct impact:
      v(S) - v(S \\ {target_pair}),
    where S is the current candidate_pairs set and v(·) is the validation
    performance after training on the corresponding DreamEnv.
    """
    # Build evaluation environment (same structure as in Shapley helper).
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

    # Coalition S and S \\ {target_pair}
    coalition = list(candidate_pairs or [])
    if target_pair not in coalition:
        coalition.append(target_pair)
    coalition_without = [p for p in coalition if p != target_pair]

    env_with = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim,
        allowed_pairs=coalition,
    )
    env_without = DreamEnv(
        world_model=dream_env.world_model,
        obs_dim=dream_env.obs_dim,
        action_dim=dream_env.action_dim,
        allowed_pairs=coalition_without or None,
    )

    # Snapshot base parameters and original environment.
    base_params = None
    if hasattr(base_agent, "get_parameters"):
        base_params = base_agent.get_parameters()
    original_env = None
    if hasattr(base_agent, "get_env"):
        try:
            original_env = base_agent.get_env()
        except Exception:
            original_env = None

    try:
        # v(S \\ {i})
        if base_params is not None and hasattr(base_agent, "set_parameters"):
            base_agent.set_parameters(base_params)
        if hasattr(base_agent, "clear_replay_buffer"):
            try:
                base_agent.clear_replay_buffer()
            except Exception:
                pass
        if coalition_without:
            base_agent.set_env(env_without)
            base_agent.learn(total_timesteps=mini_train_steps)
        base_agent.set_env(eval_env)
        perf_without = compute_validation_reward(
            base_agent,
            eval_env,
            episodes=num_episodes,
            max_steps=25,
        )

        # v(S)
        if base_params is not None and hasattr(base_agent, "set_parameters"):
            base_agent.set_parameters(base_params)
        if hasattr(base_agent, "clear_replay_buffer"):
            try:
                base_agent.clear_replay_buffer()
            except Exception:
                pass
        base_agent.set_env(env_with)
        base_agent.learn(total_timesteps=mini_train_steps)
        base_agent.set_env(eval_env)
        perf_with = compute_validation_reward(
            base_agent,
            eval_env,
            episodes=num_episodes,
            max_steps=25,
        )

        # Collect a diagnostic trajectory on the target_pair alone.
        single_pair_env = DreamEnv(
            world_model=dream_env.world_model,
            obs_dim=dream_env.obs_dim,
            action_dim=dream_env.action_dim,
            allowed_pairs=[target_pair],
        )
        trajectories = []

        if base_params is not None and hasattr(base_agent, "set_parameters"):
            base_agent.set_parameters(base_params)
        if hasattr(base_agent, "clear_replay_buffer"):
            try:
                base_agent.clear_replay_buffer()
            except Exception:
                pass
        base_agent.set_env(single_pair_env)
        obs, _ = single_pair_env.reset()
        done = False
        step_count = 0
        episode = []
        while not done and step_count < single_pair_env.max_steps:
            action, _ = base_agent.predict(obs, deterministic=False)
            next_obs, reward, done, _, info = single_pair_env.step(action)
            episode.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            step_count += 1
        trajectories.append(episode)

        return perf_with - perf_without, trajectories
    finally:
        if base_params is not None and hasattr(base_agent, "set_parameters"):
            try:
                base_agent.set_parameters(base_params)
            except Exception:
                pass
        if hasattr(base_agent, "clear_replay_buffer"):
            try:
                base_agent.clear_replay_buffer()
            except Exception:
                pass
        if hasattr(base_agent, "set_env"):
            try:
                base_agent.set_env(original_env)
            except Exception:
                pass
        gc.collect()

import numpy as np
import numpy as np
from scipy.spatial.distance import cdist

def collect_episodes(env, agent, num_episodes=5, max_steps=50):
    """
    Collect complete episodes from `env` using `agent.predict(...)`.
    Returns a list of episodes, where each episode is a list of transitions:
        episode[i] = (obs, action, reward, next_obs, done, info)
    """
    episodes = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode = []
        while not done and step_count < max_steps:
            action, _ = agent.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, info = env.step(action)

            episode.append((obs, action, reward, next_obs, done, info))

            obs = next_obs
            step_count += 1
        episodes.append(episode)
    return episodes

def collect_random_episodes(env, num_episodes=50, max_steps=25):
    """
    Run a random policy on the env for a given number of episodes and max steps,
    and return the collected episodes (list of transitions).
    """
    episodes = []
    for ep_idx in range(num_episodes):
        ep = []
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            # Use NumPy's RNG (seeded by the caller) for determinism
            action = int(np.random.randint(0, env.action_space.n))
            next_obs, reward, done, truncated, info = env.step(action)
            ep.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            steps += 1
        episodes.append(ep)

    return episodes

def one_hot_encode(state_idx, grid_size):
    arr = np.zeros(grid_size * grid_size, dtype=np.float32)
    arr[state_idx] = 1.0
    return arr

def select_diverse_pairs(all_pairs, all_pair_values, top_n=100, temperature=1.0):
    """
    Select pairs using a softmax probability distribution with temperature.
    
    Parameters:
        all_pairs: List of (start_idx, goal_idx) pairs
        all_pair_values: Values for each pair
        top_n: Number of pairs to select
        temperature: Higher values (>1.0) increase diversity, lower values (<1.0) focus on top pairs
    
    Returns:
        List of selected pairs
    """
    values_array = np.array(all_pair_values)
    #print("Raw values:", values_array)
    scaled_values = values_array / temperature
    #print("Scaled values:", scaled_values)
    exp_values = np.exp(scaled_values - np.max(scaled_values))
    #print("Exponential values:", exp_values)
    probabilities = exp_values / np.sum(exp_values)
    print("Softmax probabilities:", probabilities[0:20])
  
    # Sample pairs without replacement according to the probability distribution
    selected_indices = np.random.choice(
        len(all_pairs), 
        size=min(top_n, len(all_pairs)),
        replace=False,
        p=probabilities
    )
    
    return [all_pairs[idx] for idx in selected_indices]

def _pair_to_xy(idx, grid):
    """Convert a flat cell index to (x,y) with (0,0) top-left."""
    y, x = divmod(idx, grid)
    return x, y

def mmr_select_pairs(all_pairs,
                     all_pair_values,
                     grid_size,
                     top_n        = 25,
                     lambda_param = 0.5,
                     prefer_low   = False):
    """
    Greedy MMR (Maximal Marginal Relevance) for start–goal pairs.
      • relevance  = predicted value (normalised 0-1)
      • diversity  = Euclidean(start) + Euclidean(goal) (normalised 0-1)
      • prefer_low toggles whether lower values are treated as more relevant.

    Returns a list of indices for the *selected* pairs (len ≤ top_n).
    """
    n = len(all_pairs)
    if n == 0:
        return []

    embed = np.zeros((n, 4), dtype=np.float32)
    for i, (s, g) in enumerate(all_pairs):
        sx, sy = _pair_to_xy(s, grid_size)
        gx, gy = _pair_to_xy(g, grid_size)
        embed[i] = [sx, sy, gx, gy]

    D = cdist(embed, embed, metric="euclidean")
    d0, d1 = D.min(), D.max()
    Dn = np.zeros_like(D) if d1 == d0 else (D - d0) / (d1 - d0)

    v = np.asarray(all_pair_values, dtype=np.float32)
    v0, v1 = v.min(), v.max()
    rn = np.zeros_like(v) if v1 == v0 else (v - v0) / (v1 - v0)
    if prefer_low and rn.size:
        rn = 1.0 - rn

    # Precompute initial z-score for relevance
    eps = 1e-8
    rel_mean = rn.mean() if rn.size else 0.0
    rel_std = rn.std() if rn.size else 1.0
    rel_std = rel_std if rel_std > eps else 1.0
    rn = (rn - rel_mean) / rel_std

    # Precompute diversity z-score baseline
    div_all = Dn.copy()
    div_mean = div_all.mean(axis=1, keepdims=True)
    div_std = div_all.std(axis=1, keepdims=True)
    div_std = np.where(div_std > eps, div_std, 1.0)
    div_all = (div_all - div_mean) / div_std

    # Use a set to track candidates but iterate in a deterministic order
    cand = set(range(n))
    sel  = []

    while len(sel) < top_n and cand:
        best_i, best_score = None, -np.inf

        cand_list = sorted(cand)

        rel_map = {i: rn[i] for i in cand_list}
        if not sel:
            div_map = {i: 0.0 for i in cand_list}
        else:
            div_map = {}
            for i in cand_list:
                nearest = min(div_all[i, j] for j in sel)
                div_map[i] = nearest

        # Deterministic iteration to avoid hash-order dependence
        for i in cand_list:
            rel = rel_map[i]
            div = div_map[i]
            score = lambda_param * rel + (1 - lambda_param) * div
            # Deterministic tie-break: prefer smaller index on equal score
            if score > best_score or (score == best_score and (best_i is None or i < best_i)):
                best_score, best_i = score, i
        sel.append(best_i)
        cand.remove(best_i)

    return sel

#!/usr/bin/env python3
"""
Analyze value estimator trained on aggregated data from multiple seeds.
"""
import os
import json
import glob
import re
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr

GRID_SIZE = 6
CACHE_DIR = os.path.join(os.path.dirname(__file__), "logs/pretrain_10000/valuation_cache")


def load_shapley_data(cache_dir, mode="top_with_mmr", include_stages=False, z_score=False):
    """Load sampled pairs and their Shapley values from all seed files.
    
    Args:
        cache_dir: Path to valuation cache directory
        mode: Which mode files to load (e.g., "top_with_mmr")
        include_stages: If True, also load stage files for more data
        z_score: If True, z-score normalize values within each file before combining
    """
    all_pairs = []
    all_values = []
    
    # Find files
    if include_stages:
        pattern = os.path.join(cache_dir, f"seed_*_{mode}*.json")
    else:
        pattern = os.path.join(cache_dir, f"seed_*_{mode}.json")
    
    files = glob.glob(pattern)
    
    # Sort files naturally (seed_1, seed_2, ... seed_10, not seed_1, seed_10, seed_2)
    def sort_key(f):
        base = os.path.basename(f)
        # Extract seed number and optional stage number
        match = re.match(r'seed_(\d+)_.*?(?:_stage(\d+))?\.json', base)
        if match:
            seed = int(match.group(1))
            stage = int(match.group(2)) if match.group(2) else 0
            return (seed, stage)
        return (999, 0)
    
    files = sorted(files, key=sort_key)
    print(f"Found {len(files)} {mode} files")
    
    file_stats = []
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
        
        sampled = data.get("sampled_pairs", [])
        values = data.get("sampled_pair_values", [])
        
        if sampled and values and len(sampled) == len(values):
            values_arr = np.array(values, dtype=float)
            
            # Z-score normalize within this file
            if z_score and len(values_arr) > 1:
                mean_v = values_arr.mean()
                std_v = values_arr.std()
                if std_v > 1e-8:
                    values_arr = (values_arr - mean_v) / std_v
                else:
                    values_arr = values_arr - mean_v
                file_stats.append((os.path.basename(filepath), len(sampled), mean_v, std_v))
            else:
                file_stats.append((os.path.basename(filepath), len(sampled), values_arr.mean(), values_arr.std()))
            
            all_pairs.extend([tuple(p) for p in sampled])
            all_values.extend(values_arr.tolist())
    
    # Print summary
    if z_score:
        print("\nPer-file statistics (before z-scoring):")
        for fname, n, mean, std in file_stats[:10]:  # Show first 10
            print(f"  {fname}: n={n}, mean={mean:.4f}, std={std:.4f}")
        if len(file_stats) > 10:
            print(f"  ... and {len(file_stats) - 10} more files")
    
    return all_pairs, all_values


def pairs_to_features(pairs, grid_size=GRID_SIZE):
    """Convert (start, goal) pairs to feature vectors."""
    feats = []
    for s, g in pairs:
        s, g = int(s), int(g)
        rs, cs = divmod(s, grid_size)
        rg, cg = divmod(g, grid_size)
        dx = abs(cs - cg)
        dy = abs(rs - rg)
        feats.append([cs, rs, cg, rg, dx, dy])
    return np.array(feats, dtype=float)


def shortest_path_length(start_idx, goal_idx, grid_size=GRID_SIZE, maze=None):
    """BFS shortest path on open grid (no barriers if maze=None)."""
    if start_idx == goal_idx:
        return 0
    
    sr, sc = divmod(start_idx, grid_size)
    gr, gc = divmod(goal_idx, grid_size)
    
    # If no maze provided, use Manhattan distance (open grid)
    if maze is None:
        return abs(sr - gr) + abs(sc - gc)
    
    # BFS with maze
    if maze[sr, sc] == 1 or maze[gr, gc] == 1:
        return np.inf
    
    q = deque([(sr, sc, 0)])
    seen = {(sr, sc)}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while q:
        r, c, d = q.popleft()
        if (r, c) == (gr, gc):
            return d
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if maze is None or maze[nr, nc] == 0:
                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc, d + 1))
    return np.inf


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--z-score", action="store_true", help="Z-score normalize values per file")
    parser.add_argument("--include-stages", action="store_true", help="Include stage files for more data")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Loading Shapley data from cache...")
    print(f"  Z-score normalization: {args.z_score}")
    print(f"  Include stage files: {args.include_stages}")
    print("=" * 60)
    
    pairs, values = load_shapley_data(
        CACHE_DIR, 
        mode="top_with_mmr",
        include_stages=args.include_stages,
        z_score=args.z_score
    )
    
    print(f"\nTotal training samples: {len(pairs)}")
    print(f"Value range: {min(values):.4f} to {max(values):.4f}")
    print(f"Value mean: {np.mean(values):.4f}, std: {np.std(values):.4f}")
    
    # Count start==goal pairs
    eq_pairs = [(p, v) for p, v in zip(pairs, values) if p[0] == p[1]]
    print(f"\nStart==goal pairs in training: {len(eq_pairs)}")
    if eq_pairs:
        eq_vals = [v for _, v in eq_pairs]
        print(f"  Values: min={min(eq_vals):.4f}, max={max(eq_vals):.4f}, mean={np.mean(eq_vals):.4f}")
    
    # Train forest
    print("\n" + "=" * 60)
    print("Training RandomForest estimator...")
    print("=" * 60)
    
    X = pairs_to_features(pairs)
    y = np.array(values)
    
    forest = RandomForestRegressor(n_estimators=100, random_state=42)
    forest.fit(X, y)
    
    r2 = forest.score(X, y)
    print(f"Training R²: {r2:.4f}")
    
    # Feature importances
    print("\nFeature importances:")
    feature_names = ["cs", "rs", "cg", "rg", "dx", "dy"]
    for name, imp in zip(feature_names, forest.feature_importances_):
        print(f"  {name}: {imp:.4f}")
    
    # Generate all possible pairs and predict
    print("\n" + "=" * 60)
    print("Predicting values for ALL possible pairs...")
    print("=" * 60)
    
    all_possible = [(s, g) for s in range(GRID_SIZE**2) for g in range(GRID_SIZE**2)]
    X_all = pairs_to_features(all_possible)
    pred_all = forest.predict(X_all)
    
    # Analyze start==goal predictions
    eq_indices = [i for i, (s, g) in enumerate(all_possible) if s == g]
    eq_preds = pred_all[eq_indices]
    
    print(f"\nPredictions for start==goal pairs ({len(eq_indices)} pairs):")
    print(f"  Min: {eq_preds.min():.4f}")
    print(f"  Max: {eq_preds.max():.4f}")
    print(f"  Mean: {eq_preds.mean():.4f}")
    print(f"  Std: {eq_preds.std():.4f}")
    
    # Compare to non-equal pairs
    neq_indices = [i for i, (s, g) in enumerate(all_possible) if s != g]
    neq_preds = pred_all[neq_indices]
    
    print(f"\nPredictions for start!=goal pairs ({len(neq_indices)} pairs):")
    print(f"  Min: {neq_preds.min():.4f}")
    print(f"  Max: {neq_preds.max():.4f}")
    print(f"  Mean: {neq_preds.mean():.4f}")
    print(f"  Std: {neq_preds.std():.4f}")
    
    # Correlation with path length (Manhattan distance on open grid)
    print("\n" + "=" * 60)
    print("Correlation with path length (Manhattan distance)...")
    print("=" * 60)
    
    lengths = np.array([shortest_path_length(s, g) for s, g in all_possible])
    
    # Only non-equal pairs for correlation
    neq_lengths = lengths[neq_indices]
    neq_preds_arr = pred_all[neq_indices]
    
    pearson_r, pearson_p = pearsonr(neq_lengths, neq_preds_arr)
    spearman_r, spearman_p = spearmanr(neq_lengths, neq_preds_arr)
    
    print(f"Pearson correlation (pred vs length): r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman correlation (pred vs length): r={spearman_r:.4f}, p={spearman_p:.4g}")
    
    # Also check correlation in training data
    print("\n" + "=" * 60)
    print("Correlation in TRAINING data (actual Shapley vs length)...")
    print("=" * 60)
    
    train_lengths = np.array([shortest_path_length(s, g) for s, g in pairs])
    train_values = np.array(values)
    
    # Filter out start==goal for correlation
    mask = train_lengths > 0
    if mask.sum() > 2:
        train_pearson_r, train_pearson_p = pearsonr(train_lengths[mask], train_values[mask])
        train_spearman_r, train_spearman_p = spearmanr(train_lengths[mask], train_values[mask])
        print(f"Pearson (actual Shapley vs length): r={train_pearson_r:.4f}, p={train_pearson_p:.4g}")
        print(f"Spearman (actual Shapley vs length): r={train_spearman_r:.4f}, p={train_spearman_p:.4g}")
    
    # Rank analysis: where do start==goal pairs rank?
    print("\n" + "=" * 60)
    print("Ranking analysis...")
    print("=" * 60)
    
    sorted_indices = np.argsort(pred_all)[::-1]  # descending
    
    # Find rank of each start==goal pair
    eq_ranks = []
    for eq_idx in eq_indices:
        rank = np.where(sorted_indices == eq_idx)[0][0]
        eq_ranks.append(rank)
    
    print(f"Start==goal pairs rank (out of {len(all_possible)}):")
    print(f"  Best rank: {min(eq_ranks)}")
    print(f"  Worst rank: {max(eq_ranks)}")
    print(f"  Mean rank: {np.mean(eq_ranks):.1f}")
    print(f"  Median rank: {np.median(eq_ranks):.1f}")
    
    # How many would be selected in top 200?
    top_200 = set(sorted_indices[:200])
    eq_in_top_200 = sum(1 for idx in eq_indices if idx in top_200)
    print(f"\nStart==goal pairs in top 200: {eq_in_top_200}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if eq_preds.mean() < neq_preds.mean():
        print("✓ Model predicts LOWER values for start==goal (good!)")
        print(f"  Difference: {neq_preds.mean() - eq_preds.mean():.4f}")
    else:
        print("✗ Model predicts HIGHER values for start==goal (bad!)")
        print(f"  Difference: {eq_preds.mean() - neq_preds.mean():.4f}")
    
    if pearson_r > 0:
        print(f"✓ Positive correlation with path length: {pearson_r:.4f}")
    else:
        print(f"✗ Negative/no correlation with path length: {pearson_r:.4f}")


if __name__ == "__main__":
    main()


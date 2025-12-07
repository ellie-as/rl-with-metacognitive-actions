#!/usr/bin/env python3
"""
Find the best sklearn estimator for predicting start-goal pair values.
Tries different feature engineering approaches and models.
"""
import os
import sys
import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))


def load_warm_start_data_raw(cache_dir, mode="top_with_mmr", include_stages=True):
    """Load warm-start data WITHOUT normalization to see raw values."""
    import glob
    
    all_pairs = []
    all_values = []
    
    pattern = f"seed_*_{mode}*.json" if include_stages else f"seed_*_{mode}.json"
    files = glob.glob(os.path.join(cache_dir, pattern))
    
    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            sampled = data.get("sampled_pairs", [])
            values = data.get("sampled_pair_values", [])
            
            if sampled and values and len(sampled) == len(values):
                all_pairs.extend([tuple(p) for p in sampled])
                all_values.extend(values)
        except Exception:
            continue
    
    return all_pairs, all_values


def pairs_to_features_basic(pairs, grid_size=6):
    """Basic features: start/goal positions + manhattan distance."""
    feats = []
    for s, g in pairs:
        s, g = int(s), int(g)
        rs, cs = divmod(s, grid_size)
        rg, cg = divmod(g, grid_size)
        dx = abs(cs - cg)
        dy = abs(rs - rg)
        feats.append([cs, rs, cg, rg, dx, dy])
    return np.array(feats, dtype=float)


def pairs_to_features_extended(pairs, grid_size=6):
    """Extended features: basic + is_same + normalized positions + distance features."""
    feats = []
    for s, g in pairs:
        s, g = int(s), int(g)
        rs, cs = divmod(s, grid_size)
        rg, cg = divmod(g, grid_size)
        dx = abs(cs - cg)
        dy = abs(rs - rg)
        
        # Basic
        is_same = 1.0 if s == g else 0.0
        manhattan = dx + dy
        
        # Normalized positions (0-1)
        cs_norm = cs / (grid_size - 1)
        rs_norm = rs / (grid_size - 1)
        cg_norm = cg / (grid_size - 1)
        rg_norm = rg / (grid_size - 1)
        
        # Distance features
        euclidean = np.sqrt(dx**2 + dy**2)
        max_dist = np.sqrt(2) * (grid_size - 1)
        dist_norm = euclidean / max_dist if max_dist > 0 else 0
        
        # Directional features
        dir_x = (cg - cs) / (grid_size - 1) if grid_size > 1 else 0
        dir_y = (rg - rs) / (grid_size - 1) if grid_size > 1 else 0
        
        # Edge/corner features
        start_edge = 1.0 if rs == 0 or rs == grid_size-1 or cs == 0 or cs == grid_size-1 else 0.0
        goal_edge = 1.0 if rg == 0 or rg == grid_size-1 or cg == 0 or cg == grid_size-1 else 0.0
        start_corner = 1.0 if (rs in [0, grid_size-1]) and (cs in [0, grid_size-1]) else 0.0
        goal_corner = 1.0 if (rg in [0, grid_size-1]) and (cg in [0, grid_size-1]) else 0.0
        
        feats.append([
            cs, rs, cg, rg,  # positions
            dx, dy,  # deltas
            is_same,  # same check
            manhattan, euclidean, dist_norm,  # distances
            dir_x, dir_y,  # directions
            cs_norm, rs_norm, cg_norm, rg_norm,  # normalized positions
            start_edge, goal_edge, start_corner, goal_corner,  # edge features
        ])
    return np.array(feats, dtype=float)


def pairs_to_features_onehot(pairs, grid_size=6):
    """One-hot encoding of start and goal positions."""
    n_cells = grid_size * grid_size
    feats = []
    for s, g in pairs:
        s, g = int(s), int(g)
        start_oh = np.zeros(n_cells)
        goal_oh = np.zeros(n_cells)
        start_oh[s] = 1.0
        goal_oh[g] = 1.0
        feats.append(np.concatenate([start_oh, goal_oh]))
    return np.array(feats, dtype=float)


def pairs_to_features_combined(pairs, grid_size=6):
    """Combine extended features with one-hot encoding."""
    ext = pairs_to_features_extended(pairs, grid_size)
    oh = pairs_to_features_onehot(pairs, grid_size)
    return np.hstack([ext, oh])


def main():
    print("=" * 70)
    print("Finding Best Value Estimator")
    print("=" * 70)
    
    # Load data
    cache_dir = os.path.join(os.path.dirname(__file__), "logs/pretrain_10000/warm_start_cache")
    print(f"\nLoading data from: {cache_dir}")
    
    pairs, values = load_warm_start_data_raw(cache_dir, mode="top_with_mmr", include_stages=True)
    print(f"Loaded {len(pairs)} pairs")
    
    if not pairs:
        print("ERROR: No data found!")
        return
    
    y = np.array(values)
    print(f"\nRaw value stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Analyze start==goal values
    eq_vals = [v for (s, g), v in zip(pairs, values) if s == g]
    neq_vals = [v for (s, g), v in zip(pairs, values) if s != g]
    print(f"\nstart==goal: n={len(eq_vals)}, mean={np.mean(eq_vals):.4f}, std={np.std(eq_vals):.4f}")
    print(f"start!=goal: n={len(neq_vals)}, mean={np.mean(neq_vals):.4f}, std={np.std(neq_vals):.4f}")
    
    # Z-score normalize for training
    y_norm = (y - y.mean()) / y.std()
    
    # Feature sets to try
    feature_sets = {
        "basic (6 feats)": pairs_to_features_basic(pairs),
        "extended (20 feats)": pairs_to_features_extended(pairs),
        "one-hot (72 feats)": pairs_to_features_onehot(pairs),
        "combined (92 feats)": pairs_to_features_combined(pairs),
    }
    
    # Models to try
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "KNN(5)": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "KNN(10)": KNeighborsRegressor(n_neighbors=10, weights="distance"),
    }
    
    # Try each combination
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS (5-fold)")
    print("=" * 70)
    
    results = {}
    
    for feat_name, X in feature_sets.items():
        print(f"\n--- {feat_name} ---")
        print(f"{'Model':<25} {'CV R² Mean':<15} {'CV R² Std':<15}")
        print("-" * 55)
        
        for model_name, model in models.items():
            try:
                scores = cross_val_score(model, X, y_norm, cv=5, scoring='r2', n_jobs=-1)
                key = f"{feat_name} + {model_name}"
                results[key] = (scores.mean(), scores.std(), feat_name, model_name)
                print(f"{model_name:<25} {scores.mean():.4f}          {scores.std():.4f}")
            except Exception as e:
                print(f"{model_name:<25} ERROR: {e}")
    
    # Find best
    print("\n" + "=" * 70)
    print("TOP 5 COMBINATIONS")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for i, (name, (mean_r2, std_r2, feat_name, model_name)) in enumerate(sorted_results[:5]):
        print(f"{i+1}. {name}: R²={mean_r2:.4f} ± {std_r2:.4f}")
    
    # Best combination
    best_name, (best_r2, best_std, best_feat, best_model) = sorted_results[0]
    print(f"\n*** BEST: {best_name} ***")
    print(f"    R² = {best_r2:.4f} ± {best_std:.4f}")
    
    # Train best model and check predictions
    print("\n" + "=" * 70)
    print("BEST MODEL ANALYSIS")
    print("=" * 70)
    
    X_best = feature_sets[best_feat]
    final_model = models[best_model].__class__(**models[best_model].get_params())
    final_model.fit(X_best, y_norm)
    
    # Check start==goal predictions
    all_test_pairs = [(s, g) for s in range(36) for g in range(36)]
    if best_feat == "basic (6 feats)":
        X_test = pairs_to_features_basic(all_test_pairs)
    elif best_feat == "extended (20 feats)":
        X_test = pairs_to_features_extended(all_test_pairs)
    elif best_feat == "one-hot (72 feats)":
        X_test = pairs_to_features_onehot(all_test_pairs)
    else:
        X_test = pairs_to_features_combined(all_test_pairs)
    
    preds = final_model.predict(X_test)
    eq_preds = [p for (s, g), p in zip(all_test_pairs, preds) if s == g]
    neq_preds = [p for (s, g), p in zip(all_test_pairs, preds) if s != g]
    
    print(f"\nPredictions on all 36x36 pairs:")
    print(f"  start==goal: mean={np.mean(eq_preds):.4f}, std={np.std(eq_preds):.4f}")
    print(f"  start!=goal: mean={np.mean(neq_preds):.4f}, std={np.std(neq_preds):.4f}")
    print(f"  Separation: {np.mean(neq_preds) - np.mean(eq_preds):.4f}")
    
    # Feature importance (if available)
    if hasattr(final_model, 'feature_importances_'):
        print(f"\nTop 10 feature importances:")
        importances = final_model.feature_importances_
        if best_feat == "extended (20 feats)":
            feat_names = ['cs', 'rs', 'cg', 'rg', 'dx', 'dy', 'is_same', 
                         'manhattan', 'euclidean', 'dist_norm', 'dir_x', 'dir_y',
                         'cs_norm', 'rs_norm', 'cg_norm', 'rg_norm',
                         'start_edge', 'goal_edge', 'start_corner', 'goal_corner']
        elif best_feat == "basic (6 feats)":
            feat_names = ['cs', 'rs', 'cg', 'rg', 'dx', 'dy']
        else:
            feat_names = [f"feat_{i}" for i in range(len(importances))]
        
        sorted_idx = np.argsort(importances)[::-1]
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
            print(f"  {name}: {importances[idx]:.4f}")


if __name__ == "__main__":
    main()


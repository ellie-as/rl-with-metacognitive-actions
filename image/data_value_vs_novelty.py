#!/usr/bin/env python3
"""Replicate the analysis from image/Data value vs novelty.ipynb as a CLI script."""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from config import Config
from environment import MetaLearningEnv
from seed_utils import derive_seed, seed_all

NUM_TRIALS = 10
NUM_STAGES = 3
N_EVAL_IMGS = 500  # subset of test images used for evaluation
MODES = ("buffer", "gen")
BASE_SEED = Config.SEED

NOVELTY_METRICS = ("distance", "difficulty")
METRIC_LABELS = {
    "distance": "Distance from class mean (L2 norm)",
    "difficulty": "Ensemble difficulty (1 - accuracy)",
}

# Disable noise used in other experiments.
setattr(Config, "LEVELS", (0.0, 0.0))


def get_test_subset(data_loader, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return tensors (X, y) of n random test images."""
    idxs = rng.choice(len(data_loader.test_dataset), size=n, replace=False)
    imgs, lbls = zip(*(data_loader.test_dataset[i] for i in idxs))
    X = torch.stack(imgs)  # (n, 1, 28, 28)
    y = torch.tensor(lbls, dtype=torch.int64)
    return X, y


def run_trials_for_mode(mode: str, novelty_metric: str) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """Execute the experiment for a single training mode and collect raw arrays."""
    raw_data: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

    for trial in range(NUM_TRIALS):
        trial_seed = derive_seed(BASE_SEED, "value_vs_novelty", mode, trial)
        seed_all(trial_seed)
        trial_rng = np.random.default_rng(trial_seed)

        print(f"\n=== Trial {trial + 1}/{NUM_TRIALS} ===")
        env = MetaLearningEnv()
        # Tag outputs with trial id so valuator filenames are per-trial
        env._trial_id = trial
        env.action_space.seed(trial_seed)

        if mode == "gen":
            env._train_generator()

        # Compute per-class means from the full training set.
        train_imgs, train_lbls = zip(*(env.data_loader.train_dataset[i]
                                       for i in range(len(env.data_loader.train_dataset))))
        X_train = (
            torch.stack(train_imgs)
            .view(-1, 28 * 28)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        y_train = np.array(train_lbls, dtype=np.int32)

        class_means = {}
        for cls in range(Config.NUM_CLASSES):
            class_means[cls] = X_train[y_train == cls].mean(axis=0)

        X_test, y_test = get_test_subset(env.data_loader, N_EVAL_IMGS, trial_rng)
        X_test_np = X_test.cpu().numpy().astype(np.float32)
        X_flat = X_test_np.reshape(N_EVAL_IMGS, -1)
        y_np = y_test.cpu().numpy()

        if novelty_metric == "difficulty":
            from data_valuation_comparison import _build_ensemble_models, _compute_difficulties

            ensemble_rng = np.random.default_rng(derive_seed(trial_seed, "difficulty_ensemble"))
            ensemble = _build_ensemble_models(X_train, y_train, ensemble_rng)
            buffer_samples = [
                (X_flat[idx], int(y_np[idx]))
                for idx in range(N_EVAL_IMGS)
            ]
            difficulty_scores = 1.0 - _compute_difficulties(buffer_samples, ensemble)
        else:
            difficulty_scores = None

        for stage in range(NUM_STAGES):
            if mode == "gen":
                env._train_on_generated()
            else:
                env._train_classifier_on_buffer()

            print(f"-- Stage {stage + 1}/{NUM_STAGES} --")

            one_hot = np.eye(Config.NUM_CLASSES, dtype=np.float32)[y_np]
            preds = env.value_estimator.predict(
                np.concatenate([X_flat, one_hot], axis=1)
            )

            if difficulty_scores is not None:
                novelty_values = difficulty_scores
            else:
                means_stack = np.stack([class_means[label] for label in y_np], axis=0)
                novelty_values = np.linalg.norm(X_flat - means_stack, axis=1)

            raw_data[(trial, stage)] = {
                "novelty": novelty_values,
                "preds": preds,
                "labels": y_np.copy(),
                "mode": mode,
            }

    return raw_data


def _extract_novelty(stage_data: Dict[str, np.ndarray]) -> np.ndarray:
    if "novelty" in stage_data:
        return stage_data["novelty"]
    return stage_data["distances"]


def plot_single_trial_scatter(raw_data, mode: str, metric: str, trial: int = 0) -> None:
    for stage in range(NUM_STAGES):
        stage_key = (trial, stage)
        d_all = _extract_novelty(raw_data[stage_key])
        p_all = raw_data[stage_key]["preds"]

        plt.figure(figsize=(4, 3))
        plt.scatter(d_all, p_all, s=10, alpha=0.5)
        plt.xlabel(METRIC_LABELS.get(metric, "Novelty"))
        plt.ylabel("Predicted value")
        plt.title(f"Stage {stage + 1}")
        plt.tight_layout()
        plt.savefig(f"value_vs_{metric}_stage{stage + 1}_{mode}.png")
        plt.show()


def fisher_ci(r: float, N: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute a two-sided confidence interval for Pearson r via Fisher z."""
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(N - 3)
    z_crit = 1.96  # 95% CI
    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    return np.tanh(z_lo), np.tanh(z_hi)


def summarize_across_trials(raw_data, mode: str, metric: str) -> None:
    r_vals = []
    err_vals = []

    for stage in range(NUM_STAGES):
        distances = []
        predictions = []
        for trial in range(NUM_TRIALS):
            distances.append(_extract_novelty(raw_data[(trial, stage)]))
            predictions.append(raw_data[(trial, stage)]["preds"])

        d_all = np.concatenate(distances)
        p_all = np.concatenate(predictions)

        coeff, intercept = np.polyfit(d_all, p_all, 1)
        xs = np.linspace(d_all.min(), d_all.max(), 200)
        ys = coeff * xs + intercept

        r_val, p_val = pearsonr(d_all, p_all)
        conf_lo, conf_hi = fisher_ci(r_val, len(d_all))
        err = 0.5 * (conf_hi - conf_lo)

        r_vals.append(r_val)
        err_vals.append(err)

        plt.figure(figsize=(4, 3))
        plt.scatter(d_all, p_all, s=8, alpha=0.4, label="data")
        plt.plot(xs, ys, "r-", linewidth=2,
                 label=f"fit: y = {coeff:.3f}x + {intercept:.3f}")
        plt.xlabel(METRIC_LABELS.get(metric, "Novelty"))
        plt.ylabel("Predicted value")
        plt.title(f"Stage {stage + 1}\nPearson r = {r_val:.3f} (p = {p_val:.2e})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"value_vs_{metric}_fit_stage{stage + 1}_{mode}.png")
        plt.show()

    plt.figure(figsize=(3, 2))
    stages = [f"Step {idx}" for idx in range(NUM_STAGES)]
    plt.bar(stages, r_vals, yerr=err_vals, capsize=5, alpha=1)
    plt.ylabel("Value-novelty $r_p$")
    plt.tight_layout()
    plt.savefig(f"pearson_r_by_stage_{metric}_{mode}.png", dpi=200)
    plt.show()


def save_raw_data(raw_data, mode: str, metric: str) -> None:
    out_path = f"raw_data_{metric}_{mode}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(raw_data, f)
    print(f"Raw data saved → {out_path}")


def load_raw_data(mode: str, metric: str) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    target = Path(f"raw_data_{metric}_{mode}.pkl")
    if not target.exists() and metric == "distance":
        legacy = Path(f"raw_data_dist_{mode}.pkl")
        if legacy.exists():
            target = legacy
    with target.open("rb") as f:
        return pickle.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--novelty-metric",
        choices=NOVELTY_METRICS,
        default="distance",
        help="Metric to correlate against predicted values",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(BASE_SEED)

    for mode in MODES:
        raw = run_trials_for_mode(mode, args.novelty_metric)
        plot_single_trial_scatter(raw, mode, args.novelty_metric, trial=0)
        save_raw_data(raw, mode, args.novelty_metric)

    for mode in ("gen", "buffer"):
        raw = load_raw_data(mode, args.novelty_metric)
        summarize_across_trials(raw, mode, args.novelty_metric)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Comprehensive data valuation analysis.

This script consolidates the functionality of the previous
`data_valuation_comparison.py` and `compare_data_valuation.py` utilities.
It evaluates:

1. Buffer-based selection strategies (data valuation network, random,
   most challenging, least challenging).
2. Fixed action sequences in the MetaLearningEnv across all combinations of
   the `Config.MMR` and `Config.SHAPLEY` flags for both buffer-only and
   generator-based regimes.

All results are cached to JSON so repeated runs reuse prior computations
unless `--force` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ttest_ind
from sklearn.linear_model import SGDRegressor
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline

from classifier import Classifier
from config import Config
from environment import MetaLearningEnv
from environment import MNISTStageLoader as StageLoader
from seed_utils import derive_seed, seed_all
from generator import GaussianGenerator
from analysis_utils import (
    _summary_stats,
    _rng_seed_seq,
    _sample_training_data,
    _build_ensemble_models,
    _compute_difficulties,
    _mmr_select_indices,
    _partial_fit_classifier,
    _extract_arrays,
    _label_counts,
    _sample_generative_batch,
    _select_dvn_indices,
    _select_random_indices,
    _select_extreme_indices,
    _compute_sem,
    _significance_stars,
)

LOGGER = logging.getLogger("data_valuation_comparison")

APPROACH_LABELS = (
    "Data valuation network",
    "Random selection",
    "Most challenging",
    "Least challenging",
)

MEMORY_PHASES = 3
GENERATIVE_PHASES = 2


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    actions: Sequence[int]
    steps_per_episode: int


REPLAY_SCENARIOS: Sequence[ReplayScenario] = (
    ReplayScenario(name="buffer", actions=(0,), steps_per_episode=1),
    ReplayScenario(name="generated", actions=(1, 2), steps_per_episode=2),
)

REPLAY_COMBINATIONS: Sequence[Tuple[bool, bool]] = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "logs" / "data_valuation_results.json"


class DifficultySelectionEnv(MetaLearningEnv):
    """Env variant that selects buffer/generated samples by ensemble difficulty.

    mode: "most" for hardest (lowest ensemble accuracy) or "least" for easiest.
    """

    def __init__(self, ensemble: Sequence[Pipeline], mode: str = "most"):
        super().__init__()
        self._ensemble = list(ensemble)
        if mode not in {"most", "least"}:
            raise ValueError("mode must be 'most' or 'least'")
        self._mode = mode

    def _select_indices_by_difficulty(self, samples: List[tuple[np.ndarray, int]], k: int) -> List[int]:
        if not samples or k <= 0:
            return []
        diffs = _compute_difficulties(samples, self._ensemble)
        order = np.argsort(diffs)
        if self._mode == "most":
            chosen = order[:k]
        else:
            chosen = order[-k:]
        return [int(i) for i in chosen]

    def _train_classifier_on_buffer(self):
        if not self.buffer:
            return
        X_flat = np.array([x.flatten() for x, _ in self.buffer])
        yb = np.array([lbl for _, lbl in self.buffer], dtype=np.int64)
        k = min(Config.NUM_TO_SELECT, len(self.buffer))

        indices = self._select_indices_by_difficulty(list(self.buffer), k)
        if not indices:
            return

        Xb = X_flat[indices].astype(np.float32)
        yb_sel = yb[indices]
        _partial_fit_classifier(self.classifier, Xb, yb_sel, Config.PARTIAL_FIT_EPS)
        _, per_class_acc = self.classifier.evaluate_with_breakdown(self.val_loader)
        self.class_acc = per_class_acc

    def _train_on_generated(self):
        if not getattr(self.generator, 'class_cluster_params', None):
            logging.warning("Generator not trained – skipping.")
            return
        classes = [
            c
            for c, (_, params) in sorted(self.generator.class_cluster_params.items())
            if params
        ]
        if not classes:
            logging.warning("Generator holds no clusters – skipping.")
            return

        per = Config.BUFFER_CAPACITY // max(1, len(classes)) if (Config.MMR or Config.SHAPLEY) else max(1, Config.NUM_TO_SELECT // max(1, len(classes)))
        Xg_list, yg_list = [], []
        for c in classes:
            imgs, labs = self.generator.generate(c, per)
            Xg_list.append(imgs)
            yg_list.append(labs)
        Xg = np.concatenate([x.view(x.size(0), -1).cpu().numpy() for x in Xg_list], axis=0)
        yg = np.concatenate([y.cpu().numpy() for y in yg_list], axis=0).astype(np.int64)

        samples = [(Xg[i].reshape(28, 28), int(yg[i])) for i in range(len(yg))]
        k = min(Config.NUM_TO_SELECT, len(samples))
        indices = self._select_indices_by_difficulty(samples, k)
        if not indices:
            return
        Xb = Xg[indices].astype(np.float32)
        yb = yg[indices]
        _partial_fit_classifier(self.classifier, Xb, yb, Config.PARTIAL_FIT_EPS)
        _, per_class_acc = self.classifier.evaluate_with_breakdown(self.val_loader)
        self.class_acc = per_class_acc


## Refactored: use environment.MNISTStageLoader to avoid duplication


## moved to analysis_utils: _rng_seed_seq


## moved to analysis_utils: _sample_training_data


## moved to analysis_utils: _build_ensemble_models


def _compute_difficulties(buffer_samples: List[tuple[np.ndarray, int]], ensemble: List[Pipeline]) -> np.ndarray:
    scores = np.zeros(len(buffer_samples), dtype=np.float32)
    for idx, (img_np, label) in enumerate(buffer_samples):
        features = img_np.reshape(1, -1)
        correct = 0
        for model in ensemble:
            pred = int(model.predict(features)[0])
            if pred == int(label):
                correct += 1
        scores[idx] = correct / max(1, len(ensemble))
    return scores


def _mmr_select_indices(
    X_flat: np.ndarray,
    vals: np.ndarray,
    k: int,
    *,
    labels: Optional[np.ndarray] = None,
    balance_classes: bool = True,
) -> List[int]:
    λ = Config.LAMBDA_PARAM
    n = len(vals)
    if n == 0 or k == 0:
        return []
    k = min(k, n)

    v_min, v_max = float(vals.min()), float(vals.max())
    if math.isclose(v_min, v_max):
        rel_norm = np.zeros_like(vals, dtype=np.float32)
    else:
        rel_norm = (vals - v_min) / (v_max - v_min)

    distances = np.linalg.norm(X_flat[:, None, :] - X_flat[None, :, :], axis=2)
    d_min, d_max = float(distances.min()), float(distances.max())
    if math.isclose(d_min, d_max):
        dist_norm = np.zeros_like(distances, dtype=np.float32)
    else:
        dist_norm = (distances - d_min) / (d_max - d_min)

    candidates = set(range(n))
    selected: List[int] = []
    class_counts: Dict[int, int] = {}

    if balance_classes and labels is not None and labels.size > 0:
        unique_labels = np.unique(labels)
        quota = max(1, int(math.ceil(k / len(unique_labels))))
    else:
        quota = None

    while len(selected) < k and candidates:
        best_idx, best_score = None, -np.inf
        for i in list(candidates):
            if quota is not None and labels is not None:
                lbl = int(labels[i])
                if class_counts.get(lbl, 0) >= quota:
                    continue

            relevance = rel_norm[i]
            if not selected:
                score = relevance
            else:
                div = dist_norm[i, selected].min()
                score = λ * relevance + (1.0 - λ) * div

            if score > best_score:
                best_idx, best_score = i, score

        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
        if quota is not None and labels is not None:
            lbl = int(labels[best_idx])
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

    return selected


def _partial_fit_classifier(model: Classifier, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
    y_int = y.astype(np.int64)
    remaining_epochs = max(0, int(epochs))
    if not hasattr(model, "initialized_"):
        model.partial_fit(X, y_int, classes=np.arange(Config.NUM_CLASSES))
        remaining_epochs -= 1
    for _ in range(max(0, remaining_epochs)):
        model.partial_fit(X, y_int)


def _extract_arrays(
    buffer_samples: Sequence[tuple[np.ndarray, int]],
    indices: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    idx_list = [int(i) for i in indices]
    if not idx_list:
        return np.empty((0, 28 * 28), dtype=np.float32), np.empty(0, dtype=np.int64)
    X = np.stack([buffer_samples[i][0] for i in idx_list]).astype(np.float32)
    y = np.array([buffer_samples[i][1] for i in idx_list], dtype=np.int64)
    return X, y


def _label_counts(samples: Sequence[tuple[np.ndarray, int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, label in samples:
        lbl = int(label)
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _sample_generative_batch(
    generator: GaussianGenerator,
    label_counts: Dict[int, int],
    batch_size: int,
    rng: np.random.Generator,
) -> List[tuple[np.ndarray, int]]:
    if batch_size <= 0 or not label_counts:
        return []

    labels = np.array(list(label_counts.keys()), dtype=np.int64)
    weights = np.array([label_counts[int(lbl)] for lbl in labels], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return []
    probs = weights / total
    counts = rng.multinomial(batch_size, probs)

    generated: List[tuple[np.ndarray, int]] = []
    for lbl, count in zip(labels, counts):
        if count <= 0:
            continue
        try:
            imgs, labels_tensor = generator.generate(int(lbl), num_samples=int(count))
        except Exception as exc:
            LOGGER.warning("Generator failed for label %s: %s", lbl, exc)
            continue
        img_np = imgs.cpu().numpy()
        lbl_np = labels_tensor.cpu().numpy()
        for idx in range(img_np.shape[0]):
            generated.append((img_np[idx], int(lbl_np[idx])))

    return generated


def _run_strategy_pipeline(
    strategy: str,
    stage_loader: StageLoader,
    ensemble: Sequence[Pipeline],
    test_loader: DataLoader,
    val_dataset,
    *,
    memory_phases: int,
    generative_phases: int,
    epochs: int,
    base_seed: int,
    rng: np.random.Generator,
) -> float:
    seed_all(base_seed)

    classifier = Classifier(verbose=0, train_split=False)
    estimator = SGDRegressor(random_state=_rng_seed_seq(rng)) if strategy == "Data valuation network" else None

    total_stage_options = max(1, len(Config.STAGES))
    stage_index = 0
    cached_samples: Optional[List[tuple[np.ndarray, int]]] = None
    cached_difficulties: Optional[np.ndarray] = None
    memory_selected: List[tuple[np.ndarray, int]] = []

    for phase_idx in range(max(0, memory_phases)):
        phase_seed = derive_seed(base_seed, "memory_phase", strategy, phase_idx)
        seed_all(phase_seed)

        if strategy in {"Most challenging", "Least challenging"}:
            if cached_samples is None:
                samples = stage_loader.get_stage_samples(stage_index, Config.BUFFER_CAPACITY)
                cached_samples = samples
                cached_difficulties = _compute_difficulties(samples, ensemble)
            samples = cached_samples
            difficulties = cached_difficulties
        else:
            samples = stage_loader.get_stage_samples(stage_index, Config.BUFFER_CAPACITY)
            difficulties = None

        if strategy == "Data valuation network":
            if estimator is None:
                estimator = SGDRegressor(random_state=_rng_seed_seq(rng))
            indices, estimator = _select_dvn_indices(classifier, estimator, samples, val_dataset, rng)
        elif strategy == "Random selection":
            indices = _select_random_indices(len(samples), Config.NUM_TO_SELECT, rng)
        elif strategy == "Most challenging":
            indices = _select_extreme_indices(cached_difficulties, Config.NUM_TO_SELECT, hardest=True) if cached_difficulties is not None else []
        else:
            indices = _select_extreme_indices(cached_difficulties, Config.NUM_TO_SELECT, hardest=False) if cached_difficulties is not None else []

        if not indices:
            LOGGER.warning("%s memory phase %d selected no samples", strategy, phase_idx + 1)
        else:
            X, y = _extract_arrays(samples, indices)
            _partial_fit_classifier(classifier, X, y, epochs)
            memory_selected.extend((samples[i][0], samples[i][1]) for i in indices)

        if strategy not in {"Most challenging", "Least challenging"}:
            stage_index = (stage_index + 1) % total_stage_options

    world_model: Optional[GaussianGenerator] = None
    label_counts = _label_counts(memory_selected)
    if label_counts:
        world_model = GaussianGenerator()
        try:
            world_model.fit(memory_selected)
        except Exception as exc:
            LOGGER.warning("World model fit failed for %s: %s", strategy, exc)
            world_model = None
    else:
        LOGGER.warning("%s had no memory samples for world model update", strategy)

    for gen_idx in range(max(0, generative_phases)):
        if world_model is None or not label_counts:
            LOGGER.warning("%s generative phase %d skipped", strategy, gen_idx + 1)
            continue
        gen_seed = derive_seed(base_seed, "generative_phase", strategy, gen_idx)
        seed_all(gen_seed)
        gen_samples = _sample_generative_batch(world_model, label_counts, Config.NUM_TO_SELECT, rng)
        if not gen_samples:
            LOGGER.warning("%s generative phase %d produced no samples", strategy, gen_idx + 1)
            continue
        Xg, yg = _extract_arrays(gen_samples, range(len(gen_samples)))
        _partial_fit_classifier(classifier, Xg, yg, epochs)

    return float(classifier.evaluate(test_loader))


## moved to analysis_utils: _select_dvn_indices


def _select_random_indices(length: int, k: int, rng: np.random.Generator) -> List[int]:
    if length == 0 or k == 0:
        return []
    k = min(k, length)
    return list(rng.choice(length, size=k, replace=False))


def _select_extreme_indices(difficulty: np.ndarray, k: int, *, hardest: bool) -> List[int]:
    if difficulty.size == 0 or k == 0:
        return []
    k = min(k, difficulty.size)
    order = np.argsort(difficulty)
    if hardest:
        return list(order[:k])
    return list(order[-k:])


## Removed Buffer selection strategies output path per request


def run_strategy_test(actions: Sequence[int], *, episodes: int, seed: int) -> List[float]:
    base_seed = derive_seed(seed, "strategy", tuple(actions))
    seed_all(base_seed)
    env = MetaLearningEnv()
    env.action_space.seed(base_seed)
    rewards: List[float] = []

    try:
        for episode_idx in range(episodes):
            episode_seed = derive_seed(base_seed, "episode", episode_idx)
            seed_all(episode_seed)
            env.reset(seed=episode_seed)
            env.set_test_mode(actions)

            done = False
            reward = 0.0
            while not done:
                _, reward, done, _, _ = env.step(0)

            rewards.append(float(reward))
    finally:
        env.close()

    return rewards


def run_strategy_test_custom_env(env_builder, actions: Sequence[int], *, episodes: int, seed: int) -> List[float]:
    base_seed = derive_seed(seed, "strategy", tuple(actions))
    seed_all(base_seed)
    env = env_builder()
    env.action_space.seed(base_seed)
    rewards: List[float] = []
    try:
        for episode_idx in range(episodes):
            episode_seed = derive_seed(base_seed, "episode", episode_idx)
            seed_all(episode_seed)
            env.reset(seed=episode_seed)
            env.set_test_mode(actions)
            done = False
            reward = 0.0
            while not done:
                _, reward, done, _, _ = env.step(0)
            rewards.append(float(reward))
    finally:
        env.close()
    return rewards


def evaluate_replay_regimes(*, episodes: int, seed: int) -> Dict[str, object]:
    original_steps = Config.STEPS_PER_EPISODE
    original_mmr = Config.MMR
    original_shapley = Config.SHAPLEY

    results: Dict[str, List[Dict[str, object]]] = {scenario.name: [] for scenario in REPLAY_SCENARIOS}

    try:
        for scenario in REPLAY_SCENARIOS:
            Config.STEPS_PER_EPISODE = scenario.steps_per_episode

            for combo_idx, (mmr_flag, shapley_flag) in enumerate(REPLAY_COMBINATIONS):
                Config.MMR = mmr_flag
                Config.SHAPLEY = shapley_flag

                combo_seed = derive_seed(seed, "replay_combo", scenario.name, mmr_flag, shapley_flag)
                seed_all(combo_seed)

                rewards = run_strategy_test(
                    scenario.actions,
                    episodes=episodes,
                    seed=combo_seed,
                )

                summary = _summary_stats(rewards)
                results[scenario.name].append(
                    {
                        "mmr": bool(mmr_flag),
                        "shapley": bool(shapley_flag),
                        "rewards": [float(r) for r in rewards],
                        **summary,
                    }
                )

            # After standard MMR/Shapley combos, append difficulty-based baselines
            # Build an ensemble once (uses train split) for difficulty scoring
            stage_loader = StageLoader()
            rng = np.random.default_rng(derive_seed(seed, "difficulty", scenario.name))
            train_X, train_y = _sample_training_data(stage_loader.train_dataset, sample_size=2000, rng=rng)
            ensemble = _build_ensemble_models(train_X, train_y, rng)

            for mode_label, mode_key in (("Most challenging", "most"), ("Least challenging", "least")):
                def _builder(m=mode_key, ens=ensemble):
                    return DifficultySelectionEnv(ens, mode=m)
                diff_rewards = run_strategy_test_custom_env(_builder, scenario.actions, episodes=episodes, seed=seed)
                diff_summary = _summary_stats(diff_rewards)
                results[scenario.name].append(
                    {
                        "method": mode_label,
                        "rewards": [float(r) for r in diff_rewards],
                        **diff_summary,
                    }
                )
    finally:
        Config.STEPS_PER_EPISODE = original_steps
        Config.MMR = original_mmr
        Config.SHAPLEY = original_shapley

    return {
        "per_scenario": results,
        "episodes": int(episodes),
    }


def _format_flag(label: str, value: bool) -> str:
    return f"{label}={'on' if value else 'off'}"


def _fmt(value: object) -> str:
    if value is None:
        return "nan"
    return f"{value:.4f}"


## Removed printing of Buffer selection strategies per request


def print_replay_summary(replay_results: Dict[str, object]) -> None:
    per_scenario: Dict[str, List[Dict[str, object]]] = replay_results.get("per_scenario", {})
    print("\nReplay regime evaluations:")
    for scenario in REPLAY_SCENARIOS:
        print(f"  Scenario: {scenario.name}")
        entries = per_scenario.get(scenario.name, [])
        if not entries:
            print("    (no data)")
            continue
        for entry in entries:
            if "mmr" in entry and "shapley" in entry:
                flags = f"{_format_flag('MMR', entry.get('mmr', False))}, {_format_flag('Shapley', entry.get('shapley', False))}"
                mean = _fmt(entry.get("mean"))
                std = _fmt(entry.get("std"))
                sem = _fmt(entry.get("sem"))
                count = entry.get("count", 0)
                print(f"    {flags}: mean={mean}, std={std}, sem={sem}, n={count}")
            else:
                label = entry.get("method", "method")
                mean = _fmt(entry.get("mean"))
                std = _fmt(entry.get("std"))
                sem = _fmt(entry.get("sem"))
                count = entry.get("count", 0)
                print(f"    {label}: mean={mean}, std={std}, sem={sem}, n={count}")


def _compute_sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def plot_data_vs_random(buffer_results: Dict[str, object], *, output_dir: Optional[Path] = None) -> Optional[Path]:
    per_strategy = buffer_results.get("per_strategy", {})
    dvn_scores = per_strategy.get("Data valuation network", {}).get("scores", [])
    rnd_scores = per_strategy.get("Random selection", {}).get("scores", [])

    if len(dvn_scores) == 0 or len(rnd_scores) == 0:
        LOGGER.warning("Insufficient data to plot data valuation comparison (dvn=%d, random=%d)",
                       len(dvn_scores), len(rnd_scores))
        return None

    dvn_arr = np.asarray(dvn_scores, dtype=np.float32)
    rnd_arr = np.asarray(rnd_scores, dtype=np.float32)

    groups = ["Data\nvaluation", "Random\nselection"]
    means = [float(np.mean(dvn_arr)), float(np.mean(rnd_arr))]
    errors = [_compute_sem(dvn_arr), _compute_sem(rnd_arr)]

    t_stat, p_val = ttest_ind(dvn_arr, rnd_arr, equal_var=False)
    stars = _significance_stars(float(p_val))

    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    x = np.arange(len(groups))
    ax.bar(x, means, yerr=errors, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Accuracy")

    stage_count = buffer_results.get("stages")
    if stage_count == 1:
        title = "Real data"
        filename = "data_vs_random_real.png"
    elif stage_count == 2:
        title = "Generated data"
        filename = "data_vs_random_generated.png"
    else:
        title = f"{stage_count} stages" if stage_count is not None else "Data valuation"
        filename = f"data_vs_random_{stage_count or 'summary'}.png"

    ax.set_title(title)

    y_max = max(means[idx] + errors[idx] for idx in range(len(groups)))
    y = y_max + 0.03
    ax.hlines(y, x[0], x[1], lw=1.5, color="black")
    ax.text(np.mean(x), y + 0.005, stars, ha="center", va="bottom", fontsize=12)

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * 1.17)

    plt.tight_layout()

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / filename
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved data valuation vs random plot → %s (t=%.3f, p=%.3f, stars=%s)",
                fig_path, t_stat, p_val, stars)
    return fig_path


def _one_tailed_p(two_sided_p: float, mean_a: float, mean_b: float) -> float:
    """Convert a two-sided p-value to one-tailed for H1: mean_a > mean_b."""
    if not np.isfinite(two_sided_p):
        return float("nan")
    if mean_a > mean_b:
        return max(0.0, float(two_sided_p) / 2.0)
    return max(0.0, 1.0 - float(two_sided_p) / 2.0)


def plot_dvn_vs_random_for_scenario(
    per_scenario_results: Dict[str, List[Dict[str, object]]],
    scenario_name: str,
    *,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Create a bar chart comparing DVN (MMR+Shapley) vs Random for a scenario.

    - Uses a one-tailed Welch t-test (H1: DVN > Random) and annotates significance.
    - Saves to dvn_vs_random_{scenario}.png
    """
    entries = per_scenario_results.get(scenario_name, [])
    if not entries:
        LOGGER.warning("No results for scenario '%s'", scenario_name)
        return None

    dvn = next((e for e in entries if e.get("mmr") is True and e.get("shapley") is True), None)
    rnd = next((e for e in entries if e.get("mmr") is False and e.get("shapley") is False), None)
    if dvn is None or rnd is None:
        LOGGER.warning("Missing DVN or Random entry in '%s' (dvn=%s, rnd=%s)", scenario_name, dvn is not None, rnd is not None)
        return None

    dvn_scores = np.asarray(dvn.get("rewards", []), dtype=np.float32)
    rnd_scores = np.asarray(rnd.get("rewards", []), dtype=np.float32)
    if dvn_scores.size == 0 or rnd_scores.size == 0:
        LOGGER.warning("Empty rewards for scenario '%s' (dvn=%d, rnd=%d)", scenario_name, dvn_scores.size, rnd_scores.size)
        return None

    dvn_mean = float(np.mean(dvn_scores))
    rnd_mean = float(np.mean(rnd_scores))
    dvn_sem = _compute_sem(dvn_scores)
    rnd_sem = _compute_sem(rnd_scores)

    t_stat, p_two = ttest_ind(dvn_scores, rnd_scores, equal_var=False)
    p_one = _one_tailed_p(float(p_two), dvn_mean, rnd_mean)
    stars = _significance_stars(p_one)

    groups = ["Data valuation", "Random"]
    means = [dvn_mean, rnd_mean]
    errors = [dvn_sem, rnd_sem]

    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    x = np.arange(len(groups))
    bars = ax.bar(x, means, yerr=errors, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Buffer" if scenario_name == "buffer" else "Generated")

    y_max = max(means[i] + errors[i] for i in range(len(groups)))
    y_annot = y_max + 0.03
    ax.hlines(y_annot, x[0], x[1], lw=1.5, color="black")
    ax.text(np.mean(x), y_annot + 0.005, stars, ha="center", va="bottom", fontsize=12)

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * 1.17)

    plt.tight_layout()

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"dvn_vs_random_{scenario_name}.png"
    fig_path = output_dir / out_name
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info(
        "Saved DVN vs Random plot for %s → %s (t=%.3f, p_one=%.3g, stars=%s)",
        scenario_name, fig_path, t_stat, p_one, stars
    )
    return fig_path


def load_cached_results(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - cache corruption fallback
        LOGGER.warning("Failed to load cached results from %s: %s", path, exc)
        return None


def save_results(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidated data valuation analysis.")
    parser.add_argument("--trials", type=int, default=3, help="Number of buffer resamplings to average over.")
    parser.add_argument(
        "--ensemble-samples",
        type=int,
        default=4000,
        help="Training samples for the difficulty ensemble.",
    )
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed for reproducibility.")
    parser.add_argument(
        "--partial-fit-epochs",
        type=int,
        default=Config.PARTIAL_FIT_EPS,
        help="Epochs for classifier fine-tuning per selection.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=Config.STEPS_PER_EPISODE,
        help="(Deprecated) Memory phases are fixed to three sequential replays.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for computing test accuracy.",
    )
    parser.add_argument(
        "--shapley-iters",
        type=int,
        default=3,
        help="Monte Carlo iterations for Shapley estimation.",
    )
    parser.add_argument(
        "--shapley-fraction",
        type=float,
        default=0.1,
        help="Fraction of buffer samples used for true Shapley labels.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per (MMR, Shapley) combination when evaluating replay regimes.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to cache the aggregated results (JSON).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached results and recompute everything.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    seed_all(args.seed)

    output_path = Path(args.output_path)
    if not args.force:
        cached = load_cached_results(output_path)
        if cached is not None:
            print(f"Loaded cached data valuation results from {output_path}")
            if "replay_regimes" in cached:
                print_replay_summary(cached["replay_regimes"])
                # Generate DVN vs Random plots from cached results as requested
                per_scenario_cached = cached["replay_regimes"].get("per_scenario", {})
                plot_dvn_vs_random_for_scenario(per_scenario_cached, "buffer", output_dir=Path(__file__).resolve().parent)
                plot_dvn_vs_random_for_scenario(per_scenario_cached, "generated", output_dir=Path(__file__).resolve().parent)
            return

    replay_results = evaluate_replay_regimes(
        episodes=args.episodes,
        seed=args.seed,
    )

    combined = {
        "replay_regimes": replay_results,
        "metadata": {
            "args": vars(args),
        },
    }

    save_results(output_path, combined)
    print(f"Saved combined data valuation results to {output_path}")

    print_replay_summary(replay_results)

    # New: DVN (MMR+Shapley) vs Random bar charts for both buffer and generated
    per_scenario = replay_results.get("per_scenario", {})
    plot_dvn_vs_random_for_scenario(per_scenario, "buffer", output_dir=Path(__file__).resolve().parent)
    plot_dvn_vs_random_for_scenario(per_scenario, "generated", output_dir=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

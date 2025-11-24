import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO

from config import Config
from environment import MetaLearningEnv
from seed_utils import derive_seed, seed_all
BASE_DIR = Path(__file__).resolve().parent
MODEL_STEMS = {
    "non_cl": BASE_DIR / "meta_controller_non_cl",
    "cl": BASE_DIR / "meta_controller_cl",
}

STAGE_CONFIGS = {
    "non_cl": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "cl": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
}

POLICY_LABELS = ["metacontroller", "memory_replay", "world_model_replay"]
DISPLAY_LABELS = {
    "metacontroller": "Metacontroller",
    "memory_replay": "Memory replay only",
    "world_model_replay": "Generative replay only",
}


def _apply_setting(setting: str) -> Dict[str, object]:
    original = {
        "stages": deepcopy(Config.STAGES),
        "shapley": Config.SHAPLEY,
        "mmr": Config.MMR,
    }
    Config.STAGES = deepcopy(STAGE_CONFIGS[setting])
    Config.SHAPLEY = False
    Config.MMR = False
    return original


def _restore_setting(original: Dict[str, object]) -> None:
    Config.STAGES = original["stages"]
    Config.SHAPLEY = original["shapley"]
    Config.MMR = original["mmr"]


def _require_metacontroller(setting: str) -> Path:
    model_stem = MODEL_STEMS[setting]
    model_zip = model_stem.with_suffix(".zip")
    if not model_zip.exists():
        raise FileNotFoundError(
            f"Missing trained metacontroller for '{setting}'. "
            "Run `python image/experiment.py --regime {setting}` first."
        )
    return model_stem


def _evaluate_metacontroller(model_stem: Path, episodes: int, seed_value: int) -> np.ndarray:
    model = RecurrentPPO.load(str(model_stem))
    resolved_seed = seed_value
    seed_all(resolved_seed)
    env = MetaLearningEnv()
    env.action_space.seed(resolved_seed)
    results: List[float] = []

    wrapped_env = gym.wrappers.FlattenObservation(env)
    stages_to_run = len(Config.STAGES) if len(Config.STAGES) else 0
    if stages_to_run == 0:
        return np.array(results, dtype=np.float32)

    for eval_idx in range(episodes):
        for stage_idx in range(stages_to_run):
            # Use a shared tag so metacontroller and fixed-policy runs share identical seeds
            stage_seed = derive_seed(resolved_seed, "baseline_eval", eval_idx, stage_idx)
            seed_all(stage_seed)
            obs, _ = wrapped_env.reset(seed=stage_seed)
            state = None
            episode_start = np.array([True], dtype=bool)
            done = False
            reward = 0.0

            while not done:
                action, state = model.predict(
                    obs,
                    state=state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                action_array = np.asarray(action)
                action_int = int(action_array.flatten()[0]) if action_array.size else int(action)
                obs_raw, reward, done, _, _ = env.step(action_int)
                episode_start = np.array([done], dtype=bool)
                if not done:
                    obs = wrapped_env.observation(obs_raw)

            results.append(float(reward))

    return np.array(results, dtype=np.float32)


def _evaluate_fixed_policy(actions: List[int], episodes: int, seed_value: int) -> np.ndarray:
    resolved_seed = seed_value
    seed_all(resolved_seed)
    env = MetaLearningEnv()
    env.action_space.seed(resolved_seed)
    results: List[float] = []
    stages_to_run = len(Config.STAGES) if len(Config.STAGES) else 0
    if stages_to_run == 0:
        return np.array(results, dtype=np.float32)

    for eval_idx in range(episodes):
        for stage_idx in range(stages_to_run):
            # Use the same tag as metacontroller path for identical seed schedule
            stage_seed = derive_seed(resolved_seed, "baseline_eval", eval_idx, stage_idx)
            seed_all(stage_seed)
            env.reset(seed=stage_seed)
            env.set_test_mode(actions)
            done = False
            reward = 0.0

            while not done:
                _, reward, done, _, _ = env.step(0)

            results.append(float(reward))

    return np.array(results, dtype=np.float32)


def _memory_replay_actions() -> List[int]:
    return [0] * Config.STEPS_PER_EPISODE


def _world_model_actions() -> List[int]:
    if Config.STEPS_PER_EPISODE == 0:
        return []
    return [1] + [2] * (Config.STEPS_PER_EPISODE - 1)


def run_comparison() -> Dict[str, Dict[str, np.ndarray]]:
    logging.info("Running baseline comparison using pre-trained controllers")
    summary: Dict[str, Dict[str, np.ndarray]] = {}

    for setting in ("non_cl", "cl"):
        logging.info("\n=== %s regime ===", setting.upper())
        original_cfg = _apply_setting(setting)

        try:
            model_stem = _require_metacontroller(setting)
            # Use 25 episodes for non-CL so n matches CL (5 stages × 5 episodes = 25)
            episodes = 25 if setting == "non_cl" else max(len(Config.STAGES), 3)
            seed_value = derive_seed(Config.SEED, "baseline", setting)
            meta_scores = _evaluate_metacontroller(model_stem, episodes, seed_value)
            memory_scores = _evaluate_fixed_policy(_memory_replay_actions(), episodes, seed_value)
            world_model_scores = _evaluate_fixed_policy(_world_model_actions(), episodes, seed_value)

            summary[setting] = {
                "metacontroller": meta_scores,
                "memory_replay": memory_scores,
                "world_model_replay": world_model_scores,
            }
        finally:
            _restore_setting(original_cfg)

    return summary


def _print_summary(results: Dict[str, Dict[str, np.ndarray]]) -> None:
    means: Dict[str, Dict[str, float]] = {}
    for setting, scores in results.items():
        means[setting] = {}
        for label in POLICY_LABELS:
            mean_score = float(scores[label].mean()) if len(scores[label]) else float("nan")
            means[setting][label] = mean_score
            logging.info("%s regime – %s: %.4f", setting.upper(), DISPLAY_LABELS[label], mean_score)

    header = [
        "Approach",
        "Accuracy (CL)",
        "Accuracy (non-CL)",
        "Mean accuracy",
        "SEM (mean)",
    ]
    rows: List[List[str]] = [header]

    for label in POLICY_LABELS:
        # Pull per-episode arrays to compute SEMs
        cl_vals = results.get("cl", {}).get(label, np.array([], dtype=np.float32))
        non_cl_vals = results.get("non_cl", {}).get(label, np.array([], dtype=np.float32))

        cl_score = float(cl_vals.mean()) if len(cl_vals) else float("nan")
        non_cl_score = float(non_cl_vals.mean()) if len(non_cl_vals) else float("nan")
        mean_acc = float(np.nanmean([cl_score, non_cl_score]))

        def _sem(arr: np.ndarray) -> float:
            n = len(arr)
            if n <= 1:
                return 0.0 if n == 1 else float("nan")
            return float(np.std(arr, ddof=1) / np.sqrt(n))

        sem_cl = _sem(cl_vals) if len(cl_vals) else float("nan")
        sem_noncl = _sem(non_cl_vals) if len(non_cl_vals) else float("nan")
        if np.isfinite(sem_cl) and np.isfinite(sem_noncl):
            sem_mean = float(np.sqrt(sem_cl ** 2 + sem_noncl ** 2) / 2.0)
        elif np.isfinite(sem_cl):
            sem_mean = sem_cl
        elif np.isfinite(sem_noncl):
            sem_mean = sem_noncl
        else:
            sem_mean = float("nan")

        rows.append([
            DISPLAY_LABELS[label],
            f"{cl_score:.3f}",
            f"{non_cl_score:.3f}",
            f"{mean_acc:.3f}",
            f"{sem_mean:.3f}" if np.isfinite(sem_mean) else "nan",
        ])

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    logging.info("\nFinal accuracy summary:")
    for row in rows:
        logging.info(
            "%s\t%s\t%s\t%s\t%s",
            row[0].ljust(col_widths[0]),
            row[1].rjust(col_widths[1]),
            row[2].rjust(col_widths[2]),
            row[3].rjust(col_widths[3]),
            row[4].rjust(col_widths[4]),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = run_comparison()
    _print_summary(results)

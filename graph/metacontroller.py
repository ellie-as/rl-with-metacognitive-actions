from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList

from family_tree_task import FamilyTreeBuilder
from spatial_task import SpatialBuilder
from callbacks import GraphLoggerExtended, ActionProbabilityLoggerExtended
from environment import GraphEnv
from seed_utils import derive_seed, seed_all
from visualisation import (
    inference_bar_plot_across_runs,
    plot_aggregated_action_probs,
    plot_aggregated_rewards,
    plot_combined_2row4col,
)


LOGGER = logging.getLogger("graph.metacontroller")
DEFAULT_TOTAL_TIMESTEPS = 150_000
DEFAULT_RUNS = 1
DEFAULT_MAX_META_STEPS = 10
DEFAULT_INITIAL_OBS = 20
DEFAULT_BASE_SEED = 42

BUILDER_REGISTRY: Dict[str, callable] = {
    "family": lambda: FamilyTreeBuilder(base_num_children=2, grandparent_num_children=2),
    "spatial": lambda: SpatialBuilder(),
}


def _make_env(builder, output_dir: Path, max_meta_steps: int, initial_obs_count: int, seed_value: int) -> GraphEnv:
    env = GraphEnv(builder, output_dir=output_dir, max_meta_steps=max_meta_steps, initial_obs_count=initial_obs_count)
    env.action_space.seed(seed_value)
    env.reset(seed=seed_value)
    return env


def train_single_run(
    *,
    builder,
    run_id: int,
    total_timesteps: int,
    max_meta_steps: int,
    initial_obs_count: int,
    base_seed: int,
    results_dir: Path,
) -> Tuple[RecurrentPPO, np.ndarray, np.ndarray, pd.DataFrame]:
    run_seed = derive_seed(base_seed, builder.__class__.__name__, run_id)
    LOGGER.info("Run %d seed: %s", run_id, run_seed)
    seed_all(run_seed)

    if hasattr(builder, "seed"):
        builder.seed(run_seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        device = "cuda"
    else:
        device = "cpu"

    env_seed = derive_seed(run_seed, "train_env")
    env = _make_env(builder, results_dir, max_meta_steps, initial_obs_count, env_seed)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=0,
        ent_coef=0.05,
        learning_rate=1e-4,
        policy_kwargs={
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
        },
        device=device,
        seed=run_seed,
    )

    graph_logger = GraphLoggerExtended(run_id, str(results_dir), eval_freq=1000, verbose=0)
    action_logger = ActionProbabilityLoggerExtended(
        run_id,
        str(results_dir),
        n_actions=env.action_space.n,
        max_episode_steps=max_meta_steps,
        verbose=0,
    )

    callback = CallbackList([graph_logger, action_logger])
    model.learn(total_timesteps=total_timesteps, callback=callback)

    steps, rewards = graph_logger.get_data()
    action_df = action_logger.get_data_long_form()
    action_df["run_id"] = run_id

    model.save(str(results_dir / f"run_{run_id}_model"))
    return model, steps, rewards, action_df


def aggregate_runs(
    builder,
    *,
    n_runs: int,
    total_timesteps: int,
    max_meta_steps: int,
    initial_obs_count: int,
    base_seed: int,
    results_dir: Path,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], pd.DataFrame]:
    # Keep only model file paths to avoid retaining training envs in memory
    models: List[str] = []
    steps_list: List[np.ndarray] = []
    rewards_list: List[np.ndarray] = []
    action_frames: List[pd.DataFrame] = []

    for run_id in range(1, n_runs + 1):
        LOGGER.info("=== %s | Run %d/%d ===", builder.__class__.__name__, run_id, n_runs)
        model, steps, rewards, action_df = train_single_run(
            builder=builder,
            run_id=run_id,
            total_timesteps=total_timesteps,
            max_meta_steps=max_meta_steps,
            initial_obs_count=initial_obs_count,
            base_seed=base_seed,
            results_dir=results_dir,
        )
        # Persist and free training resources: close env and keep only the model path
        model_path = str(results_dir / f"run_{run_id}_model.zip")
        try:
            env_vec = model.get_env()
            if env_vec is not None:
                env_vec.close()
        except Exception:
            pass
        # Drop reference to the model (and its env) to prevent memory growth across runs
        del model

        models.append(model_path)
        steps_list.append(steps)
        rewards_list.append(rewards)
        action_frames.append(action_df)

    combined_actions = pd.concat(action_frames, ignore_index=True)
    return models, steps_list, rewards_list, combined_actions


def _selected_builders(selection: Sequence[str]) -> List[str]:
    if not selection:
        return list(BUILDER_REGISTRY.keys())
    if "all" in selection:
        return list(BUILDER_REGISTRY.keys())
    missing = [name for name in selection if name not in BUILDER_REGISTRY]
    if missing:
        raise ValueError(f"Unknown builder(s): {missing}. Choices: {tuple(BUILDER_REGISTRY)}")
    return list(selection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train graph metacontrollers with reproducible seeds")
    parser.add_argument(
        "--builder",
        nargs="*",
        default=("family", "spatial"),
        help="Which builders to run (default: family spatial). Use 'all' for every builder.",
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--meta-steps", type=int, default=DEFAULT_MAX_META_STEPS)
    parser.add_argument("--initial-obs", type=int, default=DEFAULT_INITIAL_OBS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting after training")
    parser.add_argument("--no-eval", action="store_true", help="Skip inference evaluation plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    selected = _selected_builders(args.builder)

    for builder_key in selected:
        builder = BUILDER_REGISTRY[builder_key]()
        builder_seed = derive_seed(args.seed, builder_key)
        results_dir = Path(f"{builder.__class__.__name__}_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        models, steps_list, rewards_list, action_df = aggregate_runs(
            builder,
            n_runs=args.runs,
            total_timesteps=args.timesteps,
            max_meta_steps=args.meta_steps,
            initial_obs_count=args.initial_obs,
            base_seed=builder_seed,
            results_dir=results_dir,
        )

        if not args.no_plots:
            plot_aggregated_rewards(
                steps_list,
                rewards_list,
                window=600,
                output_path=results_dir / "aggregated_rewards.png",
            )
            plot_aggregated_action_probs(
                action_df=action_df,
                n_actions=6,
                max_episode_steps=args.meta_steps,
                window=600,
                output_dir=results_dir,
            )
            plot_combined_2row4col(
                builder_name=builder.__class__.__name__,
                steps_list=steps_list,
                rewards_list=rewards_list,
                action_df=action_df,
                n_actions=6,
                max_episode_steps=args.meta_steps,
                window=600,
            )

        if not args.no_eval:
            eval_seed = derive_seed(builder_seed, "inference")
            inference_bar_plot_across_runs(
                builder=builder,
                models=models,
                n_eval_episodes=200,
                max_meta_steps=args.meta_steps,
                seed_value=eval_seed,
            )

        LOGGER.info("Finished builder %s", builder.__class__.__name__)


if __name__ == "__main__":
    main()

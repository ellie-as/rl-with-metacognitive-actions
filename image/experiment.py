import argparse
import csv
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import MetaLearningEnv
from config import Config
from seed_utils import derive_seed, seed_all

setattr(Config, "SHAPLEY", False)
setattr(Config, "MMR",     False)
setattr(Config, "LEVELS",  (0.0,))

REGIME_STAGE_CONFIGS = {
    "non_cl": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "cl": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
}

DATASET_TAG = ("mnist" if getattr(Config, "DATASET", "fashion_mnist").lower() == "mnist" else "fashion")
LOG_DIR = Path(__file__).resolve().parent / f"logs/{DATASET_TAG}"
PLOT_ROLL_WINDOW = 100
ACTION_COLUMNS = ["prob_action_0", "prob_action_1", "prob_action_2"]
ACTION_LABELS = ["Memory replay", "Update WM", "Generative replay"]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


class ActionProbLogger(BaseCallback):
    """
    Records π(a|s) for env-0 at every step and streams the data to CSV.

    A header is written at training start; afterwards rows are appended
    every `save_every` steps so that you can inspect the file while the
    run is still in progress.

        episode_index, step_in_episode, prob_a0, prob_a1, prob_a2
    """
    def __init__(
        self,
        csv_path: str,
        save_every: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.csv_path   = Path(csv_path)
        self.save_every = save_every
        self.buffer: list[list[float]] = []   # rows waiting to be flushed
        self.ep_idx     = 0
        self.step_in_ep = 0

    def _on_training_start(self) -> None:
        # create parent dir + write header once
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["episode_index", "step_in_episode",
                  "prob_action_0", "prob_action_1", "prob_action_2"]
        with self.csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(header)

    def _on_step(self) -> bool:
        obs_tensor     = self.locals["obs_tensor"]          # (n_env, obs_dim)
        lstm_states    = self.locals.get("lstm_states")
        episode_starts = self.locals.get("episode_starts")
        dones          = self.locals["dones"]

        # unwrap lstm state expected by RecurrentPolicy
        if lstm_states is not None and isinstance(lstm_states[0], tuple):
            lstm_states = lstm_states[0]

        obs_0          = obs_tensor[0:1]
        episode_start0 = episode_starts[0:1] if episode_starts is not None else None

        with torch.no_grad():
            dist, _ = self.model.policy.get_distribution(
                obs_0, lstm_states, episode_start0
            )
            probs = dist.distribution.probs.cpu().numpy()[0]   # (3,)

        # buffer the row
        self.buffer.append([self.ep_idx, self.step_in_ep, *probs.tolist()])

        # flush every `save_every` steps
        if self.num_timesteps % self.save_every == 0:
            self._flush()

        # bookkeeping
        self.step_in_ep += 1
        if dones[0]:
            self.ep_idx    += 1
            self.step_in_ep = 0
        return True

    def _flush(self) -> None:
        if not self.buffer:
            return
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerows(self.buffer)
        if self.verbose:
            print(f"[ActionProbLogger] flushed {len(self.buffer)} rows "
                  f"@ t={self.num_timesteps}")
        self.buffer.clear()

    def _on_training_end(self) -> None:
        self._flush()   # write anything still in memory
        if self.verbose:
            print(f"[ActionProbLogger] finished. CSV → {self.csv_path.resolve()}")


class EarlyStoppingCallback(BaseCallback):
    """Stop training when recent episode rewards plateau or hit a threshold."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-3,
        window_size: int = 5,
        target_reward: Optional[float] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.patience = max(1, patience)
        self.min_delta = min_delta
        self.window_size = max(1, window_size)
        self.target_reward = target_reward
        self.recent_returns: list[float] = []
        self.wait = 0
        self.best_mean = -np.inf
        self.episode_returns: Optional[np.ndarray] = None

    def _on_training_start(self) -> None:
        if self.training_env is None:
            raise RuntimeError("EarlyStoppingCallback requires a VecEnv.")
        n_envs = self.training_env.num_envs
        self.episode_returns = np.zeros(n_envs, dtype=np.float64)
        self.recent_returns.clear()
        self.wait = 0
        self.best_mean = -np.inf

    def _on_step(self) -> bool:
        if self.episode_returns is None:
            return True

        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        self.episode_returns += rewards

        stop_now = False
        for idx, done in enumerate(dones):
            if not done:
                continue

            episode_return = float(self.episode_returns[idx])
            self.episode_returns[idx] = 0.0
            self.recent_returns.append(episode_return)
            if len(self.recent_returns) > self.window_size:
                self.recent_returns.pop(0)

            mean_return = float(np.mean(self.recent_returns))
            self.model.logger.record("early_stop/mean_recent_reward", mean_return)

            if self.target_reward is not None and mean_return >= self.target_reward:
                stop_now = True
                break

            if mean_return > self.best_mean + self.min_delta:
                self.best_mean = mean_return
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    stop_now = True
                    break

        if stop_now:
            if self.verbose:
                print("[EarlyStopping] Triggered early stop.")
            return False
        return True


def _make_lr_schedule(
    start_lr: float = 1e-4,
    target_lr: float = 3e-4,
    warmup_fraction: float = 0.2,
) -> Callable[[float], float]:
    """Linear warmup schedule that ramps to `target_lr` over initial fraction."""

    warmup_fraction = max(1e-6, min(warmup_fraction, 1.0))

    def schedule(progress_remaining: float) -> float:
        progress_elapsed = 1.0 - progress_remaining
        if progress_elapsed < warmup_fraction:
            ratio = progress_elapsed / warmup_fraction
            return start_lr + ratio * (target_lr - start_lr)
        return target_lr

    return schedule


def train_meta_controller(
    *,
    model_path: str = "meta_controller",
    total_timesteps: int = 5000,
    action_log_path: Optional[str] = None,
    seed_value: Optional[int] = None,
) -> RecurrentPPO:
    resolved_seed = seed_value or Config.SEED
    seed_all(resolved_seed)

    env_seed = derive_seed(resolved_seed, "train_env")

    def _make_env() -> gym.Env:
        env = MetaLearningEnv()
        env.action_space.seed(env_seed)
        return gym.wrappers.FlattenObservation(env)

    env = DummyVecEnv([_make_env])

    if action_log_path is None:
        action_log_path = "./logs/action_probs_train.csv"

    callbacks = CallbackList([
        ActionProbLogger(action_log_path, verbose=1),
    ])

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        device = "cuda"
    else:
        device = "cpu"

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        n_steps=128,
        ent_coef=0.01,
        learning_rate=_make_lr_schedule(),
        policy_kwargs={
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
        },
        device=device,
        seed=resolved_seed,
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(model_path)
    return model


def evaluate_controller(model: RecurrentPPO, num_episodes: int = 3, seed_value: Optional[int] = None) -> None:
    if seed_value is not None:
        seed_all(seed_value)

    resolved_seed = seed_value or Config.SEED
    env = MetaLearningEnv()                     # *unwrapped* env
    env.action_space.seed(resolved_seed)

    for ep in range(num_episodes):
        wrapped_env = gym.wrappers.FlattenObservation(env)
        episode_seed = derive_seed(resolved_seed, "eval_episode", ep)
        seed_all(episode_seed)
        obs, _ = wrapped_env.reset(seed=episode_seed)
        done = False
        ep_rewards, action_counts = [], [0, 0, 0]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs_raw, reward, done, _, _ = env.step(action)
            obs = wrapped_env.observation(obs_raw)

            ep_rewards.append(reward)
            action_counts[action] += 1

        total_r = sum(ep_rewards)
        final_acc = env.classifier.evaluate(env.test_loader)
        logging.info(f"Episode {ep+1}: reward={total_r:.4f}, "
                     f"acc={final_acc:.4f}, "
                     f"actions={action_counts}")


def train_regime(
    regime: str,
    *,
    total_timesteps: int = 5000,
    evaluate: bool = True,
) -> RecurrentPPO:
    if regime not in REGIME_STAGE_CONFIGS:
        raise ValueError(f"Unknown regime '{regime}'. Expected one of {tuple(REGIME_STAGE_CONFIGS)}")

    original = {
        "stages": deepcopy(Config.STAGES),
        "shapley": Config.SHAPLEY,
        "mmr": Config.MMR,
    }

    Config.STAGES = deepcopy(REGIME_STAGE_CONFIGS[regime])
    Config.SHAPLEY = False
    Config.MMR = False

    model_path = Path(__file__).resolve().parent / f"{DATASET_TAG}/meta_controller_{regime}"
    action_log_path = LOG_DIR / f"action_probs_{regime}.csv"

    regime_seed = derive_seed(Config.SEED, "train_regime", regime)
    seed_all(regime_seed)

    try:
        model = train_meta_controller(
            model_path=str(model_path),
            total_timesteps=total_timesteps,
            action_log_path=str(action_log_path),
            seed_value=regime_seed,
        )
        if evaluate:
            episodes = max(len(Config.STAGES), 3)
            eval_seed = derive_seed(regime_seed, "evaluation")
            evaluate_controller(model, num_episodes=episodes, seed_value=eval_seed)
        return model
    finally:
        Config.STAGES = original["stages"]
        Config.SHAPLEY = original["shapley"]
        Config.MMR = original["mmr"]


def create_action_probability_plot(regime: str) -> bool:
    """Replicates the Create plots.ipynb visualisation for a given regime."""

    csv_path = LOG_DIR / f"action_probs_{regime}.csv"
    if not csv_path.exists():
        logging.warning("No action probability log found for regime '%s' (expected %s)",
                        regime, csv_path)
        return False

    df = pd.read_csv(csv_path)
    if df.empty:
        logging.warning("CSV %s is empty; skipping plot generation", csv_path)
        return False

    step_ids = sorted(df["step_in_episode"].unique())
    if not step_ids:
        logging.warning("No step identifiers present in %s; skipping", csv_path)
        return False

    fig, axes = plt.subplots(1, len(step_ids), figsize=(3 * len(step_ids), 2.5), sharey=True)
    if len(step_ids) == 1:
        axes = [axes]

    for ax, step in zip(axes, step_ids):
        sub = df[df["step_in_episode"] == step].reset_index(drop=True)
        roll = sub[ACTION_COLUMNS].rolling(PLOT_ROLL_WINDOW, min_periods=1)
        mean = roll.mean()
        # Rolling standard deviation within the same window; NaNs (e.g. for the
        # very first point) are set to 0 so the band collapses to the mean there.
        std = roll.std().fillna(0.0)

        for idx, col in enumerate(ACTION_COLUMNS):
            ax.plot(mean.index, mean[col], label=ACTION_LABELS[idx])
            upper = mean[col] + std[col]
            lower = mean[col] - std[col]
            ax.fill_between(mean.index, lower, upper, alpha=0.2)

        ax.set_title(f"Step {step} in episode")
        ax.set_xlabel("Training episode")

    axes[0].set_ylabel("Probability")
    if len(axes) > 1:
        legend_axis = axes[1]
        legend_axis.legend(loc="upper right", bbox_to_anchor=(0.5, -0.3), ncols=3)
    else:
        axes[0].legend(loc="upper right", ncols=3)
    plt.tight_layout()

    out_name = "actions over time cl.png" if regime == "cl" else "actions over time non cl.png"
    out_path = Path(__file__).resolve().parent / f"{DATASET_TAG}/{out_name}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved action probability plot → %s", out_path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train metacontroller regimes")
    parser.add_argument(
        "--regime",
        choices=("non_cl", "cl", "all"),
        default="all",
        help="Which regime to train. Default 'all' trains both sequentially.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only generate plots from existing logs without training.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training.",
    )
    args = parser.parse_args()

    seed_all(Config.SEED)

    regimes = ("cl", "non_cl") if args.regime == "all" else (args.regime,)

    # If requested, skip training entirely and only (re)generate plots from
    # any existing logged action probabilities.
    if args.plots_only:
        for reg in regimes:
            logging.info("Generating plots for regime '%s' from existing logs.", reg)
            create_action_probability_plot(reg)
        raise SystemExit(0)

    timesteps_for_regime = {
        "cl": 20000,
        "non_cl": 2000,
    }
    generated = []
    for reg in regimes:
        logging.info("Training regime: %s", reg)
        train_regime(reg, total_timesteps=timesteps_for_regime[reg], evaluate=not args.no_eval)
        generated.append(reg)

    for reg in generated:
        create_action_probability_plot(reg)

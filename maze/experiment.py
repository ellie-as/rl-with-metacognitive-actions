import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config
from meta_env import MetaEnv
from seed_utils import derive_seed, seed_all


LOG_DIR = Path(__file__).resolve().parent / "logs"
MODEL_PATH = Path(__file__).resolve().parent / "meta_controller"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


class ActionProbLogger(BaseCallback):
    """Stream π(a|s) probabilities to CSV during training."""

    def __init__(
        self,
        csv_path: Path,
        save_every: int = 20,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.save_every = max(1, int(save_every))
        self.buffer: list[list[float]] = []
        self.ep_idx = 0
        self.step_in_ep = 0

    def _on_training_start(self) -> None:
        # Reset internal state in case the callback instance is reused.
        self.ep_idx = 0
        self.step_in_ep = 0
        self.buffer.clear()

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "episode_index",
            "step_in_episode",
            "prob_action_0",
            "prob_action_1",
            "prob_action_2",
        ]
        with self.csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(header)

    def _flush(self) -> None:
        if not self.buffer:
            return
        with self.csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerows(self.buffer)
        if self.verbose:
            logging.info("[ActionProbLogger] flushed %d rows", len(self.buffer))
        self.buffer.clear()

    def _on_step(self) -> bool:
        obs_tensor = self.locals["obs_tensor"]
        lstm_states = self.locals.get("lstm_states")
        episode_starts = self.locals.get("episode_starts")
        dones = self.locals["dones"]

        if lstm_states is not None and isinstance(lstm_states[0], tuple):
            lstm_states = lstm_states[0]

        obs_0 = obs_tensor[0:1]
        episode_start0 = episode_starts[0:1] if episode_starts is not None else None

        with torch.no_grad():
            dist, _ = self.model.policy.get_distribution(obs_0, lstm_states, episode_start0)
            probs = dist.distribution.probs.cpu().numpy()[0]

        self.buffer.append([self.ep_idx, self.step_in_ep, *probs.tolist()])

        if self.num_timesteps % self.save_every == 0:
            self._flush()

        self.step_in_ep += 1
        if dones[0]:
            self.ep_idx += 1
            self.step_in_ep = 0
        return True

    def _on_training_end(self) -> None:
        self._flush()
        if self.verbose:
            logging.info("[ActionProbLogger] finished. CSV → %s", self.csv_path)


class EarlyStoppingCallback(BaseCallback):
    """Stop training when recent returns plateau or reach a target."""

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
                logging.info("[EarlyStopping] Triggered early stop.")
            return False
        return True


def _make_lr_schedule(
    start_lr: float = 1e-4,
    target_lr: float = 3e-4,
    warmup_fraction: float = 0.2,
) -> Callable[[float], float]:
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
    model_path: Path = MODEL_PATH,
    total_timesteps: int = 5000,
    action_log_path: Optional[Path] = None,
    seed_value: Optional[int] = None,
) -> RecurrentPPO:
    resolved_seed = seed_value or Config.SEED
    seed_all(resolved_seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        device = "cuda"
    else:
        device = "cpu"

    env_seed = derive_seed(resolved_seed, "train_env")

    def _make_env() -> gym.Env:
        env = MetaEnv(mode=Config.MODE)
        env.action_space.seed(env_seed)
        env.reset(seed=env_seed)
        return gym.wrappers.FlattenObservation(env)

    vec_env = DummyVecEnv([_make_env])

    if action_log_path is None:
        action_log_path = LOG_DIR / "action_probs_train.csv"
    else:
        action_log_path = Path(action_log_path)

    callbacks = CallbackList([
        ActionProbLogger(action_log_path, verbose=1),
    ])

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        n_steps=256,
        ent_coef=0.05,
        learning_rate=_make_lr_schedule(),
        policy_kwargs={
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
        },
        device=device,
        seed=resolved_seed,
        normalize_advantage=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(model_path))
    return model


def evaluate_controller(
    model: RecurrentPPO,
    *,
    num_episodes: int = 3,
    seed_value: Optional[int] = None,
) -> None:
    resolved_seed = seed_value or Config.SEED
    seed_all(resolved_seed)

    env = MetaEnv(mode=Config.MODE)
    env.action_space.seed(resolved_seed)

    for ep in range(num_episodes):
        wrapped_env = gym.wrappers.FlattenObservation(env)
        episode_seed = derive_seed(resolved_seed, "eval_episode", ep)
        seed_all(episode_seed)
        obs, _ = wrapped_env.reset(seed=episode_seed)
        done = False
        ep_rewards: list[float] = []
        action_counts = [0, 0, 0]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_array = np.asarray(action)
            action_int = int(action_array.flatten()[0]) if action_array.size else int(action)
            obs_raw, reward, done, _, _ = env.step(action_int)
            obs = wrapped_env.observation(obs_raw)

            ep_rewards.append(reward)
            action_counts[action_int] += 1

        total_r = sum(ep_rewards)
        logging.info(
            "Episode %d: reward=%.4f, actions=%s",
            ep + 1,
            total_r,
            action_counts,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the maze metacontroller")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=8000,
        help="Training timesteps for PPO",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=max(Config.RESET_INTERVAL, 3),
        help="Evaluation episodes after training",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(Config.SEED)

    # Use random valuation mode during training and ensure the world model
    # uses the cached transition variant rather than the 'debug' helper.
    setattr(Config, "MODE", "random")
    setattr(Config, "WORLD_MODEL_TYPE", "cache")

    train_seed = derive_seed(Config.SEED, "maze_train")
    model = train_meta_controller(
        model_path=MODEL_PATH,
        total_timesteps=args.timesteps,
        action_log_path=LOG_DIR / "action_probs_train.csv",
        seed_value=train_seed,
    )

    if not args.no_eval:
        eval_seed = derive_seed(train_seed, "evaluation")
        evaluate_controller(model, num_episodes=args.episodes, seed_value=eval_seed)


if __name__ == "__main__":
    main()

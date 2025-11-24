from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency for the SB3-backed agent.
    from stable_baselines3 import DQN as SB3DQN  # type: ignore[import]
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure as sb3_configure_logger
except Exception:  # pragma: no cover
    SB3DQN = None
    BaseCallback = None
    sb3_configure_logger = None

from config import Config


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    discount: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    hidden_size: int = 128  # retained for compatibility (unused by SB3)
    sb3_kwargs: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


class BaseDQNAgent:
    """Minimal base class maintained for backward compatibility."""

    def __init__(self, cfg: Optional[DQNConfig] = None) -> None:
        self.cfg = cfg or DQNConfig()

    def set_env(self, env) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def get_env(self):  # pragma: no cover - interface only
        raise NotImplementedError

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Any]]:
        raise NotImplementedError

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError

    def learn_episodes(self, total_episodes: int, max_episode_steps: int) -> None:
        # Default fallback: approximate via total timesteps.
        total_timesteps = int(total_episodes) * int(max_episode_steps)
        self.learn(total_timesteps=total_timesteps)

    def learn_from_episodes(self, episodes: Sequence[Sequence[Any]], total_episodes: Optional[int] = None) -> None:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_parameters(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def reset_epsilon(self, epsilon: Optional[float] = None) -> None:
        if epsilon is not None:
            self.cfg.epsilon = float(epsilon)

    def clear_replay_buffer(self) -> None:  # pragma: no cover
        return


class SB3DQNAgent(BaseDQNAgent):
    """Stable-Baselines3-backed DQN agent."""

    def __init__(self, env, cfg: Optional[DQNConfig] = None) -> None:
        if SB3DQN is None:  # pragma: no cover - exercised only when SB3 missing
            raise ImportError("stable-baselines3 is required for AGENT_TYPE='sb3_dqn'.")
        super().__init__(cfg)
        self.model: SB3DQN = self._build_model(env)
        self._env = env

    def _build_model(self, env):
        sb3_kwargs = dict(self.cfg.sb3_kwargs or {})
        defaults = {
            "learning_rate": float(self.cfg.learning_rate),
            "gamma": float(self.cfg.discount),
            "exploration_initial_eps": float(self.cfg.epsilon),
            "exploration_final_eps": float(self.cfg.min_epsilon),
            "exploration_fraction": sb3_kwargs.pop("exploration_fraction", 0.3),
            "learning_starts": sb3_kwargs.pop("learning_starts", 0),
            "target_update_interval": sb3_kwargs.pop("target_update_interval", 500),
            "train_freq": sb3_kwargs.pop("train_freq", 1),
            "gradient_steps": sb3_kwargs.pop("gradient_steps", 1),
            "verbose": sb3_kwargs.pop("verbose", 0),
            "device": sb3_kwargs.pop("device", "cpu"),
        }
        if self.cfg.seed is not None:
            defaults["seed"] = int(self.cfg.seed)
        defaults.update(sb3_kwargs)
        if getattr(Config, "FIX_EPSILON", False):
            defaults["exploration_initial_eps"] = float(self.cfg.epsilon)
            defaults["exploration_final_eps"] = float(self.cfg.epsilon)
            defaults["exploration_fraction"] = 0.0
        model = SB3DQN("MlpPolicy", env, **defaults)
        # Some SB3 versions / deepcopy paths may drop the internal logger.
        # Ensure a logger exists to avoid AttributeError on model.logger.
        if not hasattr(model, "_logger") and sb3_configure_logger is not None:  # pragma: no cover
            model._logger = sb3_configure_logger(None, ["stdout"])
        return model

    def set_env(self, env) -> None:
        if env is None:
            return
        self.model.set_env(env)
        self._env = env

    def get_env(self):
        return self._env

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Any]]:
        action, state = self.model.predict(obs, deterministic=deterministic)
        if np.isscalar(action):
            action_arr = np.array([int(action)], dtype=np.int64)
        else:
            action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        return action_arr, state

    def learn(self, total_timesteps: int) -> None:
        if total_timesteps <= 0:
            return
        self.model.learn(total_timesteps=int(total_timesteps), reset_num_timesteps=False, progress_bar=False)
        if getattr(Config, "FIX_EPSILON", False):
            self.reset_epsilon()

    def learn_episodes(self, total_episodes: int, max_episode_steps: int) -> None:
        if total_episodes <= 0:
            return
        if BaseCallback is None:
            super().learn_episodes(total_episodes, max_episode_steps)
            return
        callback = EpisodeLimitCallback(total_episodes)
        total_timesteps = int(total_episodes) * int(max_episode_steps)
        self.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=callback,
        )
        if getattr(Config, "FIX_EPSILON", False):
            self.reset_epsilon()

    def learn_from_episodes(self, episodes: Sequence[Sequence[Any]], total_episodes: Optional[int] = None) -> None:
        if not episodes:
            return
        if total_episodes is None or total_episodes <= 0:
            selected = episodes
        else:
            total_episodes = int(total_episodes)
            if total_episodes >= len(episodes):
                selected = episodes
            else:
                indices = np.random.choice(len(episodes), size=total_episodes, replace=False)
                selected = [episodes[int(idx)] for idx in indices]

        transitions = 0
        for episode in selected:
            for (obs, action, reward, next_obs, done, _info) in episode:
                act_arr = np.array([action], dtype=np.int64)
                rew_arr = np.array([reward], dtype=np.float32)
                done_arr = np.array([bool(done)], dtype=bool)
                self.model.replay_buffer.add(
                    np.asarray(obs, dtype=np.float32),
                    np.asarray(next_obs, dtype=np.float32),
                    act_arr,
                    rew_arr,
                    done_arr,
                    [{}],
                )
                transitions += 1
        if transitions == 0:
            return
        batch_size = min(self.model.batch_size, transitions)
        gradient_steps = max(1, transitions // batch_size)
        self.model.train(gradient_steps=gradient_steps, batch_size=batch_size)
        if getattr(Config, "FIX_EPSILON", False):
            self.reset_epsilon()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "sb3_parameters": self.model.get_parameters(),
            "exploration_rate": float(getattr(self.model, "exploration_rate", self.cfg.epsilon)),
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        if not params:
            return
        model_params = params.get("sb3_parameters", params)
        self.model.set_parameters(model_params, exact_match=False)
        exploration = params.get("exploration_rate")
        if exploration is not None:
            self.reset_epsilon(float(exploration))

    def reset_epsilon(self, epsilon: Optional[float] = None) -> None:
        super().reset_epsilon(epsilon)
        value = float(self.cfg.epsilon)
        if hasattr(self.model, "exploration_schedule"):
            schedule = self.model.exploration_schedule
            schedule.initial_eps = value
            if getattr(Config, "FIX_EPSILON", False) and hasattr(schedule, "final_eps"):
                schedule.final_eps = value
        if hasattr(self.model, "exploration_initial_eps"):
            self.model.exploration_initial_eps = value
        if getattr(Config, "FIX_EPSILON", False) and hasattr(self.model, "exploration_final_eps"):
            self.model.exploration_final_eps = value
        self.model.exploration_rate = value

    def clear_replay_buffer(self) -> None:
        buffer = getattr(self.model, "replay_buffer", None)
        if buffer is not None:
            buffer.reset()


def make_dqn_agent(
    env,
    cfg: Optional[DQNConfig] = None,
    agent_type: str = "dqn",
    seed: Optional[int] = None,
) -> BaseDQNAgent:
    agent_type = (agent_type or "dqn").lower()
    if cfg is None:
        cfg = DQNConfig()
    if seed is not None and cfg.seed != seed:
        cfg = replace(cfg, seed=seed)
    if agent_type in {"sb3_dqn", "dqn"}:  # treat legacy "dqn" as sb3 implementation
        return SB3DQNAgent(env, cfg)
    raise ValueError(f"Unsupported agent_type '{agent_type}'. Use 'sb3_dqn'.")


__all__ = [
    "BaseDQNAgent",
    "SB3DQNAgent",
    "DQNConfig",
    "make_dqn_agent",
]
if BaseCallback is not None:  # pragma: no cover - defined only when SB3 available
    class EpisodeLimitCallback(BaseCallback):  # type: ignore[misc]
        def __init__(self, max_episodes: int):
            super().__init__()
            self.max_episodes = max(0, int(max_episodes))
            self.episode_count = 0

        def _on_step(self) -> bool:
            if self.max_episodes <= 0:
                return False
            dones = self.locals.get("dones")
            if dones is not None and np.any(dones):
                self.episode_count += int(np.sum(dones))
                if self.episode_count >= self.max_episodes:
                    return False
            return True

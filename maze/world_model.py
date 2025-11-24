import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, model_type: str = "nn"):
        super(WorldModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model_type = model_type
        if obs_dim % 2 != 0:
            raise ValueError("Observation dimension must be even for cache-based models")
        self.agent_obs_dim = obs_dim // 2
        self.goal_obs_dim = obs_dim // 2
        grid_size_float = math.sqrt(self.agent_obs_dim)
        if not grid_size_float.is_integer():
            raise ValueError("Agent observation dimension must be a perfect square")
        self.grid_size = int(grid_size_float)

        if model_type == "nn":
            self.model = nn.Sequential(
                nn.Linear(self.agent_obs_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.agent_obs_dim)
            )
        elif model_type in {"cache", "debug"}:
            self.cache: Dict[Tuple[int, int], int] = {}
        else:
            raise ValueError(f"Unknown world model type: {model_type}")

    def forward(self, state, action_one_hot):
        if self.model_type == "nn":
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action_one_hot.dim() == 1:
                action_one_hot = action_one_hot.unsqueeze(0)

            agent_slice = state[:, : self.agent_obs_dim]
            goal_slice = state[:, self.agent_obs_dim :]
            x = torch.cat([agent_slice, action_one_hot], dim=1)
            agent_logits = self.model(x)
            agent_pred = torch.softmax(agent_logits, dim=1)
            return torch.cat([agent_pred, goal_slice], dim=1)

        if self.model_type in {"cache", "debug"}:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action_one_hot.dim() == 1:
                action_one_hot = action_one_hot.unsqueeze(0)

            action_idx = torch.argmax(action_one_hot, dim=1)
            outputs = []
            device = state.device
            dtype = state.dtype
            for state_row, action_value in zip(state, action_idx):
                agent_slice = state_row[: self.agent_obs_dim]
                goal_slice = state_row[self.agent_obs_dim :].clone()
                agent_idx = int(torch.argmax(agent_slice).item())
                key = (agent_idx, int(action_value.item()))
                next_agent_idx = self.cache.get(key, agent_idx)
                next_agent = torch.zeros(self.agent_obs_dim, device=device, dtype=dtype)
                next_agent[next_agent_idx] = 1.0
                outputs.append(torch.cat([next_agent, goal_slice], dim=0))
            return torch.stack(outputs, dim=0)

        raise RuntimeError(f"Unsupported world model type: {self.model_type}")

    def cache_transitions(self, states, actions, next_states):
        if self.model_type not in {"cache", "debug"}:
            raise RuntimeError("cache_transitions is only available for cache-based world models")

        actions = actions.view(-1)
        for state_row, action_value, next_state_row in zip(states, actions, next_states):
            agent_idx = self._agent_index(state_row)
            next_agent_idx = self._agent_index(next_state_row)
            self.set_transition_index(agent_idx, int(action_value.item()), next_agent_idx)

    def clear_cache(self):
        if self.model_type not in {"cache", "debug"}:
            raise RuntimeError("clear_cache is only available for cache-based world models")
        self.cache.clear()

    def set_transition_index(self, state_idx: int, action_value: int, next_state_idx: int):
        if self.model_type not in {"cache", "debug"}:
            raise RuntimeError("set_transition_index is only available for cache-based world models")
        self.cache[(int(state_idx), int(action_value))] = int(next_state_idx)

    def _agent_index(self, state_row: torch.Tensor) -> int:
        agent_slice = state_row[: self.agent_obs_dim]
        return int(torch.argmax(agent_slice).item())

import os
import shutil
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as th
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import RGCNConv


class GraphLoggerExtended(BaseCallback):
    """
    Logs final rewards per episode, stores them, writes CSV at training end.
    """
    def __init__(self, run_id, output_dir, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.run_id = run_id
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.episode_rewards=[]
        self.steps=[]

    def _on_step(self)->bool:
        dones=self.locals.get("dones")
        rewards=self.locals.get("rewards")
        if dones is not None and rewards is not None:
            for idx, done in enumerate(dones):
                if done:
                    ep_reward=rewards[idx]
                    self.episode_rewards.append(ep_reward)
                    self.steps.append(self.num_timesteps)
                    if self.verbose>0:
                        print(f"[CALLBACK run={self.run_id}] step={self.num_timesteps}, ep_reward={ep_reward:.2f}")
        return True

    def _on_training_end(self):
        csv_path = os.path.join(self.output_dir,f"run_{self.run_id}_rewards.csv")
        with open(csv_path,"w", newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["timesteps","episode_reward"])
            for t,r in zip(self.steps,self.episode_rewards):
                writer.writerow([t,r])
        if self.verbose>0:
            print(f"[GraphLoggerExtended] Saved reward data for run={self.run_id} to {csv_path}")

    def get_data(self):
        return np.array(self.steps), np.array(self.episode_rewards)


class ActionProbabilityLoggerExtended(BaseCallback):
    """
    Logs the policy's action probabilities at each step (from env 0), saves to CSV
    """
    def __init__(self, run_id, output_dir, n_actions, max_episode_steps=5, verbose=0):
        super().__init__(verbose)
        self.run_id=run_id
        self.output_dir=output_dir
        self.n_actions=n_actions
        self.max_episode_steps=max_episode_steps

        self.action_probs_by_step=[[] for _ in range(self.max_episode_steps)]
        self.episode_count=0
        self.episode_step=0
        self.long_form_data=[]

    def _on_training_start(self):
        self.episode_count=0
        self.episode_step=0
        self.action_probs_by_step=[[] for _ in range(self.max_episode_steps)]
        self.long_form_data=[]

    def _on_step(self)->bool:
        actions=self.locals["actions"]
        obs_tensor=self.locals["obs_tensor"]
        dones=self.locals["dones"]

        lstm_states=self.locals.get("lstm_states")
        episode_starts=self.locals.get("episode_starts")
        if lstm_states is not None and isinstance(lstm_states[0], tuple):
            lstm_states=lstm_states[0]

        with th.no_grad():
            if lstm_states is not None and episode_starts is not None:
                dist = self.model.policy.get_distribution(obs_tensor, lstm_states, episode_starts)[0]
            else:
                dist = self.model.policy.get_distribution(obs_tensor)[0]
            probs=dist.distribution.probs

        p0=probs[0].cpu().numpy()
        if self.episode_step<self.max_episode_steps:
            self.action_probs_by_step[self.episode_step].append(p0)
        row=[self.episode_count, self.episode_step]+p0.tolist()
        self.long_form_data.append(row)

        self.episode_step+=1
        if dones and dones[0]:
            self.episode_count+=1
            self.episode_step=0

        return True

    def _on_training_end(self):
        csv_path=os.path.join(self.output_dir,f"run_{self.run_id}_action_probs.csv")
        header=["episode_index","step_in_episode"]+[f"action_{a}_prob" for a in range(self.n_actions)]
        with open(csv_path,"w", newline="") as f:
            writer=csv.writer(f)
            writer.writerow(header)
            for row in self.long_form_data:
                writer.writerow(row)
        if self.verbose>0:
            print(f"[ActionProbabilityLoggerExtended] Saved action-prob data for run={self.run_id} to {csv_path}")

    def get_data_long_form(self):
        cols=["episode_index","step_in_episode"]+[f"action_{a}_prob" for a in range(self.n_actions)]
        df=pd.DataFrame(self.long_form_data, columns=cols)
        return df


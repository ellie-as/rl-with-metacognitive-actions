import random
from typing import Optional

import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import RGCNConv

from faker import Faker
fake = Faker()


class GraphBuilder:
    """
    Abstract-like base class for building and sampling from a directed graph.
    Subclasses must define:
      - self.relation_types (list of strings)
      - self.rel_to_id (dict: relation_name -> int)
      - build_graph()
      - sample_observations(G, n=...)
    """
    def __init__(self):
        self.relation_types = []
        self.test_relation_types = []
        self.rel_to_id = {}
        self._rng = random.Random()
        self._seed: Optional[int] = None

    def get_relation_types(self):
        return self.relation_types

    def get_test_relation_types(self):
        return self.test_relation_types

    def get_relation_to_id_map(self):
        return self.rel_to_id

    def build_graph(self):
        """
        Must return a 'true' directed graph (nx.DiGraph or nx.MultiDiGraph)
        with the relationship name stored in edge data, e.g.:
          G.add_edge(u, v, relationship="SOME_REL")
        """
        raise NotImplementedError()

    def sample_observations(self, G, n=1):
        """
        Return a list of (u, v, relationship_name) sampled from G.
        """
        raise NotImplementedError()

    def seed(self, seed: Optional[int]) -> None:
        """Optional hook for subclasses that maintain their own RNG state."""
        self._seed = None if seed is None else int(seed)
        if seed is not None:
            self._rng.seed(int(seed))

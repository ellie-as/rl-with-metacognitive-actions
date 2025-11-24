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
from graph_builder import GraphBuilder

from faker import Faker


class FamilyTreeBuilder(GraphBuilder):
    """
    Builder for "family tree" domain.
    """
    # Domain-specific relationships (including "NO_RELATION")
    RELATIONSHIP_TYPES = [
        "NO_RELATION",
        "PARENT_OF",
        "CHILD_OF",
        "SPOUSE_OF",
        "SIBLING_OF",
        "GRANDPARENT_OF",
        "GRANDCHILD_OF",
    ]

    def __init__(self, base_num_children=2, grandparent_num_children=2):
        super().__init__()
        self.base_num_children = base_num_children
        self.grandparent_num_children = grandparent_num_children
        self.relation_types = self.RELATIONSHIP_TYPES
        self.test_relation_types = self.RELATIONSHIP_TYPES
        self.rel_to_id = {r: i for i, r in enumerate(self.relation_types)}
        self.fake = Faker()

    def _generate_name(self):
        return self.fake.name()

    def _create_nuclear_family(self, num_children, child_names):
        parent1 = self._generate_name()
        parent2 = self._generate_name()
        while parent2 == parent1:
            parent2 = self._generate_name()
        additional_needed = num_children - len(child_names)
        additional_children = [self._generate_name() for _ in range(additional_needed)]
        children = child_names + additional_children

        relationships = {
            parent1: {"SPOUSE_OF": [parent2], "PARENT_OF": children},
            parent2: {"SPOUSE_OF": [parent1], "PARENT_OF": children},
        }
        for c in children:
            relationships[c] = {
                "CHILD_OF": [parent1, parent2],
                "SIBLING_OF": [sib for sib in children if sib != c]
            }
        return relationships

    def _infer_grandparent_edges(self, relationships):
        # If X is parent of Y, and Y is parent of Z => X is grandparent of Z
        temp = {}
        for person, rel_dict in relationships.items():
            if 'PARENT_OF' in rel_dict:
                for child in rel_dict['PARENT_OF']:
                    child_rels = relationships.get(child, {})
                    if 'PARENT_OF' in child_rels:
                        for grandchild in child_rels['PARENT_OF']:
                            temp.setdefault(person, {}).setdefault('GRANDPARENT_OF', []).append(grandchild)
                            temp.setdefault(grandchild, {}).setdefault('GRANDCHILD_OF', []).append(person)

        # Merge back into relationships
        for p, rdict in temp.items():
            if p not in relationships:
                relationships[p] = rdict
            else:
                for rt, names in rdict.items():
                    relationships[p].setdefault(rt, []).extend(names)
        return relationships

    def build_graph(self):
        # Build the base family
        base_rels = self._create_nuclear_family(self.base_num_children, [])
        # Identify parents
        parents = []
        for name, rels in base_rels.items():
            if 'PARENT_OF' in rels:
                parents.append(name)

        # For each parent, create a "grandparent family" around them
        combined = dict(base_rels)
        for parent_name in parents:
            gp_rels = self._create_nuclear_family(self.grandparent_num_children, [parent_name])
            # Merge
            for person, rdict in gp_rels.items():
                if person not in combined:
                    combined[person] = rdict
                else:
                    for rt, names in rdict.items():
                        combined[person].setdefault(rt, []).extend(names)

        # Infer grandparent edges
        combined = self._infer_grandparent_edges(combined)

        # Build the Nx graph
        G = nx.DiGraph()
        for person, rels in combined.items():
            G.add_node(person)
            for rel_type, others in rels.items():
                for o in others:
                    G.add_edge(person, o, relationship=rel_type)
        return G

    def sample_observations(self, G, n=1):
        edges = list(G.edges(data=True))
        if not edges:
            return []
        obs = []
        for _ in range(n):
            u, v, data = self._rng.choice(edges)
            rel_name = data.get("relationship", "NO_RELATION")
            obs.append((u, v, rel_name))
        return obs

    def seed(self, seed: int) -> None:
        super().seed(seed)
        self.fake.seed_instance(int(seed))

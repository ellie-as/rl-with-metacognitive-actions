import os
import shutil
import csv
import random
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from graph_autoencoder import *
from callbacks import *
from seed_utils import derive_seed


class GraphEnv(gym.Env):
    """
    A graph environment with 6 meta-actions:
     0 = Train MF from store
     1 = Graph <- store
     2 = Train MF from graph
     3 = Expand graph w. GCN
     4 = Update rule library
     5 = Do nothing

    Observations: [store_count, graph_edge_count, rule_flag]
    """
    def __init__(self, builder, output_dir, max_meta_steps=10, initial_obs_count=20):
        super().__init__()
        self.builder = builder
        self.output_dir = os.fspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self._gcn_model_path = os.path.join(self.output_dir, "gcn_model.pt")
        self.max_meta_steps = max_meta_steps
        self.initial_obs_count = initial_obs_count

        # Relationship types from the builder
        self.relation_types = self.builder.get_relation_types()
        self.rel_to_id = self.builder.get_relation_to_id_map()

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1e6, shape=(3,), dtype=np.float32)

        self.true_graph = None
        self.graph = nx.MultiDiGraph()
        self.store = []
        self.mf_learner = {}   # Minimal "MF" learner storing edges

        self.gcn_model = None
        self.gcn_trained = False
        self.seen_graphs = []
        self.current_step = 0
        self._rng = random.Random()
        self._base_seed: Optional[int] = None
        self._episode_index = -1

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._base_seed = int(seed)
            self._episode_index = -1
        elif self._base_seed is None:
            self._base_seed = 0
            self._episode_index = -1

        self._episode_index += 1
        episode_seed = derive_seed(self._base_seed, "episode", self._episode_index)

        super().reset(seed=episode_seed)
        self._rng.seed(episode_seed)
        self.builder.seed(derive_seed(episode_seed, "builder"))

        self.current_step = 0
        self.store.clear()
        self.mf_learner.clear()
        self.graph.clear()
        self.gcn_trained = False

        self.true_graph = self.builder.build_graph()
        init_obs = self.builder.sample_observations(self.true_graph, n=self.initial_obs_count)
        self.store.extend(init_obs)

        return self._get_observation(), {}

    def _get_observation(self):
        fact_count = float(len(self.mf_learner))
        graph_count = float(self.graph.number_of_edges())
        rule_flag = 1.0 if self.gcn_trained else 0.0
        return np.array([fact_count, graph_count, rule_flag], dtype=np.float32)

    def _train_mf_learner(self, source):
        if isinstance(source, (nx.MultiDiGraph, nx.DiGraph)):
            for u,v,data in source.edges(data=True):
                r = data.get("relationship")
                if r:
                    self.mf_learner[(u,v,r)] = True
        else:
            for (u,v,r) in source:
                self.mf_learner[(u,v,r)] = True

    def _update_graph_from_store(self):
        for (u,v,r) in self.store:
            if not self.graph.has_edge(u,v):
                self.graph.add_edge(u,v, relationship=r)

    def _graph_to_pyg_data(self, G, feature_dim=16):
        nodes = list(G.nodes())
        node2idx = {n:i for i,n in enumerate(nodes)}
        num_nodes = len(nodes)
        x = torch.randn((num_nodes, feature_dim), dtype=torch.float)

        # Positive edges
        edge_list_pos = []
        edge_type_pos = []
        for (u,v,data) in G.edges(data=True):
            rname = data.get("relationship","NO_RELATION")
            rid = self.rel_to_id.get(rname, 0)
            edge_list_pos.append([node2idx[u], node2idx[v]])
            edge_type_pos.append(rid)

        # Negative edges
        neg_list = []
        neg_type = []
        num_pos = len(edge_list_pos)
        existing = set((a,b) for (a,b) in zip(*zip(*edge_list_pos)))
        tries, max_tries = 0, 2*num_pos

        while len(neg_list)<num_pos and tries<max_tries:
            i = self._rng.randint(0,num_nodes-1)
            j = self._rng.randint(0,num_nodes-1)
            if i==j or (i,j) in existing:
                tries+=1
                continue
            neg_list.append([i,j])
            neg_type.append(self.rel_to_id["NO_RELATION"])
            tries+=1

        edge_list = edge_list_pos+neg_list
        edge_type_list = edge_type_pos+neg_type

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.edge_type = edge_type
        return data

    def _train_gcn_on_all_data(self, feature_dim=16, num_epochs=20):
        if not self.seen_graphs:
            print("[TRAIN_GCN] No graphs in seen_graphs, skipping.")
            return
        if self.gcn_model is None:
            self.gcn_model = GCNLinkPredictorWrapper(
                in_channels=feature_dim,
                hidden_channels=16,
                embedding_dim=16,
                num_relations=len(self.relation_types)
            )

        model = self.gcn_model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        loader = PyGDataLoader(self.seen_graphs, batch_size=4, shuffle=True)
        model.train()

        for epoch in range(num_epochs):
            total_loss=0.0
            for batch in loader:
                optimizer.zero_grad()
                z = model(batch)
                e_src, e_dst = batch.edge_index
                labels = batch.edge_type
                logits_list=[]
                for i in range(e_src.size(0)):
                    logits_ij = model.edge_logits(z[e_src[i]], z[e_dst[i]])
                    logits_list.append(logits_ij.unsqueeze(0))
                logits = torch.cat(logits_list, dim=0)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()

            if (epoch+1)%5==0:
                avg_loss = total_loss/len(loader)
                print(f"[TRAIN_GCN] Epoch {epoch+1}/{num_epochs}, Loss={avg_loss:.4f}")

        self.gcn_trained=True
        print("[TRAIN_GCN] Finished training GCN on all seen graphs.")

    def _expand_graph(self, feature_dim=16):
        if not self.gcn_trained or self.gcn_model is None:
            print("[EXPAND] GCN not ready; call 'Update rule library' (action 4) first.")
            return

        if self.graph.number_of_nodes()<2:
            print("[EXPAND] Graph too small, skipping.")
            return

        print("[EXPAND] Graph edges before expansion:")
        print(self.graph.edges(data=True))

        data = self._graph_to_pyg_data(self.graph, feature_dim)
        self.gcn_model.eval()
        with torch.no_grad():
            z = self.gcn_model(data)

        existing_edges = set(self.graph.edges())
        nodes = list(self.graph.nodes())
        new_edges = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i==j:
                    continue
                u,v = nodes[i], nodes[j]
                if (u,v) in existing_edges:
                    continue
                logits = self.gcn_model.edge_logits(z[i], z[j])
                probs = torch.softmax(logits, dim=-1)
                if torch.max(probs).item() > 0.0:
                    pred_id = torch.argmax(probs).item()
                    if pred_id==0:
                        continue # "NO_RELATION"
                    rel_name = self.relation_types[pred_id]
                    new_edges.append((u,v,rel_name))

        for (u,v,relname) in new_edges:
            self.graph.add_edge(u,v, relationship=relname)

        print(f"[EXPAND] Added {len(new_edges)} edges. Example: {new_edges[:5]}")

    def _update_rule_library(self):
        if self.gcn_trained and self.gcn_model is not None:
            print("[UPDATE_RULE_LIBRARY] GCN already trained and loaded in memory.")
            return

        if self.gcn_model is None:
            self.gcn_model = GCNLinkPredictorWrapper(
                in_channels=16,
                hidden_channels=16,
                embedding_dim=16,
                num_relations=len(self.relation_types)
            )

        model_path = self._gcn_model_path
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            self.gcn_model.load_state_dict(state_dict)
            self.gcn_trained = True
            print("[UPDATE_RULE_LIBRARY] Loaded existing GCN model from disk.")
            return

        print("[UPDATE_RULE_LIBRARY] No existing model found; training a new GCN model.")

        training_graphs = list(self.seen_graphs)
        if self.graph.number_of_edges() > 0:
            training_graphs.append(self._graph_to_pyg_data(self.graph, 16))

        if not training_graphs:
            random_data = []
            for _ in range(500):
                G = self.builder.build_graph()
                pyg_data = self._graph_to_pyg_data(G, 16)
                random_data.append(pyg_data)
            training_graphs = random_data

        self._train_gcn_on_external_data(training_graphs, 16, 100)
        torch.save(self.gcn_model.state_dict(), model_path)
        self.gcn_trained = True
        print("[UPDATE_RULE_LIBRARY] Trained new GCN model and saved to disk.")

    def _train_gcn_on_external_data(self, data_list, feature_dim=16, num_epochs=20):
        if not data_list:
            print("[TRAIN_GCN_EXT] No external data, skipping.")
            return

        if self.gcn_model is None:
            self.gcn_model = GCNLinkPredictorWrapper(
                in_channels=feature_dim,
                hidden_channels=16,
                embedding_dim=16,
                num_relations=len(self.relation_types)
            )

        model=self.gcn_model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        loader=PyGDataLoader(data_list, batch_size=4, shuffle=True)
        model.train()
        for epoch in range(num_epochs):
            total_loss=0.0
            for batch in loader:
                optimizer.zero_grad()
                z=model(batch)
                e_src, e_dst = batch.edge_index
                labels = batch.edge_type
                logits_list=[]
                for i in range(e_src.size(0)):
                    logits_ij=model.edge_logits(z[e_src[i]], z[e_dst[i]])
                    logits_list.append(logits_ij.unsqueeze(0))
                logits = torch.cat(logits_list, dim=0)
                loss=criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()
            if (epoch+1)%5==0:
                avg_loss=total_loss/len(loader)
                print(f"[TRAIN_GCN_EXT] Epoch {epoch+1}/{num_epochs}, Loss={avg_loss:.4f}")
        print("[TRAIN_GCN_EXT] Finished training on external data.")

    def _evaluate_mf_accuracy(self):
        print(f"[EVAL] MF has learned {len(self.mf_learner)} relationships.")
        print(list(self.mf_learner.keys()))
        true_edges=[]
        for (u,v,d) in self.true_graph.edges(data=True):
            r = d.get("relationship")
            if r:
                true_edges.append((u,v,r))
        print(f"[EVAL] True edges:")
        print(true_edges)

        correct_links = [k for k in self.mf_learner.keys() if k in true_edges]
        correct_links = list(set(correct_links))
        score = len(correct_links)
        print("[EVAL] Correctly learned edges:")
        print(correct_links)
        print(f"[EVAL] Final reward={score}")
        return score

    def step(self, action):
        self.current_step+=1

        # 6 meta-actions
        if action==0:
            print("[STEP] Action 0: Train MF from store.")
            self._train_mf_learner(self.store)
        elif action==1:
            print("[STEP] Action 1: Graph <- store.")
            self._update_graph_from_store()
        elif action==2:
            print("[STEP] Action 2: Train MF from internal graph.")
            self._train_mf_learner(self.graph)
        elif action==3:
            print("[STEP] Action 3: Expand graph with GCN.")
            self._expand_graph()
        elif action==4:
            print("[STEP] Action 4: Update rule library.")
            self._update_rule_library()
        elif action==5:
            print("[STEP] Action 5: Do nothing.")

        random_done = (self._rng.random()<0.05)
        done = (self.current_step>=self.max_meta_steps) or random_done

        if action==5:
            reward=0.0
        else:
            reward=-0.2
        if done:
            reward=self._evaluate_mf_accuracy()

        return self._get_observation(), reward, done, False, {}

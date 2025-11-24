import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import gymnasium as gym
from torch_geometric.nn import RGCNConv


class RelationalGraphAutoencoder(nn.Module):
    """
    A small RGCN-based autoencoder for link prediction in relational graphs.
    """
    def __init__(self,
                 in_channels=16,
                 hidden_channels=16,
                 embedding_dim=16,
                 num_relations=7,
                 num_bases=None,
                 dropout=0.3):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, embedding_dim, num_relations, num_bases=num_bases)
        self.decoder = nn.Sequential(
            nn.Linear(2*embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_relations)
        )
        self.dropout = dropout

    def encode(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        emb = torch.cat([z[src], z[dst]], dim=1)
        return self.decoder(emb)

    def forward(self, data):
        z = self.encode(data)
        logits = self.decode(z, data.edge_index)
        return z, logits


class GCNLinkPredictorWrapper(nn.Module):
    """
    A wrapper that uses the RelationalGraphAutoencoder for link prediction
    """
    def __init__(self,
                 in_channels=16,
                 hidden_channels=16,
                 embedding_dim=16,
                 num_relations=7,
                 dropout=0.3,
                 num_bases=4):
        super().__init__()
        self.autoencoder = RelationalGraphAutoencoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            num_relations=num_relations,
            dropout=dropout,
            num_bases=num_bases
        )

    def forward(self, data):
        return self.autoencoder.encode(data)

    def edge_logits(self, z_i, z_j):
        return self.autoencoder.decoder(torch.cat([z_i, z_j], dim=-1))

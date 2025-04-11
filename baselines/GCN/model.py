import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp
import torch
from dgl.nn import SAGEConv, GATConv, GraphConv
import torch.nn as nn
import torch.nn.functional as F


# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        # 定义两层的图卷积层
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, num_heads)
        self.conv2 = GATConv(h_feats * num_heads, h_feats, num_heads)

    def forward(self, g, in_feat):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, in_feat)
        # print(h.shape)  # [2708, 8, 96] means [node_cnt, num_heads, hidden_dim]
        h = F.elu(h)
        h = h.flatten(1)
        h = self.conv2(g, h)
        # print(h.shape)
        h = h.flatten(1)
        return h

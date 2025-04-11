import torch.nn as nn
import dgl.function as fn
import pandas as pd
import dgl
import torch
import numpy as np
from ast import literal_eval


class DotPredictor(nn.Module):
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return graph.edata['score']
        #   return graph.edata['score'][:, 0]


def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_data(city, feature_dim):
    file = f"./data/{city}.csv"
    df = pd.read_csv(file)
    u = []
    v = []
    df['neighbors'] = df['neighbor_geom'].apply(literal_eval)
    for i, row in df.iterrows():
        for j in row['neighbors']:
            u.append(int(row['geom_id']))
            v.append(int(j))

    print("node nums: ", len(df))
    print("edge nums: ", len(u))

    g = dgl.graph((u, v), num_nodes=len(df))
    node_features = torch.rand(g.number_of_nodes(), feature_dim)
    g.ndata['feat'] = node_features

    return g

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import dgl.data
from model import GraphSAGE, GCN, GAT
from utils import DotPredictor, set_random_seed, load_data
from sklearn.metrics import roc_auc_score


def compute_loss(pos_score, neg_score):  # --------- 3)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).unsqueeze(1).to(scores.device)
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):  # -------- 4)
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).unsqueeze(1).cpu().numpy()
    return roc_auc_score(labels, scores)


def main():
    set_random_seed(seed)
    # Load data
    # if dataset == 'Cora':
    #     dataset = dgl.data.CoraGraphDataset()
    #     g = dataset[0]
    # else:
    g = load_data(city_name, feature_dim)
    g = g.to(device)


    print(g.ndata['feat'].shape)  # torch.Size([2708, 1433])
    # Split edge set for training and testing
    u, v = g.edges()  # u: src node, v: dest node

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)  # Random shuffle
    test_size = int(len(eids) * 0.1)  # train:test = 9:1
    train_size = g.number_of_edges() - test_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # print(len(train_neg_v), len(test_neg_v))
    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    # Move graphs to the device
    train_g = train_g.to(device)
    train_pos_g = train_pos_g.to(device)
    train_neg_g = train_neg_g.to(device)
    test_pos_g = test_pos_g.to(device)
    test_neg_g = test_neg_g.to(device)

    # Define the model
    model = {
        "GraphSage": GraphSAGE(train_g.ndata["feat"].shape[1], hidden_dim).to(device),
        "GCN": GCN(train_g.ndata["feat"].shape[1], hidden_dim).to(device),
        "GAT": GAT(train_g.ndata["feat"].shape[1], hidden_dim, ).to(device),
    }[model_name]

    # model = GraphSAGE(train_g.ndata["feat"].shape[1], hidden_dim).to(device)
    pred = DotPredictor().to(device)

    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()), lr=lr
    )

    # ----------- 4. training -------------------------------- #
    for e in range(epoch):
        # Forward pass
        h = model(train_g, train_g.ndata["feat"].to(device))
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print("In epoch {}, loss: {}".format(e, loss.item()))

    # ----------- 5. check results ------------------------ #
    with torch.no_grad():
        h = model(train_g, train_g.ndata["feat"].to(device))
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print("AUC:", compute_auc(pos_score, neg_score))
        print(h.shape)  # torch.Size([2708, 16])
        # shape is [node_num, hidden_dim]
        np.save(f"../../embeddings/{model_name}/{city_name}.npy", h.detach().cpu().numpy())


if __name__ == '__main__':
    device = torch.device("cuda:1")
    epoch = 5000
    lr = 0.001
    feature_dim = 96
    hidden_dim = 96
    models = ["GraphSage", "GCN", "GAT"]
    model_name = models[0]
    seed = 42
    for city in range(2):
        city_names = ["Chicago", "Manhattan"]
        city_name = city_names[city]
        main()

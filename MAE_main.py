import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model.MAEFinal import MAE, MAEConfig
from data.datasets import DatasetMAE
from utils.utils import set_random_seed, calc, parse_args, process_outliers
from tqdm import tqdm, trange
import random

cities = ["Chicago", "Manhattan"]
targets = ["carbon", "nightlight", "crime", "taxi", "poi", "src", "dst"]
#targets = ["carbon", "nightlight", "crime", "taxi"]

# targets = ["carbon", "population", "crime", "taxi", "poi", "src", "dst"]


def normalize(dataset):
    dataset = np.array(dataset)
    mean = dataset.mean()
    std = dataset.std()
    if std == 0:
        return dataset, True
    dataset = (dataset - mean) / std
    return dataset, False


def create_datasets(args):
    # 训练集：80% POI Taxi
    # 验证集：POI的剩下的 Taxi剩下的
    # 测试集：1 个POI， 1个 Taxi，全部的 Crime, Population, Carbon

    embedding = np.load(f"./embeddings/{args.source}/{args.dataset}.npy")
    print(len(embedding))
    if args.use_embedding == 2:
        embedding_list = []
        embedding_list.append(np.load(f"./embeddings/UrbanCLIP/{args.dataset}.npy"))
        embedding_list.append(np.load(f"./embeddings/GCN/{args.dataset}.npy"))
        embedding_list.append(np.load(f"./embeddings/ReCP/{args.dataset}.npy"))
        for i in range(3):
            embedding_list[i] = torch.FloatTensor(embedding_list[i]).to(args.gpu)

    train_datas = []
    val_datas = []
    test_datas = []

    for target in targets:
        data = pd.read_csv(f"./data/{args.dataset}/{target}.csv")
        # collect the top 80% row of the data not the first row
        # get the length of the data
        length = len(data)
        print(length)
        if length == 1:
            now_data = data.iloc[0].to_list()
            # if target == "carbon":
            # for j in range(len(now_data)):
            #     if now_data[j] > 5000 or now_data[j] < 100:
            #         now_data[j] = (now_data[j + 1] + now_data[j - 1]) / 2
            now_data = process_outliers(now_data)
            test_datas.append(now_data)
            continue
        train_size = int(0.8 * length)
        val_size = length - train_size - 1
        for i in range(length):
            if i < train_size:
                # random between 0 ~ 1
                train_datas.append(data.iloc[i].to_list())
            elif i < train_size + val_size:
                val_datas.append(data.iloc[i].to_list())
            else:
                test_datas.append(data.iloc[i].to_list())

    dim = len(embedding[0])

    print(f"Embedding dimension: {dim}")

    N = len(embedding)

    assert len(embedding) == len(
        train_datas[0]), f"The embedding {dim} and data {len(train_datas[0])} should have the same length"

    train_dataset = DatasetMAE(train_datas)
    val_dataset = DatasetMAE(val_datas)
    test_dataset = DatasetMAE(test_datas, args.seed, args.few_shot, test=1)

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.use_embedding == 2:
        return train_loader, val_loader, test_loader, dim, embedding_list, N
    return train_loader, val_loader, test_loader, dim, embedding, N


def process_output(predicts, labels_cuda, mask_cuda):
    predicts = predicts.cpu().detach().numpy()
    labels = labels_cuda.cpu().detach().numpy()
    mask = mask_cuda.cpu().detach().numpy()

    predicts = predicts[mask == 1]
    labels = labels[mask == 1]

    return predicts, labels


def train(model, criterion, optimizer, loader, args, epoch):
    device = torch.device(args.gpu)
    model.train()
    all_predictions = []
    all_truths = []
    total_loss = 0.0
    for labels, mask in loader:
        labels_cuda = labels.to(device=device)
        mask_cuda = mask.to(device=device)

        optimizer.zero_grad()

        predicts, loss = model(labels_cuda, mask_cuda)

        total_loss += loss.item()
        loss.backward()

        optimizer.step()

        predicts, y = process_output(predicts, labels_cuda, mask_cuda)
        all_predictions.extend(predicts)
        all_truths.extend(y)

    return calc("Train", epoch, all_predictions, all_truths, total_loss / len(loader), args)


def evaluate(model, loader, args, epoch):
    device = torch.device(args.gpu)
    model.eval()

    all_truths, all_predictions = [], []
    with torch.no_grad():
        for labels, mask in loader:
            labels_cuda = labels.to(device=device)
            mask_cuda = mask.to(device=device)

            predicts, loss = model(labels_cuda, mask_cuda)
            predicts, y = process_output(predicts, labels_cuda, mask_cuda)

            all_predictions.extend(predicts)
            all_truths.extend(y)

    return calc("Eval", epoch, all_predictions, all_truths, None, args)


def testIn(model, loader, args):
    device = torch.device(args.gpu)
    model.eval()
    with torch.no_grad():
        for labels, mask in loader:

            labels_cuda = labels.to(device=device)
            mask_cuda = mask.to(device=device)

            predicts, loss = model(labels_cuda, mask_cuda)

            for i in range(3):
                all_truths, all_predictions = [], []
                for j in range(len(mask_cuda[i])):
                    if mask_cuda[i][j] == 1:
                        all_truths.append(float(labels_cuda[i][j]))
                        all_predictions.append(float(predicts[i][j]))
                calc("testIn", targets[i], all_predictions, all_truths, None, args, model="MAE")


def test(model, loader, args):
    device = torch.device(args.gpu)
    model.eval()
    with torch.no_grad():
        for labels, mask in loader:

            labels_cuda = labels.to(device=device)
            mask_cuda = mask.to(device=device)

            predicts, loss = model(labels_cuda, mask_cuda)

            for i in range(3):
                all_truths, all_predictions = [], []
                for j in range(len(mask_cuda[i])):
                    if mask_cuda[i][j] == 1:
                        all_truths.append(float(labels_cuda[i][j]))
                        all_predictions.append(float(predicts[i][j]))
                calc("Test", targets[i], all_predictions, all_truths, None, args, model="MAE")


def main(args):
    set_random_seed(args.seed)

    device = torch.device(args.gpu)

    train_loader, val_loader, test_loader, dim, embedding, N = create_datasets(args)

    config = {
        "d1": MAEConfig(block_size=N, vocab_size=N, n_layer=1, n_head=4, n_embd=64),
        "d10": MAEConfig(block_size=N, vocab_size=N, n_layer=2, n_head=8, n_embd=128),
    }[args.model]

    source = args.source
    if args.use_embedding == 0:
        embedding = None
        source = "None"

    model = MAE(dim, config, embedding).to(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    criterion = nn.MSELoss()

    save_path = f"{args.checkpoint_dir}/MAE-{args.name}-{args.dataset}-{source}-{args.indicator}-{args.seed}-best.py"

    best_val = -float("inf")
    best_cnt = 0
    for epoch in range(args.epoch_num):
        train(model, criterion, optimizer, train_loader, args, epoch)
        cur_metrics = evaluate(model, val_loader, args, epoch)
        testIn(model, test_loader, args)
        checkpoint_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if cur_metrics['r2'] > best_val:
            best_val = cur_metrics['r2']
            best_cnt = 0
            torch.save(checkpoint_dict, save_path)
        else:
            best_cnt += 1

        if best_cnt > 100:
            break

    best_checkpoint = torch.load(
        save_path, map_location=torch.device("cpu")
    )

    model.load_state_dict(best_checkpoint["state_dict"])

    model.to(device)

    test(model, test_loader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    main(parser.parse_args())

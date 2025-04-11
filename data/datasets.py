import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.utils import set_random_seed, calc
from tqdm import tqdm, trange

import random


class DatasetLB(Dataset):  # Dataset For Linear Probiling
    def __init__(self, dataset, mean=None, std=None):
        super().__init__()
        self.embedding = []
        self.labels = []
        for emedding, label in tqdm(dataset):
            # if label < 100 or label > 600:
            #     continue
            self.embedding.append(emedding)
            self.labels.append(label)

        if mean is None:
            # plot labels  using mathplotlib
            import matplotlib.pyplot as plt
            plt.plot(self.labels)
            plt.show()

        if mean is None:
            self.mean = mean = np.mean(self.labels, axis=0)
            self.std = std = np.std(self.labels, axis=0)
            print(mean, std)

        self.embeddings = torch.tensor(self.embedding, dtype=torch.float32)

        self.labels = (self.labels - mean) / std
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]


def normalize(dataset):
    dataset = np.array(dataset)
    mean = dataset.mean()
    std = dataset.std()
    if std == 0:
        return dataset
    dataset = (dataset - mean) / std
    return dataset


def gen_mask(now_len):
    num_zeros = random.randint(3, 20)

    # 生成全为 1 的 mask
    mask = torch.ones(now_len, dtype=torch.int32)

    # 随机选择 num_zeros 个位置
    zero_indices = torch.randperm(now_len)[:num_zeros]

    # 将选定位置的值设置为 0
    mask[zero_indices] = 0

    return mask


def gen_new_mask(mask, now_zero):
    while True:
        mask_rate = np.random.uniform(0.1, 0.9)
        zero_mask = torch.rand(now_zero) < mask_rate
        zero_mask = zero_mask.int()
        # print(zero_mask)
        if torch.sum(zero_mask) > 0 and torch.sum(zero_mask) < now_zero:
            break

    now_mask = mask.int()
    cnt = 0
    for now_id in range(len(now_mask)):
        if now_mask[now_id] == 0:
            if zero_mask[cnt] == 1:
                now_mask[now_id] = 3
            cnt += 1
    # print(mask)
    return now_mask


class DatasetMAE(Dataset):  # Dataset For Masked Auto Encoder
    def __init__(self, datasets, seed=42, few_shot=-1, test=0, FT=0):
        super().__init__()

        self.labels = []
        # 首先 eval和 test 的部分一定是mask着的
        # 其次，train一部分是mask着的，一部分是没mask的
        self.masks = []

        self.zeros = []
        self.test = test
        self.FT = FT

        for dataset in tqdm(datasets):
            dataset = normalize(dataset)
            self.labels.append(torch.FloatTensor(dataset))

            if test == 1:
                set_random_seed(seed)

                id_set = np.arange(self.labels[-1].shape[-1])

                # split the dataset into train and test
                train_size = int(0.7 * len(id_set)) if few_shot == -1 else few_shot
                val_size = int(0.8 * len(id_set)) - train_size
                test_size = len(id_set) - val_size - train_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(id_set, [train_size, val_size,
                                                                                                  test_size])

                mask = torch.zeros(self.labels[-1].shape[-1])

                mask[val_dataset] = 2
                mask[test_dataset] = 1

                self.masks.append(mask)

                cnt = 0
                for now_id in range(len(mask)):
                    if mask[now_id] == 0:
                        cnt += 1
                self.zeros.append(cnt)

                # 随机mask掉一部分train，即mask==0的，预测另一部分？

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 非mask的部分，预测masked的了
        now_dataset = self.labels[index]
        # sample from a distribution between 0.1-0.9
        if self.test == 1:
            mask = self.masks[index]
        else:
            mask_rate = np.random.uniform(0.01, 0.99)
            mask = torch.rand(len(now_dataset)) < mask_rate
            # mask = gen_mask(len(now_dataset))
        if self.FT == 0:
            return now_dataset, mask
        else:
            now_mask = gen_new_mask(mask, self.zeros[index])
            return now_dataset, now_mask


class DatasetFT(Dataset):  # Dataset For FineTuning
    def __init__(self, datasets, seed=42, few_shot=-1, test=0):
        super().__init__()

        self.labels = []
        # 首先 eval和 test 的部分一定是mask着的
        # 其次，train一部分是mask着的，一部分是没mask的
        self.masks = []
        self.zeros = []
        self.test = test

        # for dataset, text_embedding in tqdm(pois):
        #     dataset = normalize(dataset)
        #     self.pois.append(torch.FloatTensor(dataset))
        #     self.auc_embeddings.append(torch.FloatTensor(text_embedding))
        #
        # self.pois = torch.stack(self.pois, dim=0)
        # self.auc_embeddings = torch.stack(self.auc_embeddings, dim=0)

        for dataset in tqdm(datasets):
            dataset = normalize(dataset)
            self.labels.append(torch.FloatTensor(dataset))

            if test == 1:
                set_random_seed(seed)

                id_set = np.arange(self.labels[-1].shape[-1])

                # split the dataset into train and test
                train_size = int(0.7 * len(id_set)) if few_shot == -1 else few_shot
                val_size = int(0.8 * len(id_set)) - train_size
                test_size = len(id_set) - val_size - train_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(id_set, [train_size, val_size,
                                                                                                  test_size])

                mask = torch.zeros(self.labels[-1].shape[-1])

                mask[val_dataset] = 2
                mask[test_dataset] = 1

                self.masks.append(mask)

                cnt = 0
                for now_id in range(len(mask)):
                    if mask[now_id] == 0:
                        cnt += 1
                self.zeros.append(cnt)

                # 随机mask掉一部分train，即mask==0的，预测另一部分？

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 非mask的部分，预测masked的了
        now_dataset = self.labels[index]
        # sample from a distribution between 0.1-0.9\
        mask = self.masks[index]

        now_mask = gen_new_mask(mask, self.zeros[index])

        return now_dataset, now_mask


class DatasetUIC(Dataset):  # Dataset For Masked Auto Encoder
    def __init__(self, datasets, poi_datas, seed=42, few_shot=-1, test=0, FT=0):
        super().__init__()

        self.labels = []
        self.task_embedding = []
        # 首先 eval和 test 的部分一定是mask着的
        # 其次，train一部分是mask着的，一部分是没mask的
        self.masks = []
        self.zeros = []
        self.test = test
        self.top_poi = 3
        self.FT = FT
        self.poi_distribution = []
        self.emb_distribution = []
        for dataset, text_embedding in tqdm(poi_datas):
            dataset = normalize(dataset)
            self.poi_distribution.append(torch.FloatTensor(dataset))
            self.emb_distribution.append(torch.FloatTensor(text_embedding))

        self.poi_distribution = torch.stack(self.poi_distribution, dim=0)
        self.emb_distribution = torch.stack(self.emb_distribution, dim=0)

        for dataset, text_embedding in tqdm(datasets):
            dataset = normalize(dataset)
            self.labels.append(torch.FloatTensor(dataset))
            self.task_embedding.append(torch.FloatTensor(text_embedding))

            if test == 1:
                set_random_seed(seed)

                id_set = np.arange(self.labels[-1].shape[-1])

                # split the dataset into train and test
                train_size = int(0.7 * len(id_set)) if few_shot == -1 else few_shot
                val_size = int(0.8 * len(id_set)) - train_size
                test_size = len(id_set) - val_size - train_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(id_set, [train_size, val_size,
                                                                                                  test_size])

                mask = torch.zeros(self.labels[-1].shape[-1])

                mask[val_dataset] = 2
                mask[test_dataset] = 1

                self.masks.append(mask)

                cnt = 0
                for now_id in range(len(mask)):
                    if mask[now_id] == 0:
                        cnt += 1
                self.zeros.append(cnt)

        self.task_embedding = torch.stack(self.task_embedding, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 非mask的部分，预测masked的了
        now_dataset = self.labels[index]
        # sample from a distribution between 0.1-0.9
        if self.test == 1:
            mask = self.masks[index]
        else:
            # mask_rate = np.random.uniform(0.1, 0.9)
            # mask = torch.rand(len(now_dataset)) < mask_rate
            mask = gen_mask(len(now_dataset))

        embedding_now = self.task_embedding[index]
        similarity = torch.nn.functional.cosine_similarity(embedding_now, self.emb_distribution, dim=-1)

        now = 0
        if self.test == -1:
            now = 1
        top_text_value = torch.argsort(similarity, descending=True)[now:now + self.top_poi]
        top_text_loc = self.poi_distribution[top_text_value]

        if self.FT == 0:
            return now_dataset, mask, top_text_loc
        else:
            now_mask = gen_new_mask(mask, self.zeros[index])
            return now_dataset, now_mask, top_text_loc


class DatasetUICPre(Dataset):  # Dataset For Urban In-Context Learning
    def __init__(self, datasets, pois, seed=42, few_shot=-1, test=0, top_poi=3):
        super().__init__()

        self.labels = []
        self.pois = []
        self.masks = []
        self.test = test
        self.top_poi = top_poi

        for dataset in tqdm(pois):
            dataset = normalize(dataset)
            self.pois.append(torch.FloatTensor(dataset))

        self.pois = torch.stack(self.pois, dim=0)  # [dataset_size, region_cnt]

        for dataset in tqdm(datasets):
            dataset = normalize(dataset)
            self.labels.append(torch.FloatTensor(dataset))

            if test == 1:
                set_random_seed(seed)

                id_set = np.arange(self.labels[-1].shape[-1])

                # split the dataset into train and test
                train_size = int(0.7 * len(id_set)) if few_shot == -1 else few_shot
                val_size = int(0.8 * len(id_set)) - train_size
                test_size = len(id_set) - val_size - train_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(id_set, [train_size, val_size,
                                                                                                  test_size])

                mask = torch.zeros(self.labels[-1].shape[-1])

                mask[val_dataset] = 2
                mask[test_dataset] = 1

                self.masks.append(mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        # 非mask的部分，预测masked的了
        now_dataset = self.labels[index]
        # sample from a distribution between 0.1-0.9
        if self.test == 1:
            mask = self.masks[index]
        else:
            mask_rate = np.random.uniform(0.1, 0.9)
            mask = torch.rand(len(now_dataset)) < mask_rate

        # calculate the similarity between the now_dataset and the poi using cosine similarity
        masked_now = now_dataset * mask
        masked_pois = self.pois * mask
        similarity = torch.nn.functional.cosine_similarity(masked_now, masked_pois, dim=-1)

        # select the top_poi pois with the highest similarity
        now = 0
        if self.test == -1:
            now = 1
        top_poi_value = torch.argsort(similarity, descending=True)[now:now + self.top_poi]
        top_poi_loc = self.pois[top_poi_value]
        return now_dataset, mask, top_poi_loc


class DatasetMDT(Dataset):  # Dataset For Masked Diffusion Transformer
    def __init__(self, datasets, seed=42, test=0):
        super().__init__()

        self.labels = []
        # 首先 eval和 test 的部分一定是mask着的
        # 其次，train一部分是mask着的，一部分是没mask的
        self.masks = []
        self.test = test

        for dataset in tqdm(datasets):
            dataset = normalize(dataset)
            self.labels.append(torch.FloatTensor(dataset))

            if test == 1:
                set_random_seed(seed)

                id_set = np.arange(self.labels[-1].shape[-1])

                # split the dataset into train and test
                train_size = int(0.7 * len(id_set))
                val_size = int(0.8 * len(id_set)) - train_size
                test_size = len(id_set) - val_size - train_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(id_set, [train_size, val_size,
                                                                                                  test_size])

                mask = torch.zeros(self.labels[-1].shape[-1])

                mask[val_dataset] = 2
                mask[test_dataset] = 1

                self.masks.append(mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 非mask的部分，预测masked的了
        now_dataset = self.labels[index]
        # sample from a distribution between 0.1-0.9
        if self.test == 1:
            mask = self.masks[index]
        else:
            mask_rate = np.random.uniform(0.1, 0.9)
            mask = torch.rand(len(now_dataset)) < mask_rate

        return now_dataset, mask


class DatasetUICFinal(Dataset):  # Dataset For Urban In-Context Learning
    def __init__(self, datasets, pois, seed=42, test=0, top_poi=3):
        super().__init__()

        self.labels = []
        self.embeddings = []

        self.pois = []
        self.auc_embeddings = []

        self.masks = []
        self.test = test
        self.top_poi = top_poi

        for dataset, text_embedding in tqdm(pois):
            dataset = normalize(dataset)
            self.pois.append(torch.FloatTensor(dataset))
            self.auc_embeddings.append(torch.FloatTensor(text_embedding))

        self.pois = torch.stack(self.pois, dim=0)
        self.auc_embeddings = torch.stack(self.auc_embeddings, dim=0)

        for dataset, text_embedding in tqdm(datasets):
            dataset = normalize(dataset)
            self.labels.append(torch.FloatTensor(dataset))
            self.embeddings.append(torch.FloatTensor(text_embedding))

            if test == 1:
                set_random_seed(seed)

                id_set = np.arange(self.labels[-1].shape[-1])

                # split the dataset into train and test
                train_size = int(0.7 * len(id_set))
                val_size = int(0.8 * len(id_set)) - train_size
                test_size = len(id_set) - val_size - train_size

                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(id_set, [train_size, val_size,
                                                                                                  test_size])

                mask = torch.zeros(self.labels[-1].shape[-1])

                mask[val_dataset] = 2
                mask[test_dataset] = 1

                self.masks.append(mask)

        self.embeddings = torch.stack(self.embeddings, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 非mask的部分，预测masked的了
        now_dataset = self.labels[index]
        # sample from a distribution between 0.1-0.9
        if self.test == 1:
            mask = self.masks[index]
        else:
            mask_rate = np.random.uniform(0.1, 0.9)
            mask = torch.rand(len(now_dataset)) < mask_rate

        # select top_poi pois with the highest similarity between the now_dataset and the poi in the unmasked ones

        # calculate the similarity between the now_dataset and the poi using cosine similarity
        masked_now = now_dataset * mask
        masked_pois = self.pois * mask
        similarity = torch.nn.functional.cosine_similarity(masked_now, masked_pois, dim=-1)

        # select the top_poi pois with the highest similarity
        now = 0
        if self.test == -1:
            now = 1
        top_poi_value = torch.argsort(similarity, descending=True)[now:now + self.top_poi]
        top_poi_loc = self.pois[top_poi_value]

        # calculate the similarity between the now_embedding and the text_embedding using cosine similarity
        embedding_now = self.embeddings[index]
        similarity = torch.nn.functional.cosine_similarity(embedding_now, self.auc_embeddings, dim=-1)

        now = 0
        if self.test == -1:
            now = 1
        top_text_value = torch.argsort(similarity, descending=True)[now:now + self.top_poi]
        top_text_loc = self.pois[top_text_value]

        return now_dataset, mask, top_poi_loc, top_text_loc

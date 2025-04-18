import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error)
from datetime import datetime


def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        default="Manhattan",
        choices=["Chicago", "Manhattan"],
        help="which dataset",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="cuda:2")

    parser.add_argument(
        "--source",
        type=str,
        default="UrbanCLIP",
        choices=["POI", "UrbanCLIP", "MGFN"],
        help="source model",
    )

    parser.add_argument(
        "--use_embedding",
        type=int,
        default=2,
        help="use embedding or not"
    )

    parser.add_argument(
        "--indicator",
        type=str,
        default="nightlight",
        choices=["carbon", "population", "nightlight", "taxi", "poi", "crime"],
        help="indicator",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="d1",
        help="choose model"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="unist",
        help="choose name"
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=0.01,
        help="weight decay"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch size")

    parser.add_argument(
        "--epoch_num",
        type=int,
        default=100000,
        help="epoch number")

    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="random seed"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="checkpoint path",
    )

    parser.add_argument(
        "--few_shot",
        type=int,
        default=10,
        choices=[-1, 3, 5, 10, 20, 50, 100],
        help="few shot k parameters",
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/experiments.csv",
        help="logging path",
    )

    parser.add_argument(
        "--test",
        type=int,
        default=0,
        help="only test the model?",
    )
    return parser


def calc(phase, epoch, all_predicts, all_y, loss, args, model="Linear", st_time=0):
    metrics = {}
    if loss is not None:
        metrics["loss"] = loss
    if torch.isnan(torch.tensor(all_y)).any():
        print("NaN detected in all_y")
    if torch.isnan(torch.tensor(all_predicts)).any():
        print("NaN detected in all_predicts")
    metrics["mse"] = mean_squared_error(all_y, all_predicts)
    metrics["r2"] = r2_score(all_y, all_predicts)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)
    metrics["pcc"] = np.corrcoef(all_y, all_predicts)[0, 1]

    if st_time == 0:
        st_time = datetime.now()
    # print(metrics)
    print(
        f"{phase} Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    if phase == "Test":
        output = {
            "model": model,
            "city": args.dataset,
            "indicator": epoch,
            "region_embedding": args.source,
            "use_embedding": args.use_embedding if model != "Linear" else None,
            'mae': f"{round(metrics['mae'], 4):.4f}",
            'rmse': f"{round(metrics['rmse'], 4):.4f}",
            'pcc': f"{round(metrics['pcc'], 4):.4f}",
            'r2': f"{round(metrics['r2'], 4):.4f}",
            "learning_rate": args.lr,
            "model_size": args.model if model != "Linear" else None,
            "weight_decay": args.wd,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "few_shot": args.few_shot,
            "run_time": datetime.now() - st_time,
        }
        df = pd.DataFrame(output, index=[0])
        log_file = args.log_path
        if not os.path.exists(log_file):
            df.to_csv(log_file, index=False, mode='w', header=True)
        else:
            df.to_csv(log_file, index=False, mode='a', header=False)
        # print("save here")
    return metrics


def process_outliers(arr):
    data_array = np.array(arr)

    mean_value = np.mean(data_array)
    std_value = np.std(data_array)

    z_scores = (data_array - mean_value) / std_value

    threshold = 3

    outliers = np.abs(z_scores) > threshold

    data_array[outliers] = mean_value

    if np.any(np.isnan(data_array)):
        print("fuck here")
        print(arr)
    return data_array

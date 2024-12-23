# TODO before running
# - check the data path
# - check that path develooment repo is locally installed

# inf1.6xlarge 24 vCPUs, 48GB RAM

import os
import pandas as pd
import logging
from argparse import ArgumentParser

from aeon.datasets import load_classification
from torch import (
    Tensor,
    device,
    set_default_device,
    tensor,
    logit,
    Generator,
    cuda,
    float32,
)
from torch.utils.data import DataLoader, TensorDataset
from enum import Enum

from development.so import so
from development.go import go
from development.gl import gl

from src.attention_development import (
    AttentionDevelopmentConfig,
    GroupConfig,
)
from src.logger import initialise_logger
from src.models import PDevBaggingBiLSTM
from src.training import (
    train_model,
    to_one_hot,
    to_soft_probabilities,
    train_sample_accuracy,
    test_sample_accuracy,
)


class Datasets(Enum, str):
    WalkingSittingStanding = "WalkingSittingStanding"


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--max_channels", type=int, default=4)
    parser.add_argument("--max_dim", type=int, default=4)
    parser.add_argument("--max_heads", type=int, default=4)
    parser.add_argument("--max_hidden_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--logs_filename", type=str, default="grid_search.log")
    parser.add_argument(
        "--dataset", type=Datasets, default=Datasets.WalkingSittingStanding
    )

    args = parser.parse_args()

    max_channels = args.max_channels
    max_dim = args.max_dim
    max_heads = args.max_heads
    max_hidden_size = args.max_hidden_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    logs_dir = args.logs_dir
    logs_filename = args.logs_filename
    dataset_name = parser.dataset

    logger = initialise_logger(
        logs_dir, logs_filename, logs_filename.split(".log")[0], logging.INFO
    )

    data, labels = load_classification(dataset_name)

    # as specified on UAE website
    tsx_train, y_train_labels = data[:7352], labels[:7352]
    tsx_test, y_test_labels = data[7352:], labels[7352:]

    device = device("cuda" if cuda.is_available() else "cpu")
    set_default_device(device)
    logger.info(f"Default device set to {device}")

    # labels to smoothed log-probs
    y_train = to_soft_probabilities(to_one_hot(y_train_labels.astype(float)))
    y_test = to_soft_probabilities(to_one_hot(y_test_labels.astype(float)))

    tsx_train = Tensor(tsx_train).swapaxes(1, 2).to(device)
    tsx_test = Tensor(tsx_test).swapaxes(1, 2).to(device)

    # Convert target to tensors
    y_train = logit(tensor(y_train, dtype=float32)).to(device)
    y_test = logit(tensor(y_test, dtype=float32)).to(device)

    # Create train and test data loaders
    train_dataset = TensorDataset(tsx_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=Generator(device=device),
    )
    test_dataset = TensorDataset(tsx_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=Generator(device=device),
    )

    # hyperparameters
    group_range = [so, go, gl]
    channel_range = range(2, max_channels)
    dim_range = range(3, max_dim)
    hidden_size_range = range(4, max_hidden_size)
    heads_range = range(2, max_heads)
    lstm_is_bidirectional = [True, False]

    res = pd.DataFrame(
        columns=["train_acc", "test_acc"],
        index=pd.MultiIndex.from_product(
            [
                [g.__name__ for g in group_range],
                channel_range,
                dim_range,
                heads_range,
                hidden_size_range,
                lstm_is_bidirectional,
            ],
            names=[
                "group",
                "nchannels",
                "dim",
                "n_heads",
                "hidden_size",
                "bidirectional",
            ],
        ),
    )

    res = res.sample(frac=1)

    for (
        g_name,
        nchannels,
        dim,
        n_heads,
        hidden_size,
        bidirectional,
    ) in res.index.to_series():
        group = [g for g in group_range if g.__name__ == g_name][0]
        if not hidden_size % n_heads == 0:
            continue

        logger.info(
            f">>> Starting grid search with : "
            f"group={group.__name__}, "
            f"nchannels={nchannels}, "
            f"dim={dim}, "
            f"hidden_size={hidden_size}, "
            f"nheads={n_heads}, "
            f"bidirectional={bidirectional}"
        )

        multidev_config = AttentionDevelopmentConfig(
            n_heads=n_heads,
            groups=[
                GroupConfig(group=group, dim=dim, channels=nchannels)
                for _ in range(n_heads)
            ],
        )

        model = PDevBaggingBiLSTM(
            dropout=0.05,
            input_dim=3,
            hidden_dim=hidden_size,
            out_dim=6,
            multidev_config=multidev_config,
            bidirectional=bidirectional,
        ).to(device)

        model.train()

        model, lossx = train_model(model, train_loader, n_epochs, learning_rate)

        train_acc = train_sample_accuracy(model, train_loader, y_train)

        test_acc = test_sample_accuracy(model, test_loader, y_test)

        logger.info(f"Train accuracy: {train_acc} | Test accuracy: {test_acc}")

        res.loc[
            (group.__name__, nchannels, dim, n_heads, hidden_size, bidirectional), :
        ] = [
            train_acc,
            test_acc,
        ]

        res.to_csv(os.path.join(logs_dir, "grid_search_results.csv"))

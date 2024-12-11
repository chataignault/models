from torch import nn, Tensor, argmax, sum
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


from .logger import initialise_logger


def train_model(fm: nn.Module, train_loader: DataLoader, nepochs: int, learning_rate: float):
    logger = initialise_logger()
    fm.train()
    optimizer = Adam(fm.parameters(), lr=learning_rate)
    lossx = []
    for epoch in tqdm(range(nepochs)):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = fm(x)
            loss = sum((y - y_hat) ** 2) / len(y)
            loss.backward()
            lossx.append(loss.item())
            optimizer.step()

        logger.info(f"Epoch : {epoch} | Loss {lossx[-1]} | gradient {0.}")

    return fm, lossx


def to_one_hot(y, num_classes=6):
    return np.eye(num_classes)[y.astype(int)]


# Convert to soft probabilities
def to_soft_probabilities(y_one_hot, temperature=0.2):
    exp_values = np.exp(y_one_hot / temperature)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def train_sample_accuracy(fm: nn.Module, train_loader: DataLoader, y_train: Tensor):
    fm.eval()
    n_true_prediction = 0
    preds, trues = [], []
    for x, y in train_loader:
        y_hat = fm(x)
        y_pred = argmax(y_hat, axis=1)
        y_true = argmax(y, axis=1)
        n_true_prediction += sum(y_pred == y_true).detach().cpu().numpy()
        preds = np.concatenate([preds, y_pred.detach().cpu().numpy()])
        trues = np.concatenate([trues, y_true.detach().cpu().numpy()])

    return n_true_prediction / len(y_train)


def test_sample_accuracy(fm: nn.Module, test_loader: DataLoader, y_test: Tensor):
    fm.eval()
    n_true_prediction = 0
    preds, trues = [], []
    for x, y in test_loader:
        y_hat = fm(x)
        y_pred = argmax(y_hat, axis=1)
        y_true = argmax(y, axis=1)
        n_true_prediction += sum(y_pred == y_true).detach().cpu().numpy()
        preds = np.concatenate([preds, y_pred.detach().cpu().numpy()])
        trues = np.concatenate([trues, y_true.detach().cpu().numpy()])

    return n_true_prediction / len(y_test)

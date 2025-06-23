import numpy as np
from numpy import random

from .svd import golub_kahan_svd


def generate_sample_conditionned(X: np.array, keep: int, golub_svd: bool = True):
    # can't have zero columns
    X += 0.5 * np.random.randn(*X.shape)

    if golub_svd:
        U, S, V = golub_kahan_svd(X.copy(), max_step=2)
    else:
        U, S, V = naive_svd(X.copy())

    s = np.diag(S).copy()
    s[keep:] = 0.0

    s = np.diag(s)

    samples = U[:, :keep] @ s[:keep, :keep] @ V.T[:keep, :]

    sample = samples[0]
    sample = sample.reshape(28, 28)
    return sample


def get_samples_and_normalize(data: np.array, cl: np.array, n_samples: int, label: int):
    X = [pil for (pil, l) in zip(data, cl) if l == label]
    X = np.array([np.array(pil).reshape(784) for pil in X]) / 25.0

    idx = random.choice(np.arange(len(X)), n_samples, replace=False)
    X = X[idx]
    return X

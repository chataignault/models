import pytest
import numpy as np
from numpy import random

from . import TOL

from src.em import update_pca_params


@pytest.mark.parametrize("d,q", [(3, 2), (5, 3), (8, 4)])
def test_einsum(d: int, q: int):
    random.seed(d)
    B = random.randn(d, q)
    C = random.randn(d, q)

    r = np.sum(np.einsum("dq,dq->d", B, C))

    assert np.abs(r - np.sum(np.diag(B @ C.T))) < TOL


@pytest.mark.parametrize("d,q", [(3, 2), (5, 3), (8, 4)])
def test_update_pca_params(d: int, q: int):
    random.seed(d)
    W = np.random.randn(d, q)
    S = np.random.randn(d, d)
    S = S + S.T
    s = np.random.randn() ** 2

    M_inv = np.linalg.inv(s * np.eye(q) + W.T @ W)

    W_naive = S @ W @ np.linalg.inv(s * np.eye(q) + M_inv @ W.T @ S @ W)
    s_naive = np.trace(S - S @ W @ M_inv @ W_naive.T) / d

    W_efficient, s_efficient = update_pca_params(W.copy(), S.copy(), s)

    assert np.linalg.norm(W_naive - W_efficient) < TOL
    assert np.abs(s_naive - s_efficient) < TOL

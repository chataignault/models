import pytest
import numpy as np
from numpy import random

from . import TOL
from src.svd import naive_svd, golub_kahan_step, golub_kahan_svd
from src.pure_qr import pure_QR


@pytest.mark.parametrize("n, m", [(5, 5), (7, 8), (10, 3), (20, 36)])
def test_naive_svd(n, m):
    random.seed(n)
    A = random.randn(n, m)

    # dependency with the pure QR algorithm
    S, V = pure_QR(
        A.T @ A,
        maxit=100,
        tol=1e-4,
        trid=False,
        track=False,
        shift=False,
    )

    # check orthonormality
    assert np.linalg.norm(V.T @ V - np.eye(m)) < TOL

    # check reconstruction from pure QR
    assert np.linalg.norm(A.T @ A - V @ S @ V.T) < TOL

    U, S, V = naive_svd(A)

    # check S is digonal
    assert np.linalg.norm(S - np.diag(np.diag(S))) < TOL

    # check reconstruction error
    assert np.allclose(A, U @ S @ V.T)


@pytest.mark.parametrize("n", [5, 8, 10, 20])
def test_golub_step(n):
    random.seed(n)

    B = random.randn(n, n)
    B[np.tril_indices(n, k=-1)] = 0.0
    B[np.triu_indices(n, k=2)] = 0.0

    U, Bd, V = golub_kahan_step(B)

    # assert Bd is upper bidiagonal diagonal
    assert np.min(np.abs(Bd[np.tril_indices(n, k=-1)])) == 0.0
    assert np.min(np.abs(Bd[np.triu_indices(n, k=2)])) == 0.0

    # assert orthonormality
    assert np.linalg.norm(np.eye(n) - U.T @ U) < TOL
    assert np.linalg.norm(np.eye(n) - V.T @ V) < TOL

    # assert reconstruction error
    assert np.allclose(B, U @ Bd @ V.T)


@pytest.mark.parametrize("n, m", [(5, 5), (7, 8), (10, 3), (20, 36)])
def test_golub_kahan_svd(n, m):
    random.seed(5 * n)
    A = random.randn(n, m)

    U, S, V = golub_kahan_svd(A)

    # check S is diagonal
    assert np.linalg.norm(S - np.diag(np.diag(S))) < TOL

    # check orthonormality
    assert np.linalg.norm(np.eye(n) - U @ U.T) < TOL
    assert np.linalg.norm(np.eye(m) - V @ V.T) < TOL

    # check reconstruction error is close
    assert np.allclose(A, U @ S @ V.T)

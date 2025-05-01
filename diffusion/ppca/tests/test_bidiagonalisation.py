import pytest
import numpy as np
from numpy import random

from . import TOL
from src.bidiagonalisation import golub_kahan, golub_kahan_full


@pytest.mark.parametrize("n, m", [(5, 5), (7, 8), (10, 3), (20, 36)])
def test_bidiagonalisation_full(n, m):
    random.seed(n)
    A = random.randn(n, m)

    U, B, V = golub_kahan_full(A.copy())

    # check upper bi-diagonal
    assert np.linalg.norm(np.tril(B, k=-1)) < TOL
    assert np.linalg.norm(np.triu(B, k=2)) < TOL

    # check orthonormal matrices
    assert np.linalg.norm(np.eye(n) - U @ U.T) < TOL
    assert np.linalg.norm(np.eye(m) - V @ V.T) < TOL

    # check reconstruction
    assert np.allclose(A, U @ B @ V.T)


@pytest.mark.parametrize("n, m", [(5, 5), (7, 8), (10, 3), (20, 36)])
def test_bidiagonalisation(n, m):
    random.seed(n)
    A = random.randn(n, m)

    U, B, V = golub_kahan(A.copy())

    # check upper bi-diagonal
    assert np.linalg.norm(np.tril(B, k=-1)) < TOL
    assert np.linalg.norm(np.triu(B, k=2)) < TOL

    # check orthonormal matrices
    assert np.linalg.norm(np.eye(n) - U @ U.T) < TOL
    assert np.linalg.norm(np.eye(m) - V @ V.T) < TOL

    # check reconstruction
    assert np.allclose(A, U @ B @ V.T)

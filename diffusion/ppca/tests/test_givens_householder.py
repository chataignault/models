import pytest
import numpy as np
from numpy import random
from scipy.linalg import expm

from . import TOL
from src.svd import (
    givens,
    apply_givens_left_ubi,
    apply_givens_left,
    apply_givens_right_ubi,
    apply_givens_right,
    eigenvalues_2_2,
)


@pytest.mark.parametrize("n, i", [(3, 0), (4, 1), (5, 1), (6, 3)])
def test_givens_left_ubi(n, i):
    random.seed(41 * n)
    A = random.randn(n, n)

    # make upper bidiagonal
    A[np.tril_indices(n, k=-1)] = 0.0
    A[np.triu_indices(n, k=2)] = 0.0

    # at some step of the algorithm
    A[i + 1, i] = random.randn(1).item()

    B = A.copy()

    c, s = givens(A[i, i], A[i + 1, i])
    apply_givens_left_ubi(A, i, i + 1, c, s)

    assert A.shape == (n, n)
    assert A[i + 1, i] < TOL
    assert np.min(np.abs(np.diag(A))) > 0.0
    assert np.min(A[np.tril_indices(n, k=-1)]) == 0.0

    # assert operation equals to left multiplication with Givens rotation
    Q = np.eye(n)
    Q[i, i], Q[i + 1, i + 1] = c, c
    Q[i, i + 1], Q[i + 1, i] = s, -s

    assert np.allclose(Q.T @ B, A)


@pytest.mark.parametrize("n, i", [(3, 0), (4, 1), (5, 1), (6, 3)])
def test_givens_left(n, i):
    random.seed(41 * n)
    A = random.randn(n, n)

    # make upper bidiagonal
    A[np.tril_indices(n, k=-1)] = 0.0
    A[np.triu_indices(n, k=2)] = 0.0

    # at some step of the algorithm
    A[i + 1, i] = random.randn(1).item()

    B = A.copy()

    c, s = givens(A[i, i], A[i + 1, i])
    apply_givens_left(A, i, i + 1, c, s)

    assert A.shape == (n, n)
    assert A[i + 1, i] < TOL
    assert np.min(np.abs(np.diag(A))) > 0.0

    # because of numerical errors on zero operations,
    # is not perfectly zero,
    # hence using specialised algorithm of upper bidiagonal matrices
    assert np.min(A[np.tril_indices(n, k=-1)]) < TOL

    # assert operation equals to left multiplication with Givens rotation
    Q = np.eye(n)
    Q[i, i], Q[i + 1, i + 1] = c, c
    Q[i, i + 1], Q[i + 1, i] = s, -s

    assert np.allclose(Q.T @ B, A)


@pytest.mark.parametrize("n, i", [(4, 0), (5, 1), (6, 3)])
def test_givens_right_ubi(n, i):
    random.seed(41 * n)
    A = random.randn(n, n)

    # make upper bidiagonal
    A[np.tril_indices(n, k=-1)] = 0.0
    A[np.triu_indices(n, k=2)] = 0.0

    # at some step of the algorithm
    A[i, i + 2] = random.randn(1).item()

    B = A.copy()

    c, s = givens(A[i, i + 1], A[i, i + 2])

    apply_givens_right_ubi(A, i + 1, i + 2, c, s)

    assert A.shape == (n, n)
    assert A[i, i + 2] < TOL
    assert np.min(np.abs(np.diag(A))) > 0.0
    assert np.min(A[np.triu_indices(n, k=2)]) == 0.0

    # assert operation equals to left multiplication with Givens rotation
    Q = np.eye(n)
    Q[i + 1, i + 1], Q[i + 2, i + 2] = c, c
    Q[i + 1, i + 2], Q[i + 2, i + 1] = s, -s

    assert np.allclose(B @ Q, A)


@pytest.mark.parametrize("n, i", [(4, 0), (5, 1), (6, 3)])
def test_givens_right(n, i):
    random.seed(41 * n)
    A = random.randn(n, n)

    # make upper bidiagonal
    A[np.tril_indices(n, k=-1)] = 0.0
    A[np.triu_indices(n, k=2)] = 0.0

    # at some step of the algorithm
    A[i, i + 2] = random.randn(1).item()

    B = A.copy()

    c, s = givens(A[i, i + 1], A[i, i + 2])

    apply_givens_right(A, i + 1, i + 2, c, s)

    assert A.shape == (n, n)
    assert A[i, i + 2] < TOL
    assert np.min(np.abs(np.diag(A))) > 0.0

    # round-off errors
    assert np.min(A[np.triu_indices(n, k=2)]) < TOL

    # assert operation equals to left multiplication with Givens rotation
    Q = np.eye(n)
    Q[i + 1, i + 1], Q[i + 2, i + 2] = c, c
    Q[i + 1, i + 2], Q[i + 2, i + 1] = s, -s

    assert np.allclose(B @ Q, A)


@pytest.mark.parametrize("l1, l2", [(1.0, 2.0), (3.0, 1.0), ((5.0, 2.5))])
def test_2_2_eigevalues(l1, l2):
    random.seed(37)
    u = random.randn()
    U = expm(np.array([[0.0, u], [-u, 0.0]]))
    assert np.linalg.norm(U.T @ U - np.eye(2)) < TOL
    A = U @ np.diag([l1, l2]) @ U.T
    x1, x2 = eigenvalues_2_2(A)

    assert (np.min([x1, x2]) - np.min([l1, l2])) < TOL
    assert (np.max([x1, x2]) - np.max([l1, l2])) < TOL


@pytest.mark.parametrize("n, m", [(4, 7), (3, 5), (6, 4)])
def test_householder_reflection(n, m): ...

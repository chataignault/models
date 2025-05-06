import numpy as np
from typing import Tuple

from .pure_qr import pure_QR
from src.bidiagonalisation import golub_kahan_full


def naive_svd(A: np.ndarray, maxit: int = 10, tol: float = 1e-4):
    """
    An unefficient implementation of the Singular Value Decomposition.
    Reformulates the problem to an eigenvalue decomposition of the covariance matrix
    which is a hard and instable numerical process.
    """
    n, m = A.shape
    cov = A.T.dot(A)
    S, V = pure_QR(cov, maxit=maxit, tol=tol, trid=False, track=False, shift=False)
    s = np.sqrt(np.diag(S))
    r = min(n, m)
    s = np.clip(s[:r], min=1e-5)
    V = V[:, :r]
    U = np.divide(A @ V, s)
    return U, np.diag(s), V


def givens(a: float, b: float) -> Tuple[float, float]:
    """
    Givens rotation computation that prevents from overflow
    """
    if b == 0.0:
        return 1.0, 0.0

    if abs(b) > abs(a):
        t = -a / b
        s = 1.0 / np.sqrt(1.0 + t**2)
        return t * s, s
    else:
        t = -b / a
        c = 1.0 / np.sqrt(1.0 + t**2)
        return c, t * c


def eigenvalues_2_2(A: np.ndarray) -> Tuple[float, float]:
    """
    Compute the eigenvalues of a 2-by-2 matrix
    by finding roots of the characteristic polynomial
    """
    x, y, z, t = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    c = x * t - y * z
    b = -(x + t)
    d = b**2 - 4 * c
    assert d > 0.0
    d = np.sqrt(d)
    return (-b + d) / 2.0, (-b - d) / 2.0


def apply_givens_left_ubi(A: np.ndarray, i: int, k: int, c: int, s: int):
    """
    Apply Givens rotation on A between rows i and k to the left
    Assumes that A is upper-bidiagonal
    """
    r = A[k, i : (k + 3)].copy()
    A[k, i : (i + 3)] = c * A[k, i : (i + 3)] + s * A[i, i : (i + 3)]
    A[i, i : (k + 3)] = c * A[i, i : (k + 3)] - s * r


def apply_givens_right_ubi(A: np.ndarray, i: int, k: int, c: float, s: float):
    """
    Apply Givens rotation on A between rows i and k to the right
    Assumes that A is upper-bidiagonal
    """
    r = A[i : (k + 3), (k + 1)].copy()
    A[i : (i + 3), (k + 1)] = c * A[i : (i + 3), (k + 1)] + s * A[i : (i + 3), (i + 1)]
    A[i : (k + 3), (i + 1)] = c * A[i : (k + 3), (i + 1)] - s * r


def apply_givens_left(A: np.ndarray, i: int, k: int, c: int, s: int):
    """
    Apply Givens rotation on A between rows i and k to the left
    """
    A[[i, k], :] = np.array([[c, -s], [s, c]]) @ A[[i, k], :]


def apply_givens_right(A: np.ndarray, i: int, k: int, c: float, s: float):
    """
    Apply Givens rotation on A between rows i and k to the right
    """
    A[:, [i, k]] = A[:, [i, k]] @ np.array([[c, s], [-s, c]])


def golub_kahan_step(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD step assuming B is upper bidiagonal
    with no zeros in either diagonal or superdiagonal
    """
    n, m = B.shape
    U, V = np.eye(n), np.eye(m)

    # initialisation : find trailing eigenvalue
    T = B[-2:, -2:].T.copy() @ B[-2:, -2:].copy()
    l1, l2 = eigenvalues_2_2(T)
    t = T[-1, -1]
    mu = l1 if abs(t - l1) < abs(t - l2) else l2
    y, z = T[0, 0] - mu, T[0, 1]

    # apply Givens rotations
    for k in range(n - 1):
        c, s = givens(y, z)
        apply_givens_right_ubi(B, k, k + 1, c, s)
        apply_givens_right(U, k, k + 1, c, s)
        y, z = B[k, k], B[k + 1, k]
        c, s = givens(y, z)
        apply_givens_left_ubi(B, k, k + 1, c, s)
        apply_givens_left(V, k, k + 1, c, s)
        if k < n - 2:
            y, z = B[k, k + 1], B[k, k + 2]

    return U, B, V


def golub_kahan_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bidiagonalise, then SVD steps until criterion reached
    """
    U, B, V = golub_kahan_full(A.copy())

    for i in range(len(A)):
        ...

    return U, B, V

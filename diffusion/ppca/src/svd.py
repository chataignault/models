import numpy as np
from typing import Tuple

from .pure_qr import pure_QR
from bidiagonalisation import golub_kahan_full


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


def golub_kahan_step(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD step assuming B is upper bidiagonal
    with no zeros in either diagonal or superdiagonal
    """
    n, m = B.shape
    U, V = np.eye(n), np.eye(m)

    # find trailing eigenvalue
    T = B.T @ B
    mu = 0.0
    y, z = T[0, 0] - mu
    z = T[0, 1]

    # apply Givens rotations
    for k in range(n - 1):
        ...

    return U, B, V


def golub_kahan_svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bidiagonalise, then SVD steps until criterion reached
    """
    U, B, V = golub_kahan_full(A.copy())

    return U, B, V

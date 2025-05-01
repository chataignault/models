import numpy as np
from typing import Tuple

from .pure_qr import compute_v_householder
from .pure_qr import householder


def golub_kahan(A) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Golu-Kahan bidiagonalisation with reduced sized matrices
    """
    n, m = A.shape
    l = min(n, m)
    U, V = np.eye(l, n), np.eye(m, l)

    for k in range(l - 1):
        # apply left householder and update U
        u = compute_v_householder(A[k:l, k])
        A[k:l, k:] -= 2.0 * u @ (np.conj(u.T) @ A[k:l, k:])
        U[k:l, :] -= 2.0 * u @ (np.conj(u.T) @ U[k:l, :])

        # apply right householder and update V
        v = compute_v_householder(A[k, (k + 1) :])
        A[k:, (k + 1) :] -= 2.0 * (A[k:, (k + 1) :] @ v) @ v.T
        print(v.shape)
        V[:, (k + 1) :] -= 2.0 * (V[:, (k + 1) :] @ v) @ v.T

    return U.T, A[:l, :l], V


def golub_kahan_full(A) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bidiagonalisation based on householder transforms
    Returns full size matrices
    """
    n, m = A.shape
    l = min(n, m)
    U, V = np.eye(n), np.eye(m)

    for k in range(l - 1):
        # apply left householder and update U
        u = compute_v_householder(A[k:, k])
        A[k:, k:] -= 2.0 * u @ (np.conj(u.T) @ A[k:, k:])
        U[k:, :] -= 2.0 * u @ (np.conj(u.T) @ U[k:, :])

        # apply right householder and update V
        v = compute_v_householder(A[k, (k + 1) :])
        A[k:, (k + 1) :] -= 2.0 * (A[k:, (k + 1) :] @ v) @ v.T
        V[:, (k + 1) :] -= 2.0 * (V[:, (k + 1) :] @ v) @ v.T

    u = compute_v_householder(A[l - 1 :, l - 1])
    A[l - 1 :, l - 1 :] -= 2.0 * u @ (np.conj(u.T) @ A[l - 1 :, l - 1 :])
    U[l - 1 :, :] -= 2.0 * u @ (np.conj(u.T) @ U[l - 1 :, :])

    if m > n:
        k = l - 1
        v = compute_v_householder(A[k, (k + 1) :])
        A[k:, (k + 1) :] -= 2.0 * (A[k:, (k + 1) :] @ v) @ v.T
        V[:, (k + 1) :] -= 2.0 * (V[:, (k + 1) :] @ v) @ v.T

    return U.T, A, V


if __name__ == "__main__":
    n, m = 5, 10
    from numpy import random

    A = random.randn(n, m)

    U, B, V = golub_kahan_full(A.copy())
    print(np.round(B, decimals=2))
    print(U.shape, V.shape, B.shape)
    print(np.round(U @ U.T, decimals=2))
    print(np.round(V @ V.T, decimals=2))
    print()

    print(np.linalg.norm(A - U @ B @ V.T))

    print(np.tril(B, k=-1))
    print(np.triu(B, k=2))
    print((np.linalg.norm(np.tril(B, k=-1)) + np.linalg.norm(np.triu(B, k=2))))
    assert (np.linalg.norm(np.tril(B, k=-1)) + np.linalg.norm(np.triu(B, k=2))) < 1e-8

import numpy as np
from typing import Tuple

from .pure_qr import compute_v_householder
from .pure_qr import householder


def golub_kahan_reduced_wide(
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Golu-Kahan bidiagonalisation with reduced sized matrices
    """
    n, m = A.shape
    is_thin = False
    if n > m:
        # work in the wide matrix setup
        A = A.T
        n, m = m, n
        is_thin = True

    U, V = np.eye(n), np.eye(m, n)

    for k in range(n - 1):
        # apply left householder and update U
        u = compute_v_householder(A[k:, k])
        A[k:, k:n] -= 2.0 * u @ (u.T @ A[k:n, k:n])
        U[k:, :] -= 2.0 * u @ (u.T @ U[k:n, :])

        # apply right householder and update V
        v = compute_v_householder(A[k, (k + 1) : n])
        A[k:, (k + 1) : n] -= 2.0 * (A[k:, (k + 1) : n] @ v) @ v.T
        V[:n, (k + 1) : n] -= 2.0 * (V[:n, (k + 1) : n] @ v) @ v.T

    u = compute_v_householder(A[n - 1 :, n - 1])
    A[n - 1 :, n - 1 : n] -= 2.0 * u @ (u.T @ A[n - 1 :, n - 1 : n])
    U[n - 1 :, :] -= 2.0 * u @ (u.T @ U[n - 1 :, :])

    # compute coefficients for the rest at the bottom of V
    V[n:, :] = A[:, n:].T @ U.T @ np.linalg.inv(A[:n, :n]).T

    A = A[:, :n]

    if is_thin:
        U, V = V.T, U.T
        A = A.T

    return U.T, A, V


def golub_kahan(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Golu-Kahan bidiagonalisation with reduced sized matrices
    """
    n, m = A.shape
    l = min([n, m])

    U, V = np.eye(n, l), np.eye(m, l)

    for k in range(l - 1):
        # apply left householder and update U
        u = compute_v_householder(A[k:l, k])
        A[k:l, k:l] -= 2.0 * u @ (u.T @ A[k:l, k:l])
        U[k:l, :l] -= 2.0 * u @ (u.T @ U[k:l, :l])

        # apply right householder and update V
        v = compute_v_householder(A[k, (k + 1) : l])
        A[k:l, (k + 1) : l] -= 2.0 * (A[k:l, (k + 1) : l] @ v) @ v.T
        V[:l, (k + 1) : l] -= 2.0 * (V[:l, (k + 1) : l] @ v) @ v.T

    u = compute_v_householder(A[l - 1 : l, l - 1])
    A[l - 1 : l, l - 1 : l] -= 2.0 * u @ (u.T @ A[l - 1 : l, l - 1 : l])
    U[l - 1 : l, :l] -= 2.0 * u @ (u.T @ U[l - 1 : l, :l])

    # compute coefficients for the rest at the bottom of V or U
    if n == l:
        U = U.T  # U was comuted as the transpose of the matrix we want
        V[n:, :] = A[:, n:].T @ U @ np.linalg.inv(A[:n, :n]).T
    else:
        U[:m, :m] = U[:m, :m].T
        U[m:, :] = A[m:, :] @ V @ np.linalg.inv(A[:m, :m])

    A = A[:l, :l]

    return U, A, V


def golub_kahan_full(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    n, m = 3, 10
    from numpy import random

    A = random.randn(n, m)

    print("Initial matrix")
    print(np.round(A, decimals=2))
    print()

    U, B, V = golub_kahan(A.copy())
    print("V")
    print(np.round(V, decimals=2))
    print()

    print("B")
    print(np.round(B, decimals=2))

    print("Orthonormality")
    print(np.round(V @ V.T, decimals=2))
    print(np.round(U.T @ U, decimals=2))
    print()

    print("Reconstruction")
    print(np.round(U @ B @ V.T, decimals=2))
    print(np.linalg.norm(A - U @ B @ V.T))

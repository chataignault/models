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
    j = i - 1 if i > 0 else i
    r = A[j : (k + 3), k].copy()
    A[j : (i + 3), k] = c * A[j : (i + 3), k] + s * A[j : (i + 3), i]
    A[j : (k + 3), i] = c * A[j : (k + 3), i] - s * r


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
    y, z = B[0, 0] ** 2 - mu, B[0, 0] * B[0, 1]

    # apply Givens rotations
    for k in range(n - 1):
        c, s = givens(y, z)
        apply_givens_right_ubi(B, k, k + 1, c, s)
        apply_givens_right(V, k, k + 1, c, s)
        y, z = B[k, k], B[k + 1, k]
        c, s = givens(y, z)
        apply_givens_left_ubi(B, k, k + 1, c, s)
        apply_givens_left(U, k, k + 1, c, s)
        if k < n - 2:
            y, z = B[k, k + 1], B[k, k + 2]

    return U.T, B, V


def golub_kahan_svd(
    A: np.ndarray, tol: float = 1e-10, max_step: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD algorithm from 'Matrix Computations' - Gene H. Golub, Charles F. Van Loan
    """
    n, m, transposed_to_thin = *A.shape, False
    if m > n:
        A = A.T
        n, m = m, n
        transposed_to_thin = True

    q = m - 1

    # bidiagonalise
    U, B, V = golub_kahan_full(A.copy())

    step = 0
    # apply golub steps until criterion is reached
    while q > 0 and (step < max_step or max_step < 0):
        step += 1
        while q > 0 and np.abs(B[q - 1, q]) < tol * (
            np.abs(B[q - 1, q - 1]) + np.abs(B[q, q])
        ):
            B[q - 1, q] = 0.0
            q -= 1

        p = 1
        # while q > 0 and B[q-1, q] == 0.0:
        #    q -= 1
        if q == 0:
            break
        # find largest submatrix with non-zero off-diag coefficients
        # starting from bottom right
        while B[q - p - 1, q - p] != 0.0 and q - p - 1 >= 0:
            p += 1
        for i in range(p):
            has_zero_diag = False
            if q + i < m and B[q + i, q + i] == 0.0:
                B[q + i, q + i + 1] == 0.0
                has_zero_diag = True
        if not has_zero_diag:
            # not efficient to return the matrices,
            # should modify U, B and V in place
            Ug, _, Vg = golub_kahan_step(B[q - p : (q + 1), q - p : (q + 1)].copy())
            Ug_ = np.eye(n)
            Ug_[q - p : (q + 1), q - p : (q + 1)] = Ug.copy()
            Vg_ = np.eye(m)
            Vg_[q - p : (q + 1), q - p : (q + 1)] = Vg.copy()
            U = U @ Ug_
            V = V @ Vg_
            B = Ug_.T @ B @ Vg_

    if transposed_to_thin:
        U, V = V, U
        B = B.T

    return U, B, V


if __name__ == "__main__":
    from numpy import random

    n, m = 4, 4
    A = random.randn(n, m)

    U, D, V = golub_kahan_svd(A.copy())

    svd = np.linalg.svdvals(A)

    # compare absolute svd values
    print(np.sort(np.round(svd, decimals=3)))
    print(np.sort(np.abs(np.round(np.diag(D), decimals=3))))

    # check reconstruction
    print(np.linalg.norm(U @ D @ V.T - A) / np.linalg.norm(A))

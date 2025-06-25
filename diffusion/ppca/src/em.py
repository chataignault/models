import numpy as np
from typing import Tuple
from src.svd import golub_kahan_svd


def compute_likelihood_pca(A: np.ndarray, W: np.ndarray, s: float) -> float:
    """
    First draft
    """
    d, N = A.shape
    C = W @ W.T
    C[np.diag_indices(d)] += s
    return -(
        np.log(np.clip(np.linalg.det(C), min=1e-3, max=1000.0))
        + np.trace(np.linalg.inv(C) @ A @ A.T / N)
    )


def inverse_dp(M: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a symmetric definite positive matrix
    Using the SVD decomposition
    """
    U, S, _ = np.linalg.svd(M, hermitian=True)
    assert np.min(S) > 0.0, "Matrix is not definite positive"
    return U @ np.diag(1.0 / S) @ U.T


def update_pca_params(
    W: np.ndarray, S: np.ndarray, s: float
) -> Tuple[np.ndarray, float]:
    """
    Apply aggregate E and M steps as defined in (29) - (30) of
    Tipping, Michael E., and Christopher M. Bishop.
    "Probabilistic principal component analysis."
    Journal of the Royal Statistical Society Series B: Statistical Methodology 61.3 (1999): 611-622.
    """
    d, q = W.shape
    M = W.T @ W
    M[np.diag_indices(q)] += s
    M_inv = inverse_dp(M)
    del M  # inverse of definite positive symmetric matrix with SVD
    SW = S @ W
    R = M_inv @ W.T @ SW
    R[np.diag_indices(q)] += s
    R = np.linalg.inv(R)
    W = SW @ R
    s = (np.sum(np.diag(S)) - np.sum(np.einsum("dq,dq->d", SW @ M_inv, W))) / d
    return W, s


def update_pca_params_naive(
    W: np.ndarray, S: np.ndarray, s: float
) -> Tuple[np.ndarray, float]:
    """
    Apply aggregate E and M steps as defined in (29) - (30) of
    Tipping, Michael E., and Christopher M. Bishop.
    "Probabilistic principal component analysis."
    Journal of the Royal Statistical Society Series B: Statistical Methodology 61.3 (1999): 611-622.
    """
    d, q = W.shape
    M = W.T @ W
    M[np.diag_indices(q)] += s
    M_inv = np.linalg.inv(M)
    W_next = S @ W @ np.linalg.inv(s * np.eye(q) + M_inv @ W.T @ S @ W)
    return W_next, (np.linalg.trace(S - S @ W @ M_inv @ W_next.T)) / d


def ppca(
    A: np.ndarray, q: int, tol: float = 1e-4, maxit: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Compute the first q principle components
    with iterative EM algorithm,
    which stops once likelihood improvement smaller than tol
    Assumes A is centered
    """
    d, n = A.shape
    assert q <= n, "Can't have more principal axes than data dimensionality"
    # vector containing the q approximate principal components
    W = np.random.randn(d, q)
    # average residual variance
    # sample covariance
    S = A @ A.T / n
    s = np.linalg.trace(S)
    ds = s

    i = 0
    while ds > tol:
        i += 1
        if i > maxit:
            print(f"EM algorithm hasn't converged but reach max iterations {maxit}")
            break

        # EM step
        W, s_new = update_pca_params(W, S, s)

        ds = s - s_new
        s = s_new

    return W, s


def rotate_W_orthogonal(W: np.ndarray) -> np.ndarray:
    """
    Since we don't have W components orthogonal from the algorithm
    (only the span)
    But from eigenvalue decomposition W orthogonal can be recovered
    """
    R_T = np.linalg.eig(W.T @ W).eigenvectors
    return W @ R_T

import numpy as np
from typing import Tuple


def compute_likelihood_pca(): ...


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
    M_inv = ...  # inverse of definite positive symmetric matrix with SVD
    SW = S @ W
    R = M_inv @ W.T @ SW
    R[np.diag_indices(q)] += s
    R = np.linalg.inv(R)
    W = SW @ R
    s = (np.sum(np.diag(S)) - np.sum(np.einsum("dq,dq->d", SW @ M_inv, W))) / d
    return W, s


def ppca(A: np.ndarray, q: int, tol: float = 1e-5) -> Tuple[np.ndarray, float]:
    """
    Compute the first q principle components
    with iterative EM algorithm,
    which stops once likelihood improvement smaller than tol
    """
    d, n = A.shape
    assert q <= n, "Can't have more principal axes than data dimensionality"
    # vector containing the q approximate principal components
    W = np.zeros((d, q))
    # average residual variance
    s = 1.0
    # sample mean
    mu = np.mean(A, axis=1)
    A -= mu
    # sample covariance
    S = A @ A.T / n
    # likelihood improvement
    l = compute_likelihood_pca()
    dl = 1.0

    while dl > tol:
        # EM step
        W, s = update_pca_params(W, S, s)

        # compute new likelihood and update threshold
        l_new = compute_likelihood_pca()
        dl = l - l_new
        l = l_new

    return W, s

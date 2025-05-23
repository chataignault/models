import numpy as np
from typing import Tuple


def compute_likelihood_pca(A: np.ndarray, W: np.ndarray, s: float) -> float:
    """
    First draft
    """
    d, N = A.shape
    C = W @ W.T
    C[np.diag_indices(d)] += s
    return -(
        np.log(np.clip(np.linalg.det(C), max=1000.0))
        + np.trace(np.linalg.inv(C) @ A @ A.T / N)
    )


def inverse_sdp(M: np.ndarray) -> np.ndarray: ...


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
    # M_inv = inverse_sdp(M); del M  # inverse of definite positive symmetric matrix with SVD
    M_inv = np.linalg.inv(M)
    del M
    SW = S @ W
    R = M_inv @ W.T @ SW
    R[np.diag_indices(q)] += s
    R = np.linalg.inv(R)
    W = SW @ R
    s = (np.sum(np.diag(S)) - np.sum(np.einsum("dq,dq->d", SW @ M_inv, W))) / d
    return W, s


def ppca(
    A: np.ndarray, q: int, tol: float = 1e-5, maxit: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Compute the first q principle components
    with iterative EM algorithm,
    which stops once likelihood improvement smaller than tol
    """
    d, n = A.shape
    assert q <= n, "Can't have more principal axes than data dimensionality"
    # vector containing the q approximate principal components
    W = np.random.randn(d, q)
    # average residual variance
    s = 1.0
    # sample mean
    mu = np.mean(A, axis=1).reshape(-1, 1)
    A -= mu
    # sample covariance
    S = A @ A.T / n
    # likelihood improvement
    l = compute_likelihood_pca(A, W, s)
    dl = 1.0

    i = 0
    while dl > tol:
        i += 1
        if i > maxit:
            print(f"EM algorithm hasn't converged but reach max iterations {maxit}")
            break

        # EM step
        W, s = update_pca_params(W, S, s)

        # compute new likelihood and update threshold
        l_new = compute_likelihood_pca(A, W, s)
        dl = l - l_new
        l = l_new

    return W, s

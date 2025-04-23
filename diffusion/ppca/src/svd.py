import numpy as np
from .pure_qr import pure_QR


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

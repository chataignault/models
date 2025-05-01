import pytest
import numpy as np
from numpy import random

from . import TOL
from src.svd import naive_svd
from src.pure_qr import pure_QR

@pytest.mark.parametrize("n, m", [(5, 5), (7, 8), (10, 3), (20, 36)])
def test_naive_svd(n, m):
    random.seed(n)
    A = random.randn(n, m)

    # dependency with the pure QR algorithm
    S, V = pure_QR(
        A.T @ A,
        maxit=100,
        tol=1e-4,
        trid=False,
        track=False,
        shift=False,
    )
    assert np.linalg.norm(V.T @ V - np.eye(m)) < TOL
    assert np.linalg.norm(A.T @ A - V @ S @ V.T) < TOL

    U, S, V = naive_svd(A)

    assert np.linalg.norm(S - np.diag(np.diag(S))) < TOL
    assert np.allclose(A, U @ S @ V.T)


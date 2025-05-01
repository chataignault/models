import pytest
import numpy as np
from numpy import random

from . import TOL
from src.bidiagonalisation import golub_kahan

@pytest.mark.parametrize("n, m", [(5, 5), (7, 8), (10, 3), (20, 36)])
def test_bidiagonalisation(n, m):
    random.seed(n)
    A = random.randn(n, m)

    U, B, V = golub_kahan(A)
    
    # check upper bi-diagonal
    assert np.linalg.norm(np.tril(B)) < TOL
    
    # check orthonormal matrices
    assert np.linalg.norm(np.eye(n) - U @ U.T) < TOL
    assert np.linalg.norm(np.eye(m) - V @ V.T) < TOL

    # check reconstruction
    assert np.allclose(A, U @ B @ V.T)


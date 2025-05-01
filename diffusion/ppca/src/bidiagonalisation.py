import numpy as np
from typing import Tuple

def golub_kahan(A) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m = A.shape
    return np.eye(n), A, np.eye(m)

import numpy as np
from . pure_qr import pure_QR

def naive_svd(A: np.ndarray, maxit:int=10, tol:float=1e-4):
    cov = A.T.dot(A)
    S, V = pure_QR(cov, maxit=maxit, tol=tol, trid=False, track=False, shift=False)
    s = np.sqrt(np.diag(S))
    s = np.clip(s, min=1e-5)
    U = np.divide(A @ V, s)

    return U, np.diag(s), V

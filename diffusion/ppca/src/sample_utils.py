import numpy as np
from numpy import random

from .svd import golub_kahan_svd, naive_svd
from .em import ppca, compute_likelihood_pca, rotate_W_orthogonal


def generate_sample_conditionned(
    X: np.array, q: int, golub_svd: bool = True, probabilistic=False
):
    """
    Generate a PCA sample with q orthogonal components from un-centered data X with shape (d, N)
    where d is the sample dimension and N the number of samples.

    """
    mu = np.mean(X, axis=0)
    X = X - mu

    if probabilistic:
        W, s = ppca(X.T, q)
        l = compute_likelihood_pca(X.T, W.copy(), s)
        print(f"Final parameters log-likelihood : {np.round(l, decimals=2)}")
        print("Recovering orthogonal components matrix...")
        W = rotate_W_orthogonal(W)
        sample = np.sum(W, axis=1)
    else:
        # * can't have zero columns with the current implementation !
        X += 0.5 * np.random.randn(*X.shape)

        if golub_svd:
            U, S, V = golub_kahan_svd(X, max_step=2)
        else:
            U, S, V = naive_svd(X)

        s = np.diag(S).copy()  # need assignements enabled
        s[q:] = 0.0

        s = np.diag(s)

        samples = U[:, :q] @ s[:q, :q] @ V.T[:q, :]

        sample = np.sum(samples, axis=0)

    return (sample + mu).reshape(28, 28)


def get_samples_and_normalize(data: np.array, cl: np.array, n_samples: int, label: int):
    X = [pil for (pil, l) in zip(data, cl) if l == label]
    X = np.array([np.array(pil).reshape(784) for pil in X]) / 25.0

    idx = random.choice(np.arange(len(X)), n_samples, replace=False)
    X = X[idx]
    return X

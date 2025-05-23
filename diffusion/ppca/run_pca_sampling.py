import numpy as np
from argparse import ArgumentParser
from numpy import random
from datasets import load_dataset
from matplotlib import pyplot as plt

from src.svd import naive_svd
from src.svd import golub_kahan_svd
from src.em import ppca


def generate_sample_conditionned(X: np.array, keep: int, golub_svd: bool = True):
    # can't have zero columns
    X += 0.5 * np.random.randn(*X.shape)

    if golub_svd:
        U, S, V = golub_kahan_svd(X.copy(), max_step=2)
    else:
        U, S, V = naive_svd(X.copy())

    s = np.diag(S).copy()
    s[keep:] = 0.0

    s = np.diag(s)

    samples = U[:, :keep] @ s[:keep, :keep] @ V.T[:keep, :]

    sample = samples[0]
    sample = sample.reshape(28, 28)
    return sample


def get_samples_and_normalize(data: np.array, cl: np.array, n_samples: int, label: int):
    X = [pil for (pil, l) in zip(data, cl) if l == label]
    X = np.array([np.array(pil).reshape(784) for pil in X]) / 25.0

    idx = random.choice(np.arange(len(X)), n_samples, replace=False)
    X = X[idx]
    return X


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_components", type=int, default=3)

    args = parser.parse_args()

    n_samples = args.n_samples
    n_components = args.n_components

    data_dict = load_dataset("mnist", num_proc=4)["train"]
    data, cl = data_dict["image"], data_dict["label"]

    fig, axs = plt.subplots(figsize=(10, 4), ncols=5, nrows=2)
    fig_ppca, axs_ppca = plt.subplots(figsize=(10, 4), ncols=5, nrows=2)

    for label in range(10):
        X = get_samples_and_normalize(data, cl, n_samples, label)

        sample = generate_sample_conditionned(X.copy(), n_components)

        W, s = ppca(X.T.copy(), 2)

        r, c = label // 5, label % 5
        axs[r, c].axis("off")
        axs[r, c].imshow(sample.copy(), cmap="gray")
        axs_ppca[r, c].axis("off")
        axs_ppca[r, c].imshow(W[:, 0].reshape(28, 28).copy(), cmap="gray")

    fig.suptitle("Generated samples with SVD algorithm")
    fig_ppca.suptitle("Generated samples with Proba PCA algorithm")
    plt.tight_layout()
    plt.show()
    fig.savefig("ppca_mnist.png", bbox_inches="tight")

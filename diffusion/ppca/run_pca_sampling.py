import numpy as np
from argparse import ArgumentParser
from numpy import random
from datasets import load_dataset
from matplotlib import pyplot as plt

from src.svd import naive_svd
from src.svd import golub_kahan_svd
from src.em import ppca, rotate_W_orthogonal
from src.sample_utils import get_samples_and_normalize, generate_sample_conditionned


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

        mu = np.mean(X.T, axis=1).reshape(-1, 1)

        W, s = ppca((X.T - mu).copy(), 2)

        W = rotate_W_orthogonal(W)

        r, c = label // 5, label % 5
        axs[r, c].axis("off")
        axs[r, c].imshow(sample.copy(), cmap="gray")
        axs_ppca[r, c].axis("off")
        axs_ppca[r, c].imshow(
            np.sum(mu + W, axis=1).reshape(28, 28).copy(), cmap="gray"
        )

    fig.suptitle("Generated samples with SVD algorithm")
    fig_ppca.suptitle("Generated samples with Proba PCA algorithm")
    plt.tight_layout()
    plt.show()

    fig.savefig("svd_mnist.png", bbox_inches="tight")
    fig_ppca.savefig("ppca_mnist.png", bbox_inches="tight")

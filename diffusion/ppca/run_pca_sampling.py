import numpy as np
from argparse import ArgumentParser
from datasets import load_dataset
from matplotlib import pyplot as plt

from src.em import ppca, rotate_W_orthogonal, compute_likelihood_pca
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
        print(f"\nDigit : {label}")
        print("Get samples subset...")
        X = get_samples_and_normalize(data, cl, n_samples, label)

        print("Generating new PCA samples from SVD...")
        sample_svd = generate_sample_conditionned(X.copy(), n_components)

        print("Generating samples with Probabilistic PCA...")
        sample_ppca = generate_sample_conditionned(
            X.copy(), n_components, probabilistic=True
        )

        r, c = label // 5, label % 5
        axs[r, c].axis("off")
        axs[r, c].imshow(sample_svd.copy(), cmap="gray")
        axs_ppca[r, c].axis("off")
        axs_ppca[r, c].imshow(sample_ppca.copy(), cmap="gray")

    fig.suptitle("Generated samples with SVD algorithm")
    fig_ppca.suptitle("Generated samples with Proba PCA algorithm")
    plt.tight_layout()
    fig.savefig("svd_mnist.png", bbox_inches="tight")
    fig_ppca.savefig("ppca_mnist.png", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

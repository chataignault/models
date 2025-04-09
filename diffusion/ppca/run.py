import numpy as np
from numpy import random
from datasets import load_dataset
from matplotlib import pyplot as plt

from src.svd import naive_svd


def generate_sample_naive_conditionned(X: np.array, keep: int):
    # can't have zero columns
    X += 0.5 * np.random.randn(*X.shape)

    U, S, V = naive_svd(X.copy())

    s = np.diag(S).copy()
    s[keep:] = 0.0

    s = np.diag(s)

    samples = U[:, :keep] @ s[:keep, :keep] @ V.T[:keep, :]

    sample = samples[0]
    sample = sample.reshape(28, 28)
    return sample


def get_samples_and_normalize(data:np.array, cl:np.array, n_samples:int, label:int):
    X = [pil for (pil, l) in zip(data, cl) if l == label]
    X = np.array([np.array(pil).reshape(784) for pil in X]) / 25.0
        
    idx = random.choice(np.arange(len(X)), n_samples, replace=False)
    X = X[idx]
    return X

if __name__ == "__main__":
    data_dict = load_dataset("mnist", num_proc=4)["train"]
    data, cl = data_dict["image"], data_dict["label"]

    fig, axs = plt.subplots(figsize=(10, 4), ncols=5, nrows=2)

    n_components = 3
    n_samples = 1000

    for label in range(10):
        
        X = get_samples_and_normalize(data, cl, n_samples, label)

        sample = generate_sample_naive_conditionned(X, n_components)

        r, c = label // 5, label % 5
        axs[r, c].axis("off")
        axs[r, c].imshow(sample.copy(), cmap="gray")

    plt.tight_layout()
    plt.show()
    fig.savefig("ppca_mnist.png", bbox_inches="tight")

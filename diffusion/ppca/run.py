import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

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


if __name__ == "__main__":
    data_dict = load_dataset("mnist", num_proc=4)["train"]
    data, cl = data_dict["image"], data_dict["label"]

    fig, axs = plt.subplots(figsize=(10, 4), ncols=5, nrows=2)

    keep = 5
    for label in range(10):
        X = [pil for (pil, l) in zip(data, cl) if l == label]
        X = np.array([np.array(pil).reshape(784) for pil in X]) / 25.0
        # need be square for now
        X = X[:784]

        sample = generate_sample_naive_conditionned(X, keep)

        r, c = label // 5, label % 5
        axs[r, c].axis("off")
        axs[r, c].imshow(sample.copy(), cmap="gray")

    plt.tight_layout()
    plt.show()
    fig.savefig("ppca_mnist.png", bbox_inches="tight")

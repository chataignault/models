import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from src.pure_qr import pure_QR


def naive_svd(A: np.ndarray):
    cov = A.T.dot(A)
    S, V = pure_QR(cov, maxit=10, tol=1e-4, trid=False, track=False, shift=False)
    s = np.sqrt(np.diag(S))
    s = np.clip(s, min=1e-5)
    U = np.divide(A @ V, s)

    return U, np.diag(s), V


if __name__ == "__main__":
    # n = 10
    # A = np.random.randn(n, n)
    # print(A)

    # S, V = pure_QR(
    #     A.T @ A,
    #     maxit=10000,
    #     tol=1e-4,
    #     trid=False,
    #     track=False,
    #     shift=False,
    #     )
    # print(np.linalg.norm(V.T @ V - np.eye(len(A))))
    # print(np.linalg.norm(A.T @ A - V @ S @ V.T))

    # U, S, V = naive_svd(A)

    # print(np.linalg.norm(A - U @ S @ V.T))
    # print(np.diag(S))

    data_dict = load_dataset("mnist", num_proc=4)["train"]
    data, c = data_dict["image"], data_dict["label"]

    label = 2

    data = [pil for (pil, l) in zip(data, c) if l == label]
    data = np.array([np.array(pil).reshape(784) for pil in data]) / 25.0

    # make it square for now
    data = data[:784]
    # imshow(data[0].reshape(28, 28))
    # plt.show()

    # can't have zero columns
    data += 0.5 * np.random.randn(*data.shape)

    U, S, V = naive_svd(data)

    print(U[:5, :5])

    keep = 5
    s = np.diag(S).copy()
    s[keep:] = 0.0

    s = np.diag(s)

    samples = U[:, :keep] @ s[:keep, :keep] @ V.T[:keep, :]

    sample = samples[0]
    sample = sample.reshape(28, 28)

    imshow(sample)

    plt.show()

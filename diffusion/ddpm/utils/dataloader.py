from datasets import load_dataset
from enum import Enum
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    RandomHorizontalFlip,
    Pad,
)
from torchvision.datasets import CelebA, CIFAR10
import torch
from torch.utils.data import DataLoader, Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        """
        Custom CIFAR10 dataset that returns only images with transforms applied

        Args:
            root (str): Root directory of dataset
            train (bool): If True, load training data, else load test data
            transform (callable, optional): Optional transform to be applied on images
            download (bool): If True, downloads the dataset if not already downloaded
        """
        self.cifar = CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, _ = self.cifar[idx]

        if self.transform:
            image = self.transform(image)

        return image


class CelebADataset(Dataset):
    def __init__(self, root, split="train", transform=None, download=True):
        self.celeba = CelebA(root=root, split=split, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        image, _ = self.celeba[idx]

        if self.transform:
            image = self.transform(image)

        return image


class XMNISTDataset(Dataset):
    def __init__(self, name: str, transform=None):
        """
        Custom xMNIST dataset that returns only images with transforms applied

        Args:
            name: name of the HuggingFace dataset
            transform (callable, optional): Optional transform to be applied on images
        """
        dsd = load_dataset(name.value, num_proc=4)
        self.data = dsd["train"]["image"] + dsd["test"]["image"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image


class Data(str, Enum):
    fashion_mnist = "fashion_mnist"
    mnist = "mnist"
    cifar_10 = "cifar_10"
    celeb_a = "celeb_a"


def get_dataloader(
    batch_size: int,
    device: str,
    dataset_name: Data,
    zero_pad_images: bool = False,
):
    if dataset_name == Data.celeb_a:
        transform = Compose(
            transforms=[
                ToTensor(),
                RandomHorizontalFlip(),
                Lambda(lambda t: (t * 2) - 1),
            ]
        )
        transformed_dataset = CelebADataset(
            root="data", split="train", transform=transform
        )
    if dataset_name == Data.cifar_10:
        transform = Compose(
            transforms=[
                ToTensor(),
                RandomHorizontalFlip(),
                Lambda(lambda t: (t * 2) - 1),
            ]
        )
        transformed_dataset = CIFAR10Dataset(
            root="data", train=True, transform=transform, download=True
        )
    else:
        t = [ToTensor()]
        if dataset_name != Data.mnist:
            t.append(RandomHorizontalFlip())
        if zero_pad_images:
            t.append(Pad(2))
        t.append(Lambda(lambda t: (t * 2) - 1))
        transform = Compose(t)
        transformed_dataset = XMNISTDataset(dataset_name, transform=transform)

    dataloader = DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator(device=device),
    )

    return dataloader

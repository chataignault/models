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

# Grain imports for TPU data pipeline
try:
    import grain.python as grain
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import jax.numpy as jnp
    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False


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
        # Concatenate train and test splits by converting columns to lists
        self.data = list(dsd["train"]["image"]) + list(dsd["test"]["image"])
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


def get_grain_dataloader(
    batch_size: int,
    dataset_name: Data,
    num_devices: int = 8,
    num_epochs: int = None,
    zero_pad_images: bool = False,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Create JAX grain-based dataloader optimized for TPU training.

    Args:
        batch_size: Global batch size (will be split across devices)
        dataset_name: Dataset enum (fashion_mnist, mnist, etc.)
        num_devices: Number of TPU cores for sharding
        num_epochs: Number of epochs to iterate (None = infinite)
        zero_pad_images: Whether to zero-pad images to 32x32
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling

    Returns:
        Iterator that yields batches of shape (num_devices, per_device_batch_size, H, W, C)
    """
    if not GRAIN_AVAILABLE:
        raise ImportError(
            "Grain is not installed. Install with: pip install grain tensorflow tensorflow-datasets"
        )

    assert batch_size % num_devices == 0, (
        f"Batch size {batch_size} must be divisible by num_devices {num_devices}"
    )
    per_device_batch_size = batch_size // num_devices

    # Load dataset via TFDS
    if dataset_name == Data.fashion_mnist:
        tfds_name = "fashion_mnist"
    elif dataset_name == Data.mnist:
        tfds_name = "mnist"
    elif dataset_name == Data.cifar_10:
        tfds_name = "cifar10"
    else:
        raise ValueError(f"Unsupported dataset for grain: {dataset_name}")

    # Load dataset
    ds_builder = tfds.builder(tfds_name)
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset(split="train")

    # Convert to grain data source
    data_source = grain.TensorflowDataSource(ds)

    # Define preprocessing operations
    def normalize_image(features):
        """Normalize image to [-1, 1] range."""
        image = features["image"]
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0  # Normalize to [-1, 1]
        return {"image": image}

    def random_flip(features):
        """Apply random horizontal flip."""
        image = features["image"]
        if dataset_name != Data.mnist:  # Don't flip MNIST
            image = tf.image.random_flip_left_right(image)
        return {"image": image}

    def maybe_pad(features):
        """Optionally pad images to 32x32."""
        image = features["image"]
        if zero_pad_images:
            # Pad to 32x32
            pad_width = [[2, 2], [2, 2], [0, 0]]
            image = tf.pad(image, pad_width, mode="CONSTANT", constant_values=0)
        return {"image": image}

    def extract_image(features):
        """Extract just the image tensor."""
        return features["image"]

    # Build transformation pipeline
    transformations = [
        grain.MapTransform(normalize_image),
        grain.MapTransform(random_flip),
    ]

    if zero_pad_images:
        transformations.append(grain.MapTransform(maybe_pad))

    transformations.append(grain.MapTransform(extract_image))

    # Batch per device
    transformations.append(grain.Batch(batch_size=per_device_batch_size))

    # Create sampler
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
    )

    # Create data loader
    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=4,  # Parallel data loading workers
        worker_buffer_size=2,
        read_options=grain.ReadOptions(
            num_threads=2,
            prefetch_buffer_size=4,  # Prefetch for TPU saturation
        ),
    )

    # Wrap to reshape batches for pmap
    def reshape_for_pmap(batch_iterator):
        """
        Reshape batches from (per_device_batch_size, H, W, C)
        to (num_devices, per_device_batch_size, H, W, C) for pmap.
        """
        for batch in batch_iterator:
            # Convert TensorFlow tensors to numpy/JAX arrays
            batch = tf.identity(batch).numpy()

            # Repeat batch across devices (each device gets same data initially)
            # In practice, you'd want different data per device via sharding
            # For now, we'll collect num_devices batches and stack them
            batches = [batch]
            for _ in range(num_devices - 1):
                try:
                    batches.append(next(batch_iterator).numpy())
                except StopIteration:
                    break

            if len(batches) == num_devices:
                # Stack along device dimension
                stacked = jnp.stack(batches, axis=0)
                yield stacked

    return reshape_for_pmap(iter(loader))

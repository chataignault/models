import jax
import jax.numpy as jnp
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Iterator


def jax_normalize(image):
    """Normalize image with CIFAR-10 stats using JAX"""
    mean = jnp.array([0.4914, 0.4822, 0.4465])
    std = jnp.array([0.2023, 0.1994, 0.2010])
    return (image - mean) / std


def jax_random_crop(rng, image, padding=4):
    """Random crop with padding using JAX"""
    # Pad the image
    padded = jnp.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    
    # Random crop coordinates - ensure we don't exceed bounds
    max_crop = 2 * padding
    rng, key = jax.random.split(rng)
    crop_y = jax.random.randint(key, (), 0, max_crop + 1)
    rng, key = jax.random.split(rng)
    crop_x = jax.random.randint(key, (), 0, max_crop + 1)
    
    # Crop - ensure the slice size doesn't exceed padded dimensions
    cropped = jax.lax.dynamic_slice(padded, (crop_y, crop_x, 0), (32, 32, 3))
    return rng, cropped


def jax_random_flip(rng, image):
    """Random horizontal flip using JAX"""
    rng, key = jax.random.split(rng)
    should_flip = jax.random.bernoulli(key, 0.5)
    return rng, jnp.where(should_flip, jnp.fliplr(image), image)


def augment_batch_jax(rng, images):
    """Apply data augmentation to a batch using JAX"""
    batch_size = images.shape[0]
    rng_keys = jax.random.split(rng, batch_size)
    
    def augment_single(rng_key, image):
        rng_key, image = jax_random_crop(rng_key, image, padding=4)
        rng_key, image = jax_random_flip(rng_key, image)
        return image
    
    augmented = jax.vmap(augment_single)(rng_keys, images)
    return augmented


class JAXDataLoader:
    def __init__(self, torch_loader, augment=False):
        self.torch_loader = torch_loader
        self.augment = augment
        self.rng = jax.random.PRNGKey(42)
    
    def __iter__(self):
        for batch_images, batch_labels in self.torch_loader:
            # Convert to numpy then JAX arrays and transpose from BCHW to BHWC
            images = jnp.array(batch_images.numpy()).transpose(0, 2, 3, 1)  # BCHW -> BHWC
            labels = jnp.array(batch_labels.numpy())
            
            # Images are already in [0, 1] range from ToTensor()
            
            # Apply augmentation if training
            if self.augment:
                self.rng, aug_rng = jax.random.split(self.rng)
                images = augment_batch_jax(aug_rng, images)
            
            # Normalize with CIFAR-10 stats
            images = jax.vmap(jax_normalize)(images)
            
            yield images, labels


def get_cifar10_dataloaders(batch_size: int = 64) -> Tuple[JAXDataLoader, JAXDataLoader]:
    # Use basic PyTorch transforms for initial loading
    transform_basic = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])
    
    # Load datasets using torchvision (minimal PyTorch usage)
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_basic)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_basic)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Wrap in JAX data loaders
    train_jax_loader = JAXDataLoader(trainloader, augment=True)
    test_jax_loader = JAXDataLoader(testloader, augment=False)
    
    return train_jax_loader, test_jax_loader


def get_classes():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

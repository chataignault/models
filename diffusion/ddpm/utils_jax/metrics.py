"""
Metrics module for evaluating diffusion model quality.

This module provides FID (Fréchet Inception Distance) computation functionality,
bridging JAX-based training with PyTorch-based evaluation metrics.
"""

import os
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from typing import Optional, Tuple
import logging

# Import JAX utilities
from jax import random
from utils_jax.diffusion import sample

logger = logging.getLogger(__name__)


def check_pytorch_fid_available() -> bool:
    """
    Check if pytorch-fid is installed.

    Returns:
        True if pytorch-fid is available, False otherwise.
    """
    try:
        import pytorch_fid
        import torch
        return True
    except ImportError:
        return False


def jax_to_fid_format(
    jax_images: jnp.ndarray,
    channels: int,
) -> np.ndarray:
    """
    Convert JAX format images to pytorch-fid compatible format.

    Args:
        jax_images: JAX array with shape (N, H, W, C) in range [-1, 1]
        channels: Number of channels (1 for grayscale, 3 for RGB)

    Returns:
        NumPy array with shape (N, C, H, W) in range [0, 255] as uint8

    Process:
        1. Convert JAX DeviceArray to NumPy
        2. Denormalize from [-1, 1] to [0, 1]
        3. Scale from [0, 1] to [0, 255]
        4. Convert to uint8
        5. Transpose from NHWC to NCHW format
    """
    # Step 1: Convert to NumPy
    images_np = np.array(jax_images)  # (N, H, W, C) in [-1, 1]

    # Step 2: Denormalize to [0, 1]
    images_01 = (images_np + 1.0) / 2.0
    images_01 = np.clip(images_01, 0.0, 1.0)

    # Step 3: Scale to [0, 255]
    images_255 = (images_01 * 255.0).astype(np.uint8)

    # Step 4: Transpose NHWC -> NCHW
    images_nchw = np.transpose(images_255, (0, 3, 1, 2))

    return images_nchw


def dataloader_to_fid_format(
    batch,
    channels: int,
    is_grain: bool = False,
) -> np.ndarray:
    """
    Convert dataloader batch to FID format.

    Args:
        batch: Batch from dataloader (PyTorch Tensor or JAX array)
        channels: Number of channels
        is_grain: Whether batch is from Grain dataloader (JAX format)

    Returns:
        NumPy array with shape (N, C, H, W) in range [0, 255] as uint8
    """
    if is_grain:
        # Grain returns JAX arrays in NHWC format
        # May be sharded: (num_devices, per_device_batch, H, W, C)
        # Flatten first two dimensions if needed
        if batch.ndim == 5:
            batch = batch.reshape(-1, *batch.shape[2:])
        return jax_to_fid_format(batch, channels)
    else:
        # PyTorch dataloader returns tensors in NCHW format, range [-1, 1]
        import torch

        # Convert to numpy
        if isinstance(batch, torch.Tensor):
            batch_np = batch.cpu().numpy()
        else:
            batch_np = np.array(batch)

        # Denormalize from [-1, 1] to [0, 1]
        batch_01 = (batch_np + 1.0) / 2.0
        batch_01 = np.clip(batch_01, 0.0, 1.0)

        # Scale to [0, 255] and convert to uint8
        batch_255 = (batch_01 * 255.0).astype(np.uint8)

        return batch_255


def generate_samples_for_fid(
    state,
    rng,
    num_samples: int,
    batch_size: int,
    num_devices: int,
    T: int,
    betas,
    alphas_cumprod,
    alphas_cumprod_prev,
    posterior_variance,
    channels: int,
    img_size: int,
) -> np.ndarray:
    """
    Generate samples in batches for FID computation.

    Args:
        state: Training state (possibly replicated across devices)
        rng: JAX random key
        num_samples: Total number of samples to generate (e.g., 5000)
        batch_size: Batch size per generation call (e.g., 50)
        num_devices: Number of devices (for unreplicating state)
        T: Number of diffusion timesteps
        betas: Beta schedule
        alphas_cumprod: Cumulative product of alphas
        alphas_cumprod_prev: Shifted cumulative product
        posterior_variance: Posterior variance schedule
        channels: Number of image channels
        img_size: Image size (height/width)

    Returns:
        Generated samples as NumPy array (N, C, H, W) in [0, 255] uint8

    Process:
        1. Unreplicate state to first device if distributed
        2. Generate samples in batches to avoid OOM
        3. Convert each batch from JAX NHWC [-1,1] to NumPy NCHW [0,255] uint8
        4. Concatenate all batches
    """
    import jax

    # Unreplicate state to first device only (avoid redundant computation)
    if num_devices > 1:
        single_device_state = jax.tree.map(lambda x: x[0], state)
    else:
        single_device_state = state

    # Calculate number of batches needed
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_samples = []

    logger.info(f"Generating {num_samples} samples in {num_batches} batches of {batch_size}")

    for i in tqdm(range(num_batches), desc="Generating FID samples"):
        # Determine batch size for this iteration (handle remainder)
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Split RNG for this batch
        rng, batch_rng = random.split(rng)

        # Generate batch using diffusion sampling
        sample_shape = (current_batch_size, img_size, img_size, channels)
        batch_samples = sample(
            single_device_state,
            sample_shape,
            batch_rng,
            T,
            betas,
            alphas_cumprod,
            alphas_cumprod_prev,
            posterior_variance,
            pseudo_video=False,
        )[-1]  # Get final denoised images

        # Convert to FID format: JAX NHWC [-1,1] -> NumPy NCHW [0,255] uint8
        batch_fid = jax_to_fid_format(batch_samples, channels)
        all_samples.append(batch_fid)

    # Concatenate all batches
    samples_array = np.concatenate(all_samples, axis=0)[:num_samples]

    logger.info(f"Generated {samples_array.shape[0]} samples with shape {samples_array.shape}")

    return samples_array


def precompute_fid_statistics(
    dataloader,
    dataset_name: str,
    num_samples: int = 50000,
    batch_size: int = 64,
    device: str = "cuda",
    stats_path: Optional[str] = None,
    channels: int = 3,
    is_grain: bool = False,
) -> str:
    """
    Precompute InceptionV3 statistics for real training data.

    Args:
        dataloader: PyTorch or Grain DataLoader for real data
        dataset_name: Name of dataset (for cache file naming)
        num_samples: Number of real samples to use for statistics
        batch_size: Batch size for InceptionV3 processing
        device: Device to run InceptionV3 on ('cuda' or 'cpu')
        stats_path: Path to save statistics (auto-generated if None)
        channels: Number of image channels
        is_grain: Whether using Grain dataloader

    Returns:
        Path to saved statistics file (.npz)

    Process:
        1. Collect num_samples from dataloader
        2. Convert to pytorch-fid format (NCHW, [0,255] uint8)
        3. Run through InceptionV3 to extract features
        4. Compute mean (mu) and covariance (sigma)
        5. Save to .npz file
    """
    import torch
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_activation_statistics

    # Determine stats path if not provided
    if stats_path is None:
        stats_dir = "fid_stats"
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"{dataset_name}_stats.npz")

    # Check if already exists
    if os.path.exists(stats_path):
        logger.info(f"Statistics file already exists at {stats_path}")
        return stats_path

    logger.info(f"Precomputing FID statistics for {dataset_name}")
    logger.info(f"Collecting {num_samples} real samples...")

    # Collect real samples from dataloader
    real_samples = []
    samples_collected = 0

    for batch in tqdm(dataloader, desc="Collecting real samples"):
        # Convert batch to FID format
        batch_fid = dataloader_to_fid_format(batch, channels, is_grain)
        real_samples.append(batch_fid)

        samples_collected += batch_fid.shape[0]
        if samples_collected >= num_samples:
            break

    # Concatenate and trim to exact num_samples
    real_samples = np.concatenate(real_samples, axis=0)[:num_samples]
    logger.info(f"Collected {real_samples.shape[0]} samples with shape {real_samples.shape}")

    # Setup InceptionV3 model
    logger.info("Loading InceptionV3 model...")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # Convert to torch tensor
    logger.info("Computing statistics with InceptionV3...")
    real_torch = torch.from_numpy(real_samples).float()

    # Compute statistics
    with torch.no_grad():
        mu, sigma = calculate_activation_statistics(
            real_torch, model, batch_size=batch_size, dims=2048, device=device
        )

    # Save statistics
    logger.info(f"Saving statistics to {stats_path}")
    np.savez(stats_path, mu=mu, sigma=sigma)

    logger.info(f"FID statistics saved successfully")

    return stats_path


def compute_fid_score(
    generated_samples: np.ndarray,
    real_stats_path: str,
    device: str = "cuda",
    batch_size: int = 50,
) -> float:
    """
    Compute FID score between generated samples and precomputed real statistics.

    Args:
        generated_samples: Generated images (N, C, H, W) in [0, 255] uint8
        real_stats_path: Path to precomputed .npz with real data statistics
        device: Device for InceptionV3 ('cuda' or 'cpu')
        batch_size: Batch size for processing generated samples

    Returns:
        FID score (float, lower is better)

    Process:
        1. Load real statistics (mu1, sigma1) from .npz
        2. Extract features from generated samples using InceptionV3
        3. Compute generated statistics (mu2, sigma2)
        4. Calculate FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    import torch
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance

    logger.info("Computing FID score...")

    # Load real statistics
    logger.info(f"Loading real statistics from {real_stats_path}")
    stats = np.load(real_stats_path)
    mu1, sigma1 = stats['mu'], stats['sigma']

    # Setup InceptionV3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # Convert generated samples to torch tensor
    logger.info(f"Processing {generated_samples.shape[0]} generated samples...")
    gen_torch = torch.from_numpy(generated_samples).float()

    # Compute generated statistics
    with torch.no_grad():
        mu2, sigma2 = calculate_activation_statistics(
            gen_torch, model, batch_size=batch_size, dims=2048, device=device
        )

    # Calculate FID
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    logger.info(f"FID score: {fid_value:.4f}")

    return float(fid_value)


class FIDComputationError(Exception):
    """Custom exception for FID computation failures."""
    pass

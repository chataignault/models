"""TensorBoard logger for DDPM training."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class DDPMTensorBoardLogger:
    """TensorBoard logger for DDPM training metrics and samples."""

    def __init__(self, log_dir: str, flush_secs: int = 120):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            flush_secs: Flush to disk every N seconds
        """
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.step = 0
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"To view: tensorboard --logdir={log_dir}")

    def log_scalar(self, tag: str, value: Any, step: Optional[int] = None):
        """
        Log scalar value (loss, learning rate, etc.).

        Args:
            tag: Metric name (e.g., 'train/loss')
            value: Scalar value (can be JAX array, numpy, or Python float)
            step: Step number (uses internal counter if None)
        """
        step = step if step is not None else self.step

        # Convert JAX/numpy arrays to Python float
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            value = float(value)

        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalars: Dict[str, Any], step: Optional[int] = None):
        """
        Log multiple scalars at once.

        Args:
            scalars: Dictionary of {tag: value} pairs
            step: Step number (uses internal counter if None)
        """
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)

    def log_image(
        self,
        tag: str,
        image: Any,
        step: Optional[int] = None,
        dataformats: str = 'HWC'
    ):
        """
        Log single image.

        Args:
            tag: Image tag
            image: Image array (H, W, C) or (N, H, W, C)
            step: Step number
            dataformats: 'HWC' or 'CHW'
        """
        step = step if step is not None else self.step

        # Convert to numpy
        if isinstance(image, jnp.ndarray):
            image = np.array(image)

        # Denormalize from [-1, 1] to [0, 1]
        image = (image + 1.0) / 2.0
        image = np.clip(image, 0.0, 1.0)

        self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_images_grid(
        self,
        tag: str,
        images: Any,
        step: Optional[int] = None,
        nrow: int = 4,
    ):
        """
        Log grid of images.

        Args:
            tag: Grid tag
            images: Batch of images (N, H, W, C) in NHWC format
            step: Step number
            nrow: Number of images per row in grid
        """
        step = step if step is not None else self.step

        # Convert to numpy
        if isinstance(images, jnp.ndarray):
            images = np.array(images)

        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        images = np.clip(images, 0.0, 1.0)

        # Convert NHWC to NCHW for torchvision
        images = np.transpose(images, (0, 3, 1, 2))

        # Create grid using torchvision
        import torch
        import torchvision

        grid = torchvision.utils.make_grid(
            torch.from_numpy(images),
            nrow=nrow,
            padding=2,
            normalize=False,  # Already normalized
        )

        self.writer.add_image(tag, grid, step)

    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: Optional[int] = None,
    ):
        """
        Log histogram (for weights, gradients, etc.).

        Args:
            tag: Histogram tag
            values: Array of values
            step: Step number
        """
        step = step if step is not None else self.step

        # Convert to numpy
        if isinstance(values, jnp.ndarray):
            values = np.array(values)

        self.writer.add_histogram(tag, values, step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ):
        """
        Log text (for hyperparameters, notes, etc.).

        Args:
            tag: Text tag
            text: Text content
            step: Step number
        """
        step = step if step is not None else self.step
        self.writer.add_text(tag, text, step)

    def log_hparams(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log hyperparameters.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional dictionary of metrics for this run
        """
        if metrics is None:
            metrics = {}

        self.writer.add_hparams(hparams, metrics)

    def increment_step(self):
        """Increment global step counter."""
        self.step += 1

    def flush(self):
        """Manually flush events to disk."""
        self.writer.flush()

    def close(self):
        """Close writer and flush all events."""
        self.writer.flush()
        self.writer.close()
        print("TensorBoard logger closed")

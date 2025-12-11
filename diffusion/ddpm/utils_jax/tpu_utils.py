"""TPU utilities for distributed training with JAX."""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List
from jax.experimental import multihost_utils


def detect_tpu_environment() -> Dict[str, Any]:
    """
    Detect TPU cores and return device configuration.

    Returns:
        Dictionary with:
            - device_count: Number of available devices
            - device_type: Type of device ('tpu-v2', 'tpu-v3', 'tpu-v4', 'gpu', 'cpu')
            - devices: List of JAX devices
            - is_tpu: Boolean indicating if running on TPU
    """
    devices = jax.devices()
    device_count = len(devices)

    # Determine device type
    if devices[0].platform == 'tpu':
        # Try to infer TPU version from device description
        device_desc = str(devices[0])
        if 'v4' in device_desc.lower():
            device_type = 'tpu-v4'
        elif 'v3' in device_desc.lower():
            device_type = 'tpu-v3'
        elif 'v2' in device_desc.lower():
            device_type = 'tpu-v2'
        else:
            device_type = 'tpu'
        is_tpu = True
    elif devices[0].platform == 'gpu':
        device_type = 'gpu'
        is_tpu = False
    else:
        device_type = 'cpu'
        is_tpu = False

    return {
        'device_count': device_count,
        'device_type': device_type,
        'devices': devices,
        'is_tpu': is_tpu,
    }


def replicate_tree(tree: Any, num_devices: int) -> Any:
    """
    Replicate a PyTree across devices for pmap initialization.

    Args:
        tree: PyTree to replicate (e.g., TrainState)
        num_devices: Number of devices to replicate across

    Returns:
        Replicated PyTree where each leaf is stacked num_devices times
        along the first axis
    """
    return jax.tree.map(lambda x: jnp.stack([x] * num_devices), tree)


def unreplicate_first(tree: Any) -> Any:
    """
    Extract first replica from replicated PyTree.

    Used for checkpointing to avoid saving multiple copies of the same state.

    Args:
        tree: Replicated PyTree where first axis is the device axis

    Returns:
        Unreplicated PyTree (first replica only)
    """
    return jax.tree.map(lambda x: x[0], tree)


def split_rng_for_devices(rng: jax.random.PRNGKey, num_devices: int) -> jax.random.PRNGKey:
    """
    Split RNG key into per-device keys for deterministic multi-device training.

    Args:
        rng: Single PRNG key
        num_devices: Number of devices

    Returns:
        Array of PRNG keys with shape (num_devices, 2)
    """
    return jax.random.split(rng, num_devices)


def sync_devices():
    """
    Synchronize all devices (useful for timing and debugging).

    Blocks until all async operations on all devices complete.
    """
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


def create_device_mesh(num_devices: int) -> List:
    """
    Create device mesh for data parallelism.

    For single-host TPU (8 cores), creates a 1D mesh.

    Args:
        num_devices: Number of devices in mesh

    Returns:
        List of devices in mesh order
    """
    devices = jax.devices()[:num_devices]
    return devices


def assert_replicated_shape(tree: Any, num_devices: int):
    """
    Assert that PyTree has proper replicated shape (first axis = num_devices).

    Useful for debugging distributed training setup.

    Args:
        tree: PyTree to check
        num_devices: Expected number of devices

    Raises:
        AssertionError if shape doesn't match
    """
    def check_shape(x):
        if hasattr(x, 'shape'):
            assert x.shape[0] == num_devices, (
                f"Expected first axis to be {num_devices}, got {x.shape[0]}"
            )

    jax.tree.map(check_shape, tree)

import jax
from jax import numpy as jnp
from jax import random
import functools
import optax
from flax.training import train_state
from typing import Any
from dataclasses import dataclass

from .diffusion import forward_diffusion_sample, get_index_from_list


@dataclass
class SchedulerConfig:
    warmup_epochs: int
    num_epochs: int


def create_learning_rate_fn(config, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=base_learning_rate / 10.0,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(rng, model, learning_rate_fn, train: bool, num_devices: int = 1):
    """
    Create training state, optionally replicated across devices.

    Args:
        rng: Random number generator key
        model: Flax model to initialize
        learning_rate_fn: Learning rate schedule function
        train: Whether in training mode (affects BatchNorm)
        num_devices: Number of devices to replicate across (1 = single device)

    Returns:
        TrainState (replicated if num_devices > 1)
    """
    rng, rng_tab = random.split(rng)
    rng, rng_init = random.split(rng)
    variables = model.init(
        rng_init, jnp.ones([1, 28, 28, model.channels]), jnp.ones([1]), train=train
    )
    params = variables["params"]
    # Handle models without batch_stats (e.g., models using GroupNorm instead of BatchNorm)
    batch_stats = variables.get("batch_stats", {})
    tx = optax.adam(learning_rate_fn)
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
    )

    # Replicate across devices if using distributed training
    if num_devices > 1:
        from utils_jax.tpu_utils import replicate_tree

        state = replicate_tree(state, num_devices)

    return state


def linear_beta_schedule(
    timesteps: int, start: float = 0.0001, end: float = 0.02
) -> jnp.ndarray:
    """
    output a vector of size timesteps that is equally spaced between start and end; this will be the noise that is added in each time step.
    """
    return jnp.linspace(start=start, stop=end, num=timesteps)


@functools.partial(jax.jit, static_argnums=4)
def train_step(state, batch, diff_params, rng, learning_rate_function):
    t = diff_params["t"]
    sqrt_alphas_cumprod = diff_params["sqrt_alphas_cumprod"]
    sqrt_one_minus_alphas_cumprod = diff_params["sqrt_one_minus_alphas_cumprod"]

    def loss_fn(
        params,
    ):
        """
        Define the right loss given the model, the true x_0 and the time t
        """
        x_t, eps = forward_diffusion_sample(
            batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, rng
        )
        # Handle models with or without batch_stats (GroupNorm vs BatchNorm)
        if state.batch_stats:
            predicted_eps, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                x_t,
                t,
                train=True,
                mutable=["batch_stats"],
            )
        else:
            predicted_eps = state.apply_fn(
                {"params": params},
                x_t,
                t,
                train=True,
            )
            updates = {}
        return jnp.mean((eps - predicted_eps) ** 2), updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    lr = learning_rate_function(state.step)
    # Only update batch_stats if they exist
    if updates and "batch_stats" in updates:
        state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss, lr


@functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4,))
def train_step_pmap(state, batch, diff_params, rng, learning_rate_function):
    """
    Distributed training step across TPU cores using pmap.

    Args:
        state: Replicated TrainState (per-device copy)
        batch: Per-device batch (batch_size//num_devices, H, W, C)
        diff_params: Diffusion parameters (replicated across devices)
        rng: Per-device RNG key
        learning_rate_function: Learning rate schedule (static, not replicated)

    Returns:
        state: Updated per-device state
        loss: Per-device loss (aggregated across devices)
        lr: Learning rate (same on all devices)
    """
    t = diff_params["t"]
    sqrt_alphas_cumprod = diff_params["sqrt_alphas_cumprod"]
    sqrt_one_minus_alphas_cumprod = diff_params["sqrt_one_minus_alphas_cumprod"]

    def loss_fn(params):
        """Compute loss and batch norm updates."""
        x_t, eps = forward_diffusion_sample(
            batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, rng
        )

        # Handle models with or without batch_stats (GroupNorm vs BatchNorm)
        if state.batch_stats:
            predicted_eps, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                x_t,
                t,
                train=True,
                mutable=["batch_stats"],
            )
        else:
            predicted_eps = state.apply_fn(
                {"params": params},
                x_t,
                t,
                train=True,
            )
            updates = {}

        return jnp.mean((eps - predicted_eps) ** 2), updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)

    # CRITICAL: Aggregate gradients across devices using pmean
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    state = state.apply_gradients(grads=grads)
    lr = learning_rate_function(state.step)
    # Only update batch_stats if they exist
    if updates and "batch_stats" in updates:
        state = state.replace(batch_stats=updates["batch_stats"])

    return state, loss, lr


@functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(4,))
def train_step_pmap_bf16(state, batch, diff_params, rng, learning_rate_function):
    """
    Distributed training step with bfloat16 mixed precision.

    Uses bfloat16 for forward pass computations and float32 for gradients/parameters.
    This provides ~1.5-2x speedup on TPU with minimal accuracy impact.

    Args:
        state: Replicated TrainState (params in float32)
        batch: Per-device batch (batch_size//num_devices, H, W, C) in float32
        diff_params: Diffusion parameters (will be cast to bfloat16)
        rng: Per-device RNG key
        learning_rate_function: Learning rate schedule (static)

    Returns:
        state: Updated per-device state
        loss: Per-device loss in float32 (aggregated)
        lr: Learning rate
    """
    t = diff_params["t"]

    # Cast diffusion parameters to bfloat16 for computation
    sqrt_alphas_cumprod = diff_params["sqrt_alphas_cumprod"].astype(jnp.bfloat16)
    sqrt_one_minus_alphas_cumprod = diff_params["sqrt_one_minus_alphas_cumprod"].astype(
        jnp.bfloat16
    )

    # Cast batch to bfloat16
    batch_bf16 = batch.astype(jnp.bfloat16)

    def loss_fn(params):
        """Compute loss with bfloat16 forward pass."""
        # Forward diffusion in bfloat16
        x_t, eps = forward_diffusion_sample(
            batch_bf16, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, rng
        )

        # Handle models with or without batch_stats (GroupNorm vs BatchNorm)
        if state.batch_stats:
            # Model forward pass (computations in bfloat16)
            predicted_eps, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                x_t,
                t,
                train=True,
                mutable=["batch_stats"],
            )
        else:
            predicted_eps = state.apply_fn(
                {"params": params},
                x_t,
                t,
                train=True,
            )
            updates = {}

        # Cast to float32 for loss calculation (numerical stability)
        eps_f32 = eps.astype(jnp.float32)
        predicted_eps_f32 = predicted_eps.astype(jnp.float32)
        loss = jnp.mean((eps_f32 - predicted_eps_f32) ** 2)

        return loss, updates

    # Gradients computed in float32 (params are float32)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)

    # Aggregate across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    # Update (gradients and params in float32)
    state = state.apply_gradients(grads=grads)
    lr = learning_rate_function(state.step)
    # Only update batch_stats if they exist
    if updates and "batch_stats" in updates:
        state = state.replace(batch_stats=updates["batch_stats"])

    return state, loss, lr

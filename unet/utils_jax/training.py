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
        init_value=0.0,
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


def create_train_state(rng, model, learning_rate_fn, train: bool):
    rng, rng_tab = random.split(rng)
    print(
        model.tabulate(
            rng_tab,
            jnp.ones((1, 28, 28, 1)),
            jnp.ones([1]),
            train,
            compute_flops=True,
            compute_vjp_flops=True,
        )
    )
    rng, rng_init = random.split(rng)
    variables = model.init(rng_init, jnp.ones([1, 28, 28, 1]), jnp.ones([1]), train=train)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    tx = optax.adam(learning_rate_fn)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)


def linear_beta_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02) -> jnp.ndarray:
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
        x_t = forward_diffusion_sample(
            batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, rng
        )
        eps = (
            x_t - get_index_from_list(sqrt_alphas_cumprod, t, batch.shape) * jnp.array(batch)
        ) / get_index_from_list(sqrt_one_minus_alphas_cumprod, t, batch.shape)
        predicted_eps, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            x_t,
            t,
            train=True,
            mutable=["batch_stats"],
        )
        return jnp.mean((eps - predicted_eps) ** 2), updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    lr = learning_rate_function(state.step)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss, lr

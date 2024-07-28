import jax
from jax import numpy as jnp
from jax import random
from jax import jit
import optax
from flax.training import train_state
from tqdm import tqdm


def create_train_state(rng, learning_rate, model):
    rng, rng_tab = random.split(rng)
    # print(
    #     model.tabulate(
    #         rng_tab, jnp.ones((1, 28, 28, 1)), jnp.ones([1]), compute_flops=True, compute_vjp_flops=True
    #     )
    # )
    rng, rng_init = random.split(rng)
    params = model.init(rng_init, jnp.ones([1, 28, 28, 1]), jnp.ones([1]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jit
def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = jnp.take_along_axis(vals, t, axis=-1)  # [:, None]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(
    timesteps: int, start: float = 0.0001, end: float = 0.02
) -> jnp.ndarray:
    """
    output a vector of size timesteps that is equally spaced between start and end; this will be the noise that is added in each time step.
    """
    return jnp.linspace(start=start, stop=end, num=timesteps)


def forward_diffusion_sample(
    x_0: jnp.ndarray,
    t: jnp.ndarray,
    sqrt_alphas_cumprod: jnp.ndarray,
    sqrt_one_minus_alphas_cumprod: jnp.ndarray,
    rng,
) -> jnp.ndarray:
    """
    Takes an image and a timestep as input and
    returns the noisy version of it.
    """
    mean = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) * jnp.array(x_0)
    std = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return mean + std * random.normal(rng, x_0.shape)


@jax.jit
def train_step(state, batch, diff_params, rng):
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
            x_t
            - get_index_from_list(sqrt_alphas_cumprod, t, batch.shape)
            * jnp.array(batch)
        ) / get_index_from_list(sqrt_one_minus_alphas_cumprod, t, batch.shape)
        predicted_eps = state.apply_fn({"params": params}, x_t, t)
        return jnp.mean((eps - predicted_eps) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def sample_timestep(
    state,
    x: jnp.ndarray,
    t: jnp.ndarray,
    i: int,
    posterior_variance: jnp.ndarray,
    sqrt_one_minus_alphas_cumprod: jnp.ndarray,
    sqrt_recip_alphas: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    Note that it also needs additional arguments about the posterior_variance, sqrt_minus_alphas_cumprod and sqrt_recip_alphas.
    """
    # Compute the denoised image
    pred_noise = state.apply_fn({"params": state.params}, x, t)
    x = sqrt_recip_alphas[i] * (
        x - (posterior_variance[i]) / sqrt_one_minus_alphas_cumprod[i] * pred_noise
    )

    # Apply noise if we are not in the last step
    if i > 0:
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, shape=x.shape)
        x += jnp.sqrt(posterior_variance[i]) * z

    return x


def sample(
    state,
    shape,
    rng,
    T,
    posterior_variance,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
):
    b = shape[0]
    rng, rng_samp = random.split(rng)
    img = random.normal(rng_samp, shape)
    imgs = []
    for i in tqdm(
        reversed(range(1, T)), desc="sampling loop time step", total=T
    ):  # range started at 0
        t = jnp.ones((b,), dtype=jnp.float32) * i
        img = sample_timestep(
            state,
            img,
            t,
            i,
            posterior_variance,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            rng,
        )
        imgs.append(img)
    return imgs

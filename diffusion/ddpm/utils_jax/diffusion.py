from jax import numpy as jnp
from jax import random
import jax
from tqdm import tqdm


@jax.jit
def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = jnp.take_along_axis(vals, t, axis=-1)  # [:, None]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


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
    pred_noise = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, x, t, train=False
    )
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

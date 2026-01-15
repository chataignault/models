from tqdm import tqdm
import jax
from jax import random
from jax import numpy as jnp


@jax.jit
def get_index_from_list(vals: jax.Array, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = jnp.take_along_axis(vals, t, axis=-1)  # [:, None]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# @jax.jit
def linear_beta_schedule(
    timesteps: int, start: float = 1e-4, end: float = 2e-2
) -> jnp.array:
    return jnp.linspace(start=start, stop=end, num=timesteps)


# @jax.jit
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> jnp.array:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = jnp.arange(timesteps + 1).astype(jnp.float32) / timesteps
    alphas_cumprod = jnp.cos(((steps + s) / (1 + s)) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = jnp.clip(betas, 0, 0.999)
    return betas


@jax.jit
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
    eps = random.normal(rng, x_0.shape)
    mean = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) * jnp.array(x_0)
    std = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return mean + std * eps, eps


@jax.jit
def q_posterior(
    x_start,
    x_t,
    t,
    posterior_mean_coef1,
    posterior_mean_coef2,
    posterior_variance,
    posterior_log_variance_clipped,
):
    posterior_mean = (
        get_index_from_list(posterior_mean_coef1, t, x_t.shape) * x_start
        + get_index_from_list(posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = get_index_from_list(posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = get_index_from_list(
        posterior_log_variance_clipped, t, x_t.shape
    )

    return posterior_mean, posterior_variance, posterior_log_variance_clipped


@jax.jit
def sample_timestep(
    state,
    x: jnp.ndarray,
    t: jnp.ndarray,
    i: int,
    posterior_variance: jnp.ndarray,
    sqrt_recip_alphas_cumprod: jnp.ndarray,
    sqrt_recipm1_alphas_cumprod: jnp.ndarray,
    posterior_mean_coef1: jnp.ndarray,
    posterior_mean_coef2: jnp.ndarray,
    posterior_log_variance_clipped: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    # Compute the predicted noise
    # Handle models with or without batch_stats (GroupNorm vs BatchNorm)
    if state.batch_stats:
        eps = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            x,
            t,
            train=False,
        )
    else:
        eps = state.apply_fn({"params": state.params}, x, t, train=False)
    t_int = t.astype(dtype=jnp.int32)
    x_start = (
        get_index_from_list(sqrt_recip_alphas_cumprod, t_int, x.shape) * x
        - get_index_from_list(sqrt_recipm1_alphas_cumprod, t_int, x.shape) * eps
    )

    x_start = jnp.clip(x_start, -1.0, 1.0)

    mu_prev, posterior_variance, posterior_log_variance = q_posterior(
        x_start=x_start,
        x_t=x,
        t=t_int,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
    )

    # Apply noise if we are not in the last step
    z = jax.random.normal(rng, shape=x.shape)
    mu_prev = jax.lax.cond(
        i > 0,
        lambda _: mu_prev + jnp.exp(0.5 * posterior_log_variance) * z,
        lambda _: mu_prev,
        operand=None,
    )

    return mu_prev


def sample(
    state,
    shape,
    rng,
    T,
    betas: jnp.ndarray,
    alphas_cumprod: jnp.ndarray,
    alphas_cumprod_prev: jnp.ndarray,
    posterior_variance: jnp.ndarray,
    pseudo_video: bool = False,
):
    b = shape[0]
    rng, rng_samp = random.split(rng)
    img = random.normal(rng_samp, shape)
    imgs = []

    posterior_mean_coef1 = (
        betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * jnp.sqrt(1.0 - betas) / (1.0 - alphas_cumprod)
    )
    sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1)
    posterior_log_variance_clipped = jnp.log(jnp.clip(posterior_variance, min=1e-20))

    for i in tqdm(reversed(range(T)), desc="sampling loop time step", total=T):
        t = jnp.ones((b,), dtype=jnp.float32) * i
        rng, step_rng = random.split(rng)
        img = sample_timestep(
            state,
            img,
            t,
            i,
            posterior_variance,
            posterior_mean_coef1=posterior_mean_coef1,
            posterior_mean_coef2=posterior_mean_coef2,
            posterior_log_variance_clipped=posterior_log_variance_clipped,
            sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
            rng=step_rng,
        )
        if pseudo_video:
            # imgs.append(unnormalize_to_zero_to_one(img))
            imgs.append(img)
    # imgs.append(unnormalize_to_zero_to_one(img))
    imgs.append(img)
    return imgs


def unnormalize_to_zero_to_one(t: jnp.ndarray):
    return (t + 1) * 0.5

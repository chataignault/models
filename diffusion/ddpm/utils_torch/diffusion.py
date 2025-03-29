import numpy as np
from tqdm import tqdm
from typing import Tuple, List
import torch
from torch import Tensor
from torch.nn import Module


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(
    timesteps: int, device: str, start: float = 0.0001, end: float = 0.02
) -> Tensor:
    """
    output a vector of size timesteps that is equally spaced between start and end; this will be the noise that is added in each time step.
    """
    return torch.linspace(
        start=start, end=end, steps=timesteps, requires_grad=False, device=device
    )


def cosine_beta_schedule(
    timesteps: int, device: str, start: float = 0.0001, end: float = 0.02
) -> Tensor:
    """
    output a vector of size timesteps that is equally spaced between start and end; this will be the noise that is added in each time step.
    """
    steps = (
        torch.arange(start=0, end=timesteps, requires_grad=False, device=device).to(
            torch.float32
        )
        + start
    )
    return end * (1.0 - (torch.cos(np.pi * 0.5 * (steps / timesteps))) ** 2)


def sigmoid_beta_schedule(timesteps, device, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = (
        torch.linspace(
            0,
            timesteps,
            steps,
            # dtype = torch.float64
        )
        / timesteps
    )
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).to(device)


def forward_diffusion_sample(
    x_0: Tensor,
    t: Tensor,
    sqrt_alphas_cumprod: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,
    noise: Tensor,
    device: str = "cpu",
) -> Tensor:
    """
    Takes an image and a timestep as input and
    returns the noisy version of it at a given timestep, using the Gaussian noise reparameterisation
    """
    mean = (get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) * x_0).to(device)
    std = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape).to(device)
    return mean + std * noise


# @torch.no_grad()
@torch.inference_mode()
def sample_timestep(
    model,
    x: Tensor,
    t: Tensor,
    i: int,
    betas: Tensor,
    posterior_variance: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,
    sqrt_recip_alphas: Tensor,
    device: str,
) -> Tensor:
    """
    Calls the model to predict the noise in the image and returns the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    mu_prev = sqrt_recip_alphas[i] * (
        x - betas[i] / sqrt_one_minus_alphas_cumprod[i] * model(x, t)
    )
    # x = x.clamp(-1. 1.)
    if i > 0:
        z = torch.randn_like(x).to(device)
        mu_prev += torch.sqrt(posterior_variance[i]) * z

    return mu_prev


# @torch.no_grad()
@torch.inference_mode()
def sample(
    model: Module,
    shape: Tuple,
    betas,
    posterior_variance: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,
    sqrt_recip_alphas: Tensor,
    T: int,
    pseudo_video: bool = False,
) -> List[Tensor]:
    device = next(model.parameters()).device
    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(
        shape,
        device=device,
    )
    imgs = []
    for i in tqdm(
        reversed(range(1, T)), desc="sampling loop time step", total=T
    ):  # range started at 0
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = sample_timestep(
            model,
            img,
            t,
            i,
            betas,
            posterior_variance,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            device,
        )
        # re-normalise
        # img = (img + 1.0) / 2.0
        if pseudo_video:
            imgs.append(img)
    imgs.append(img)
    return imgs


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


def p_mean_variance(
    model,
    x,
    t,
    clip_denoised=True,
    posterior_mean_coef1=None,
    posterior_mean_coef2=None,
    sqrt_recip_alphas_cumprod=None,
    sqrt_recipm1_alphas_cumprod=None,
    posterior_variance=None,
    posterior_log_variance_clipped=None,
):
    eps = model(x, t)
    x_start = (
        get_index_from_list(sqrt_recip_alphas_cumprod, t, x.shape) * x
        - get_index_from_list(sqrt_recipm1_alphas_cumprod, t, x.shape) * eps
    )
    if clip_denoised:
        x_start.clamp_(-1.0, 1.0)

    model_mean, posterior_variance, posterior_log_variance = q_posterior(
        x_start=x_start,
        x_t=x,
        t=t,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
    )
    return model_mean, posterior_variance, posterior_log_variance, x_start


@torch.inference_mode()
def p_sample(
    model,
    x,
    t: int,
    sqrt_recip_alphas_cumprod,
    sqrt_recipm1_alphas_cumprod,
    posterior_mean_coef1,
    posterior_mean_coef2,
    posterior_variance,
    posterior_log_variance_clipped,
    device="cuda",
):
    b = x.shape[0]
    batched_times = torch.full((b,), t, device=device, dtype=torch.long)
    model_mean, _, model_log_variance, x_start = p_mean_variance(
        model=model,
        x=x,
        t=batched_times,
        clip_denoised=True,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
    )
    noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    return pred_img, x_start


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


@torch.inference_mode()
def p_sample_loop(
    model,
    shape,
    num_timesteps,
    posterior_mean_coef1,
    posterior_mean_coef2,
    sqrt_recip_alphas_cumprod,
    sqrt_recipm1_alphas_cumprod,
    posterior_variance,
    posterior_log_variance_clipped,
    device="cuda",
):
    batch = shape[0]

    img = torch.randn(shape, device=device)

    x_start = None

    for t in tqdm(
        reversed(range(0, num_timesteps)),
        desc="sampling loop time step",
        total=num_timesteps,
    ):
        img, x_start = p_sample(
            model,
            img,
            t,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_mean_coef1,
            posterior_mean_coef2,
            posterior_variance,
            posterior_log_variance_clipped,
        )

    ret = img
    ret = unnormalize_to_zero_to_one(ret)
    return ret

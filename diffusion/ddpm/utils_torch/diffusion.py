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


def forward_diffusion_sample(
    x_0: Tensor,
    t: Tensor,
    sqrt_alphas_cumprod: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,
    device: str = "cpu",
) -> Tensor:
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    Hint: use the get_index_from_list function to select the right values at time t.
    """
    mean = (get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) * x_0).to(device)
    std = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape).to(device)
    return mean + std * torch.randn_like(x_0).to(device)


@torch.no_grad()
def sample_timestep(
    model,
    x: Tensor,
    t: Tensor,
    i: int,
    posterior_variance: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,
    sqrt_recip_alphas: Tensor,
    device: str,
) -> Tensor:
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    Note that it also needs additional arguments about the posterior_variance, sqrt_minus_alphas_cumprod and sqrt_recip_alphas.
    """
    x = sqrt_recip_alphas[i] * (
        x - (posterior_variance[i]) / sqrt_one_minus_alphas_cumprod[i] * (model(x, t))
    )
    if i > 0:
        z = torch.randn_like(x).to(device)
        x += torch.sqrt(posterior_variance[i]) * z

    return x


@torch.no_grad()
def sample(
    model: Module,
    shape: Tuple,
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
        shape, device=device
    )  # ! check whether the variance corresponds to forward pass
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
            posterior_variance,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            device,
        )
        if pseudo_video:
            imgs.append(img)
    imgs.append(img)
    return imgs

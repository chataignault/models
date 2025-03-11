from torch import Tensor
import torch.nn.functional as F

from .diffusion import forward_diffusion_sample, get_index_from_list


def get_loss(
    model,
    x_0: Tensor,
    t: int,
    sqrt_alphas_cumprod: Tensor,
    sqrt_one_minus_alphas_cumprod: Tensor,
    device: str,
    reduction: str = "mean",
) -> Tensor:
    """
    Define the right loss given the model, the true x_0 and the time t
    """
    x_t = forward_diffusion_sample(
        x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device=device
    )
    eps = (
        x_t - get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) * x_0
    ) / get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return F.mse_loss(eps, model(x_t, t), reduction=reduction)

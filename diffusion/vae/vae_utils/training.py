import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from rich import print
import mlflow


def arctanh(x: Tensor) -> Tensor:
    return 0.5 * torch.log((1 - x) / (1 + x))


def compute_cb_normalizing_cte(x: Tensor, l_lim: float = 0.49, u_lim: float = 0.51):
    """
    Computes the value of the log of the normalizing constant of the Continuous Bernoulli loss
    In a computationnally stable way as defined in
    https://github.com/cunningham-lab/cb_and_cc/blob/master/cb/utils.py
    """
    x_safe = torch.where(
        torch.logical_or(x < l_lim, x > u_lim), x, l_lim * torch.ones_like(x)
    )
    x_unsafe = torch.where(
        torch.logical_and(x >= l_lim, x <= u_lim), x_safe, l_lim * torch.ones_like(x)
    )
    log_norm = (
        np.log(2.0)
        + torch.log(torch.abs(arctanh(1.0 - 2.0 * x_safe)))
        - torch.log(torch.abs(1.0 - 2.0 * x_safe))
    )
    taylor = (
        np.log(2.0)
        + 4.0 / 3.0 * torch.pow(x_unsafe - 0.5, 2)
        + 104.0 / 45.0 * torch.pow(x_unsafe - 0.5, 4)
    )
    return torch.where(torch.logical_or(x < l_lim, x > u_lim), log_norm, taylor)


def ELBO_loss(
    x: Tensor,
    reconstructed_x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    continuous_bernoulli: bool = False,
    eps: float = 1e-5,
) -> Tensor:
    """
    calculates ELBO loss
    Inputs:
        - x: batch of images
        - reconstructed_x: batch of reconstructed images
        - mu: batch of latent mean values
        - logvar: batch of latent logvariances
    Outputs:
        - neg_loglikelihood: average value of negative log-likelihood term across batch
        - KL_divergence: average value of KL divergence term across batch
        - loss: average ELBO loss across batch
    """
    x = torch.clamp(x, eps, 1.0 - eps)
    reconstructed_x = torch.clamp(reconstructed_x, eps, 1.0 - eps)

    neg_loglikelihood = -1 * (
        x * torch.log(reconstructed_x) + (1 - x) * torch.log(1.0 - reconstructed_x)
    )

    # continuous bernoulli normalizing constant
    if continuous_bernoulli:
        neg_loglikelihood -= compute_cb_normalizing_cte(reconstructed_x)

    neg_loglikelihood = torch.sum(neg_loglikelihood) / x.size(0)

    # KL divergence term
    var = torch.exp(logvar)
    KL_divergence = 0.5 * (
        torch.clamp(torch.sum(logvar), min=0.0) + torch.sum(var) + torch.sum(mu**2)
    ).div(x.size(0))

    loss = neg_loglikelihood + KL_divergence

    return neg_loglikelihood, KL_divergence, loss


def ELBO_loss_Gaussian(x, reconstructed_x, mu, logvar):
    """
    calculates ELBO loss when p(x|z) is assumed to be Gaussian
    Inputs:
        - x: batch of images
        - reconstructed_x: batch of reconstructed images
        - mu: batch of latent mean values
        - logvar: batch of latent logvariances
    Outputs:
        - neg_loglikelihood: average value of negative log-likelihood term across batch
        - KL_divergence: average value of KL divergence term across batch
        - loss: average ELBO loss across batch
    """
    neg_loglikelihood = F.mse_loss(reconstructed_x, x)
    var = torch.exp(logvar)
    KL_divergence = 0.5 * (
        torch.clamp(torch.sum(logvar), min=0.0) + torch.sum(var) + torch.sum(mu**2)
    )
    loss = neg_loglikelihood + KL_divergence
    return neg_loglikelihood, KL_divergence, loss


def train(
    model, nr_epochs, optimizer, criterion, dataloader, device: str, has_labels=True
):
    total_step = 0
    for epoch in range(nr_epochs):
        # iterate through batches
        for i, data in enumerate(dataloader, 0):
            total_step += 1
            # get inputs
            if has_labels == True:
                # get the inputs if data is a list of [inputs, labels]
                images, labels = data
            else:
                images = data

            images = images.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # Compute the output for all the images in the batch_size; remember we set a batch_size of 10 in the beginning!
            reconstructed_images, mu, logvar = model(images)
            # Compute the loss value
            neg_loglikelihood, KL_divergence, loss = criterion(
                images.view(-1, 784), reconstructed_images.view(-1, 784), mu, logvar
            )
            mlflow.log_metrics(
                {
                    "loss": loss.cpu().detach().item(),
                    "neg_logl": neg_loglikelihood.cpu().detach().item(),
                    "kl_div": KL_divergence.cpu().detach().item(),
                },
                step=total_step,
            )
            # Compute the gradients
            loss.backward()
            # Take the optimisation step
            optimizer.step()

        # print results for last batch
        print(
            f"Epoch: {epoch:03} | "
            f"ELBO loss: {np.round(loss.cpu().detach().numpy().item(), decimals=3)} | "
            f"KL divergence: {np.round(KL_divergence.cpu().detach().numpy().item(), decimals=3)} | "
            f"Negative log-likelihood: {np.round(neg_loglikelihood.cpu().detach().numpy().item(), decimals=2)}"
        )

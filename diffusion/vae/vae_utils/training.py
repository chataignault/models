import torch
from torch import Tensor
from torch.nn import functional as F


def ELBO_loss(x: Tensor, reconstructed_x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
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
    eps = 1e-5
    x = torch.clamp(x.view(-1, 784), eps, 1.0 - eps)
    reconstructed_x = torch.clamp(reconstructed_x, eps, 1.0 - eps)

    neg_loglikelihood = -1 * (
        x * torch.log(reconstructed_x) + (1 - x) * torch.log(1.0 - reconstructed_x)
    )

    # continuous bernoulli normalizing constant
    c = torch.log(2.0 * x) / ((1.0 - 2.0 * x) * torch.log(2.0 - 2.0 * x))
    neg_loglikelihood = neg_loglikelihood * c

    neg_loglikelihood = torch.sum(neg_loglikelihood) / x.size(0)

    var = torch.exp(logvar)
    KL_divergence = 0.5 * (
        torch.clamp(torch.sum(logvar), min=0.0) + torch.sum(var) + torch.sum(mu**2)
    ).div(x.size(0))
    loss = neg_loglikelihood + 0.5 * KL_divergence
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
    neg_loglikelihood = F.mse_loss(
        reconstructed_x, x
    )  # torch.linalg.norm(x - reconstructed_x)**2
    var = torch.exp(logvar)
    KL_divergence = 0.5 * (
        torch.clamp(torch.sum(logvar), min=0.0) + torch.sum(var) + torch.sum(mu**2)
    )
    loss = neg_loglikelihood + KL_divergence
    return neg_loglikelihood, KL_divergence, loss


def train(
    model, nr_epochs, optimizer, criterion, dataloader, device: str, has_labels=True
):
    for epoch in range(nr_epochs):
        # iterate through batches
        for i, data in enumerate(dataloader, 0):
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
                images, reconstructed_images, mu, logvar
            )
            # Compute the gradients
            loss.backward()
            # Take the optimisation step
            optimizer.step()

        # print results for last batch
        print(
            f"Epoch: {epoch:03} | ELBO loss: {loss} | KL divergence: {KL_divergence} | Negative log-likelihood: {neg_loglikelihood}"
        )

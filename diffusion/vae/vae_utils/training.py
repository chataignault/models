import torch
from torch.nn import functional as F


def ELBO_loss(x, reconstructed_x, mu, logvar):
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
    neg_loglikelihood = F.binary_cross_entropy(
        reconstructed_x, x.view(-1, 784), reduction="sum"
    ).div(
        x.size(0)
    )  # Cross-entropy between outputted image and (flattened) original image
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

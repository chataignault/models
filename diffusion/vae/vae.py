import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt


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


def train(model, nr_epochs, optimizer, criterion, dataloader, has_labels=True):
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

    # model.to('cpu')
    print("Finished Training")


class VAE(nn.Module):
    def __init__(self, latent_dim=2, input_dim=784):
        """builds VAE
        Inputs:
            - latent_dim: dimension of latent space
            - input_dim: dimension of input space
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.relu = nn.ReLU()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.encoder_mean = nn.Linear(128, latent_dim)
        self.encoder_logvar = nn.Linear(128, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 512)
        self.fc5 = nn.Linear(512, 784)

    def encode(self, x):
        """take an image, and return latent space mean and log variance
        Inputs:
            -x: batch of images flattened to 784
        Outputs:
            -means in latent dimension
            -logvariances in latent dimension
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.encoder_mean(x), self.encoder_logvar(x)

    def reparametrise(self, mu, logvar):
        """Sample in latent space according to mean and logvariance
        Inputs:
            -mu: batch of means
            -logvar: batch of logvariances
        Outputs:
            -samples: batch of latent samples
        """
        var = torch.exp(logvar)
        samples = mu + torch.sqrt(var) * torch.normal(torch.zeros(mu.shape), std=1.0)
        return samples

    def decode(self, z):
        """Decode latent space samples
        Inputs:
            -z: batch of latent samples
        Outputs:
            -x_recon: batch of reconstructed images
        """
        x_recon = self.relu(self.fc3(z))
        x_recon = self.relu(self.fc4(x_recon))
        x_recon = self.fc5(x_recon)
        x_recon = torch.sigmoid(x_recon)
        return x_recon

    def forward(self, x):
        """Do full encode and decode of images
        Inputs:
            - x: batch of images
        Outputs:
            - self.decode(z): batch of reconstructed images
            - mu: batch of latent mean values
            - logvar: batch of latent logvariances
        """
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrise(mu, logvar)
        return self.decode(z), mu, logvar


def sample_images(model: VAE, num_im: int):
    """
    Inputs:
        - model: trained vae
        - num_im: int, number of images to sample
    """
    # Fill in steps
    # sample latent variables from p(z)
    mu_batch, logvar_batch = (
        0.0 * torch.ones((num_im, 2)),
        1.0 * torch.ones((num_im, 2)),
    )
    z = model.reparametrise(mu_batch, logvar_batch)

    # pass latent variables through the decoder
    output = model.decode(z)

    # reshape output of model into 28x28
    images = output.reshape((num_im, 28, 28))

    # plot images
    fig, ax = plt.subplots(nrows=1, ncols=num_im)
    for i in range(num_im):
        image = images[i]
        ax[i].imshow(image.detach().cpu().numpy(), cmap="gray")
        ax[i].axis("off")

    plt.show()


if __name__ == "__main__":
    batch_size = 128
    device = "cuda"
    torch.set_default_device(device)

    trainset = torchvision.datasets.MNIST(
        root="./",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=torch.Generator(device=device),
    )

    vae = VAE().to(device)
    train(
        model=vae,
        nr_epochs=20,
        optimizer=torch.optim.Adam(vae.parameters(), lr=0.001),
        criterion=ELBO_loss,
        dataloader=trainloader,
    )

    sample_images(vae, 5)

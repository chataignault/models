import torch
from torch import nn
from matplotlib import pyplot as plt


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
    _, ax = plt.subplots(nrows=1, ncols=num_im)
    for i in range(num_im):
        image = images[i]
        ax[i].imshow(image.detach().cpu().numpy(), cmap="gray")
        ax[i].axis("off")

    plt.show()

import os
import torch
from torch import nn, Tensor
from matplotlib import pyplot as plt


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2, input_dim: int = 784, depth: int = 3):
        """builds VAE
        Inputs:
            - latent_dim: dimension of latent space
            - input_dim: dimension of input space
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.relu = nn.ReLU()

        dim_step = (input_dim - latent_dim) // depth
        self.encoder_dimensions = [input_dim - dim_step * k for k in range(depth)]

        # Encoder layers
        self.encoder = nn.ModuleList([])
        for d, d_next in zip(self.encoder_dimensions[:-1], self.encoder_dimensions[1:]):
            self.encoder.append(nn.ModuleList([nn.Linear(d, d_next), nn.ReLU()]))

        self.encoder_mean = nn.Linear(self.encoder_dimensions[-1], latent_dim)
        self.encoder_logvar = nn.Linear(self.encoder_dimensions[-1], latent_dim)

        # Decoder layers
        self.decoder = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(latent_dim, self.encoder_dimensions[-1]), nn.ReLU()]
                )
            ]
        )
        for d, d_next in zip(
            self.encoder_dimensions[:0:-1], self.encoder_dimensions[-2::-1]
        ):
            is_last = d_next == self.encoder_dimensions[0]
            if is_last:
                self.decoder.append(nn.ModuleList([nn.Linear(d, d_next), nn.Sigmoid()]))
            else:
                self.decoder.append(nn.ModuleList([nn.Linear(d, d_next), nn.ReLU()]))

    def encode(self, x):
        """take an image, and return latent space mean and log variance
        Inputs:
            -x: batch of images flattened to 784
        Outputs:
            -means in latent dimension
            -logvariances in latent dimension
        """
        for lin, relu in self.encoder:
            x = relu(lin(x))
        return self.encoder_mean(x), self.encoder_logvar(x)

    def reparametrise(self, mu: Tensor, logvar: Tensor):
        """Sample in latent space according to mean and logvariance
        Inputs:
            -mu: batch of means
            -logvar: batch of logvariances
        Outputs:
            -samples: batch of latent samples
        """
        std = torch.exp(0.5 * logvar)
        samples = mu + std * torch.normal(torch.zeros(mu.shape), std=1.0)
        return samples

    def decode(self, z: Tensor):
        """Decode latent space samples
        Inputs:
            -z: batch of latent samples
        Outputs:
            -x_recon: batch of reconstructed images
        """
        for lin, relu_or_sig in self.decoder:
            z = relu_or_sig(lin(z))
        return z

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


def sample_images(model: VAE, num_im: int, path: str, name: str):
    n_cols = 5
    latent_dim = model.latent_dim
    # sample latent variables from p(z)
    mu_batch, logvar_batch = (
        0.0 * torch.ones((num_im, latent_dim)),
        1.0 * torch.ones((num_im, latent_dim)),
    )
    z = model.reparametrise(mu_batch, logvar_batch)

    # pass latent variables through the decoder
    output = model.decode(z)

    # reshape output of model into 28x28
    images = output.reshape((num_im, 28, 28))

    # plot images
    fig, axs = plt.subplots(nrows=num_im // n_cols, ncols=n_cols)
    for i in range(num_im):
        r, c = i // n_cols, i % n_cols
        image = images[i]
        axs[r, c].imshow(image.detach().cpu().numpy(), cmap="gray")
        axs[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(path, f"vae_{name}.png"))
    plt.show()


# We will use nn.Sequential to build the encoders and decoders - we will use the View class to resize the images
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(self.size)  # used to resize tensor to self.size


class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        """builds VAE
        Inputs:
            - d: dimension of latent space
        """
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        # Build VAE here
        self.encoder, self.decoder, self.encoder_mean, self.encoder_lv = self.build_VAE(
            latent_dim
        )

    def build_VAE(self, latent_dim: int):
        """builds VAE with specified latent dimension and number of layers
        Inputs:
            -d: latent dimension
        """
        encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 32, 7, 3, 0),  # 32, 8, 8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 3, 0),  # 128, 2, 2
            # Flatten images
            View((-1, 128 * 2 * 2)),
            # Linear layers
            nn.Linear(128 * 2 * 2, 16),
            nn.ReLU(),
        )

        encoder_mean = nn.Linear(16, latent_dim)
        encoder_lv = nn.Linear(16, latent_dim)

        decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 2 * 2),
            View((-1, 128, 2, 2)),
            # Transposed convolution layers - args: in_channels, out_channels, kernel_size, stride, padding
            nn.ConvTranspose2d(128, 64, 5, 3, 0),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 8, 7, 3, 0),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

        return encoder, decoder, encoder_mean, encoder_lv

    def encode(self, x: Tensor):
        """take an image, and return latent space mean + log variance
        Inputs:
            -images, x
        Outputs:
            -means in latent dimension
            -logvariances in latent dimension
        """
        h = self.encoder(x)
        return self.encoder_mean(h), self.encoder_lv(h)

    def reparametrise(self, mu: Tensor, logvar: Tensor):
        """Sample in latent space according to mean and logvariance
        Inputs:
            -mu: batch of means
            -logvar: batch of logvariances
        Outputs:
            -samples: batch of latent samples
        """
        samples = mu + torch.exp(0.5 * logvar) * torch.normal(
            torch.zeros(mu.shape), std=1.0
        )
        return samples

    def decode(self, z: Tensor):
        """Decode latent space samples
        Inputs:
            -z: batch of latent samples
        Outputs:
            -x_recon: batch of reconstructed images
        """
        raw_out = self.decoder(z)
        x_recon = torch.sigmoid(
            raw_out
        )  # pass raw output through sigmoid activation function
        return x_recon

    def forward(self, x: Tensor):
        """Do full encode and decode of images
        Inputs:
            - x: batch of images
        Outputs:
            - batch of reconstructed images
            - mu: batch of latent mean values
            - logvar: batch of latent logvariances
        """
        mu, logvar = self.encode(x)
        z = self.reparametrise(mu, logvar)
        return self.decode(z), mu, logvar

import torch
from torch import nn, Tensor
from matplotlib import pyplot as plt


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2, input_dim: int = 784):
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
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.encoder_mean = nn.Linear(64, latent_dim)
        self.encoder_logvar = nn.Linear(64, latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, input_dim)

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
        x = self.relu(self.fc3(x))
        return self.encoder_mean(x), self.encoder_logvar(x)

    def reparametrise(self, mu: Tensor, logvar: Tensor):
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

    def decode(self, z: Tensor):
        """Decode latent space samples
        Inputs:
            -z: batch of latent samples
        Outputs:
            -x_recon: batch of reconstructed images
        """
        x_recon = self.relu(self.fc4(z))
        x_recon = self.relu(self.fc5(x_recon))
        x_recon = self.relu(self.fc6(x_recon))
        x_recon = self.fc7(x_recon)
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


def sample_images(model: VAE, num_im: int, name: str):
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
    fig.savefig(f"vae_{name}.png")
    plt.show()


# We will use nn.Sequential to build the encoders and decoders - we will use the View class to resize the images
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)  # used to resize tensor to self.size


class CVAE(nn.Module):
    def __init__(self, d=20):
        """builds VAE
        Inputs:
            - d: dimension of latent space
        """
        super(CVAE, self).__init__()
        self.d = d
        # Build VAE here
        self.encoder, self.decoder, self.encoder_mean, self.encoder_lv = self.build_VAE(
            d
        )

    def build_VAE(self, d):
        """builds VAE with specified latent dimension and number of layers
        Inputs:
            -d: latent dimension
        """
        # Input size: 64x64
        encoder = nn.Sequential(
            # Convolutional layers - args: in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(
                3, 32, 4, 2, 1
            ),  # example of how your encoder could start - you may want to change this
            nn.ReLU(True),
            # Fill in
            nn.Conv2d(32, 64, 4, 1, 1),
            # nn.ReLU(True),
            # Flatten images
            nn.Flatten(start_dim=1),
            # Linear layers
            # Fill in
            nn.Linear(64 * 31 * 31, 1024),
            nn.Linear(1024, 512),
            # nn.Linear(512, 128),
        )

        encoder_mean = nn.Linear(512, d)
        encoder_lv = nn.Linear(512, d)

        decoder = nn.Sequential(
            # Linear layers
            # Fill in
            nn.Linear(d, 512),
            # nn.Linear(128, 512),
            nn.Linear(512, 1024),
            # nn.Linear(1024, 32*32*32),
            nn.Linear(1024, 64 * 31 * 31),
            # Reshape images
            # View((-1, 32, 32, 32)), # args should be (-1, num_channels, n, n) where you want to resize to images of dimension nxn
            View((-1, 64, 31, 31)),
            # Transposed convolution layers - args: in_channels, out_channels, kernel_size, stride, padding
            # Fill in - make sure the output size is 64x64. Do not pass your final output through any activation functions
            nn.ConvTranspose2d(64, 32, 4, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            # nn.Linear(128, 64)
        )

        return encoder, decoder, encoder_mean, encoder_lv

    def encode(self, x):
        """take an image, and return latent space mean + log variance
        Inputs:
            -images, x
        Outputs:
            -means in latent dimension
            -logvariances in latent dimension
        """
        h1 = self.encoder(x)
        return self.encoder_mean(h1), self.encoder_lv(h1)

    def reparametrise(self, mu, logvar):
        """Sample in latent space according to mean and logvariance
        Inputs:
            -mu: batch of means
            -logvar: batch of logvariances
        Outputs:
            -samples: batch of latent samples
        """
        samples = mu + torch.exp(logvar) * torch.normal(torch.zeros(mu.shape), std=1.0)
        return samples

    def decode(self, z):
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

    def forward(self, x):
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

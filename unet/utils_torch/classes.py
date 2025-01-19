import torch
from torch import Tensor
from torch import nn
import numpy as np
from torchvision.transforms import CenterCrop


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        """
        Output shape is len(time)
        """
        n = len(time)
        position = torch.arange(n).unsqueeze(1)
        div_term_even = torch.exp(
            torch.log(position)
            + torch.arange(0, self.dim, 2) * (-np.log(10000.0) / self.dim)
        )
        div_term_odd = torch.exp(
            torch.log(position)
            + torch.arange(1, self.dim, 2) * (-np.log(10000.0) / self.dim)
        )
        pe = torch.zeros(n, self.dim)
        pe[:, 0::2] = torch.sin(div_term_even)
        if self.dim > 1:
            pe[:, 1::2] = torch.cos(div_term_odd)
        return pe


class Block(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        up: bool = False,
        attention: bool = False,
    ):
        """
        in_ch refers to the number of channels in the input to the operation and out_ch how many should be in the output
        """
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, time_emb_dim)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(
                out_ch, out_ch, kernel_size=4, stride=2, padding=1
            )

        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Define the forward pass making use of the components above.
        Time t should get mapped through the time_mlp layer + a relu
        The input x should get mapped through a convolutional layer with relu / batchnorm
        The time embedding should get added the output from the input convolution
        A second convolution should be applied and finally passed through the self.transform.
        """
        t = self.time_mlp(t)
        x = self.conv1(x)
        x = self.relu(x)
        x = x + t.unsqueeze(1).unsqueeze(1).repeat(
            (1, x.size(1), x.size(-1), x.size(-2) // t.shape[1])
        )
        x = self.bnorm1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.transform(x)
        x = self.bnorm2(x)

        return x


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = [
            32,
            64,
            128,
        ]
        up_channels = [
            128,
            64,
            32,
        ]
        out_dim = 1
        time_emb_dim = 7

        self.pos_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.init_conv = nn.Conv2d(
            in_channels=image_channels,
            out_channels=down_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.downsampling = nn.Sequential(
            *[
                Block(
                    down_channels[i],
                    down_channels[i + 1],
                    time_emb_dim,
                    attention=(i == 1),
                )
                for i in range(len(down_channels) - 1)
            ]
        )
        self.conv_int1 = nn.Conv2d(128, 64, 1)
        self.relu = nn.ReLU()
        self.conv_int2 = nn.Conv2d(64, 128, 1)

        self.bnorm = nn.BatchNorm2d(64)

        self.upsampling = nn.Sequential(
            *[
                Block(
                    up_channels[i],
                    up_channels[i + 1],
                    time_emb_dim,
                    up=True,
                    attention=(i == 0),
                )
                for i in range(len(up_channels) - 1)
            ]
        )

        self.out_conv = nn.Conv2d(
            in_channels=up_channels[-1], out_channels=out_dim, kernel_size=1
        )

    def forward(self, x: Tensor, t: Tensor):
        t = self.pos_emb(t)
        x = x + t.unsqueeze(1).unsqueeze(1).repeat(
            (1, x.size(1), x.size(-1), x.size(-2) // t.shape[1])
        )

        x = self.init_conv(x)
        x = self.relu(x)

        x_down_ = []
        for block in self.downsampling.children():
            x = block(x, t)
            x_down_.append(x)

        x = self.conv_int1(x)
        x = self.relu(x)
        x = self.bnorm(x)

        x = self.conv_int2(x)
        x = self.relu(x)

        for k, block in enumerate(self.upsampling.children(), 1):
            residual = x_down_[-k]
            # crop residual to match x dimensions
            residual = CenterCrop(x.size(2))(residual)
            x_extended = torch.cat((x, residual), dim=1)

            x = block(x_extended, t)

        x = self.out_conv(x)

        return x

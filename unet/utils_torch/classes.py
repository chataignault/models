import torch
from torch import Tensor
from torch import nn
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor, dim: int = None) -> Tensor:
        """
        Output shape len(time), self.dim
        """
        n = len(time)
        position = torch.arange(n).unsqueeze(1)
        dim = dim or self.dim
        div_term_even = torch.exp(
            torch.log(position) + torch.arange(0, dim, 2) * (-np.log(10000.0) / dim)
        )
        div_term_odd = torch.exp(
            torch.log(position) + torch.arange(1, dim, 2) * (-np.log(10000.0) / dim)
        )
        pe = torch.zeros(n, dim)
        pe[:, 0::2] = torch.sin(div_term_even)
        if dim > 1:
            pe[:, 1::2] = torch.cos(div_term_odd)
        return pe


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, time_emb_dim: int):
        """
        in_ch refers to the number of channels in the input to the operation and out_ch how many should be in the output
        """
        super().__init__()

        self.lintemb = nn.Linear(time_emb_dim, in_ch)
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=1)
        self.bnorm1 = nn.BatchNorm2d(in_ch)
        self.bnorm2 = nn.BatchNorm2d(in_ch)
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
        h = x
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        t = self.lintemb(self.relu(t)).unsqueeze(-1).unsqueeze(-1)
        x = x + t
        x = self.bnorm2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x + h
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        up: bool = False,
    ):
        """
        in_ch refers to the number of channels in the input to the operation and out_ch how many should be in the output
        """
        super().__init__()

        self.up = up

        self.lintemb = nn.Linear(time_emb_dim, in_ch)

        if up:
            self.bnorm1 = nn.BatchNorm2d(2 * in_ch)
            self.conv1 = nn.Conv2d(2 * in_ch, in_ch, 3, padding=1, stride=1)
            self.squish_conv = nn.Conv2d(
                2 * in_ch, in_ch, kernel_size=3, padding=1, stride=1
            )
            self.transform = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1
            )

        else:
            self.bnorm1 = nn.BatchNorm2d(in_ch)
            self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=1)
            self.transform = nn.Conv2d(in_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=1)
        self.bnorm2 = nn.BatchNorm2d(in_ch)
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
        h = x
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        t = self.lintemb(self.relu(t)).unsqueeze(-1).unsqueeze(-1)
        x = x + t
        x = self.bnorm2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.up:
            h = self.squish_conv(h)
        x = self.relu(x)  # ! check
        x = x + h

        return self.transform(x), x


class AttentionBlock(nn.Module):
    """ """

    def __init__(self, in_ch: int, hidden_dim: int, time_embed_dim: int):
        super().__init__()
        self.lintemb = nn.Linear(time_embed_dim, in_ch)
        self.relu = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(in_ch)
        self.k = nn.Linear(49, 49)
        self.q = nn.Linear(49, 49)
        self.v = nn.Linear(49, 49)
        self.attention = nn.MultiheadAttention(49, 7, batch_first=True)

    def forward(self, x, t):
        t = self.relu(self.lintemb(t)).unsqueeze(-1).unsqueeze(-1)
        x = x + t
        x = self.bnorm(x)
        x = self.relu(x)
        N, B, D, _ = x.shape
        x = x.reshape((N, B, D * D))
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        x, _ = self.attention(query, key, value)
        # x, _ = self.attention(x, x, x)
        x = x.reshape((N, B, D, D))
        return x


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(
        self,
        time_emb_dim: int = 4,
    ):
        super().__init__()
        image_channels = 1
        down_channels = [
            8,
            32,
            128,
        ]
        up_channels = down_channels[::-1]

        self.time_emb_dim = time_emb_dim

        self.pos_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=time_emb_dim),
            nn.Linear(time_emb_dim, 4 * time_emb_dim),
            nn.ReLU(),
            nn.Linear(4 * time_emb_dim, 4 * time_emb_dim),
        )

        self.init_conv = nn.Conv2d(
            in_channels=image_channels,
            out_channels=down_channels[0],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.downsampling = nn.Sequential(
            *[
                Block(down_channels[i], down_channels[i + 1], 4 * time_emb_dim)
                for i in range(len(down_channels) - 1)
            ]
        )

        self.resint1 = ResBlock(down_channels[-1], 4 * time_emb_dim)
        self.bnorm = nn.BatchNorm2d(down_channels[-1])
        self.attention_int = AttentionBlock(
            down_channels[-1], down_channels[-1], 4 * time_emb_dim
        )
        self.relu = nn.ReLU()
        self.resint2 = ResBlock(down_channels[-1], 4 * time_emb_dim)

        self.upsampling = nn.Sequential(
            *[
                Block(
                    up_channels[i],
                    up_channels[i + 1],
                    4 * time_emb_dim,
                    up=True,
                )
                for i in range(len(up_channels) - 1)
            ]
        )

        self.bnorm_out = nn.BatchNorm2d(up_channels[-1])
        self.out_conv = nn.Conv2d(
            in_channels=up_channels[-1], out_channels=1, kernel_size=1
        )

    def forward(self, x: Tensor, t: Tensor):
        t = self.pos_emb(t)
        x = self.init_conv(x)
        x_down_ = [x]
        for block in self.downsampling.children():
            x, h = block(x, t)
            x_down_.append(h)
        x_down_.append(x)
        x = self.resint1(x, t)
        x = self.attention_int(x, t)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.resint2(x, t)
        for k, block in enumerate(self.upsampling.children(), 1):
            residual = x_down_[-k]
            x_extended = torch.cat([x, residual], dim=1)
            x, _ = block(x_extended, t)
        # add the ultimate residual from the initial convolution
        x = x + x_down_[0]
        x = self.bnorm_out(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x

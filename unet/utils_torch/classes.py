import numpy as np
from typing import List
import torch
from torch import nn
from torch import Tensor
from torch.optim import Adam
import lightning as L
from torch.utils.tensorboard import SummaryWriter

from .training import get_loss
from torch.optim.lr_scheduler import (
    LinearLR,
    ConstantLR,
    ExponentialLR,
    SequentialLR,
)

writer = SummaryWriter()


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
    def __init__(self, in_ch: int, time_emb_dim: int, dropout: float = 0.01):
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
        self.dropout = nn.Dropout(dropout)

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
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + h
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        up: bool = False,
        attention: bool = False,
    ):
        super().__init__()
        self.use_attention = attention
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

        if attention:
            self.attention = AttentionBlock(in_ch, in_ch, 4, time_emb_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Time t should get mapped through the time_mlp layer + a relu
        The input x should get mapped through a convolutional layer with relu / batchnorm
        The time embedding should get added the output from the input convolution
        A second convolution should be applied and finally passed through the self.transform.
        """
        h = x
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        t_ = self.relu(self.lintemb(t)).unsqueeze(-1).unsqueeze(-1)
        x = x + t_
        x = self.bnorm2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.up:
            h = self.squish_conv(h)
        # x = self.relu(x)  # ! check
        x = x + h

        if self.use_attention:
            x = self.attention(x, t)

        return self.transform(x), x


class AttentionBlockManual(nn.Module):
    """
    Step-by-step implementation of dot-product attention
    """

    def __init__(
        self,
        in_ch: int,
        hidden_dim: int,
        n_heads: int,
        time_embed_dim: int,
    ):
        super().__init__()
        self.d = hidden_dim
        self.lintemb = nn.Linear(time_embed_dim, in_ch)
        self.relu = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(in_ch)
        self.k = nn.Linear(in_ch, hidden_dim, bias=False)
        self.q = nn.Linear(in_ch, hidden_dim, bias=False)
        self.v = nn.Linear(in_ch, hidden_dim, bias=False)
        self.sm = nn.Softmax(-1)
        self.proj = nn.Linear(hidden_dim, in_ch)

    def forward(self, x, t):
        h = x
        N, C, H, W = x.shape
        t = self.relu(self.lintemb(t)).unsqueeze(-1).unsqueeze(-1)
        x = x + t
        x = x.transpose(1, 3)  # N, W, H, C
        query = self.q(x)  # N, W, H, x
        key = self.k(x)  # N, W, H, x
        qk = torch.einsum("NWHx,Nwhx->NWHwh", query, key) / np.sqrt(self.d)
        qk = qk.reshape((N, W, H, W * H))
        qk = self.sm(qk)
        qk = qk.reshape((N, W, H, W, H))
        value = self.v(x)
        x = torch.einsum("NWHwh,Nwhx->NWHx", qk, value)
        x = self.proj(x)  # N, W, H, C
        x = x.transpose(1, 3).reshape((N, C, H, W))
        return x + h


class AttentionBlock(nn.Module):
    """ """

    def __init__(
        self,
        in_ch: int,
        hidden_dim: int,
        n_heads: int,
        time_embed_dim: int,
    ):
        super().__init__()
        self.lintemb = nn.Linear(time_embed_dim, in_ch)
        self.relu = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(in_ch)
        self.k = nn.Linear(in_ch, hidden_dim, bias=True)
        self.q = nn.Linear(in_ch, hidden_dim, bias=True)
        self.v = nn.Linear(in_ch, hidden_dim, bias=True)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.proj = nn.Linear(hidden_dim, in_ch)

    def forward(self, x, t):
        h = x
        t = self.relu(self.lintemb(t)).unsqueeze(-1).unsqueeze(-1)
        x = x + t
        # x = self.bnorm(x)
        # x = self.relu(x)
        N, B, D, _ = x.shape
        x = x.reshape((N, B, D * D)).transpose(1, 2)
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        x, _ = self.attention(query, key, value)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape((N, B, D, D))
        return x + h


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(
        self,
        down_channels: List[int],
        time_emb_dim: int = 16,
        hidden_dim: int = 256,
        n_heads: int = 4,
    ):
        super().__init__()
        image_channels = 1
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
            out_channels=down_channels[0] // 2,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.init_res = ResBlock(down_channels[0] // 2, 4 * time_emb_dim)

        self.init_conv2 = nn.Conv2d(
            in_channels=down_channels[0] // 2,
            out_channels=down_channels[0],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        attention_depth = (len(down_channels) - 1) // 2

        self.downsampling = nn.Sequential(
            *[
                Block(
                    down_channels[i],
                    down_channels[i + 1],
                    4 * time_emb_dim,
                    attention=(i == attention_depth),
                )
                for i in range(len(down_channels) - 1)
            ]
        )

        self.resint1 = ResBlock(down_channels[-1], 4 * time_emb_dim)
        self.bnorm = nn.BatchNorm2d(down_channels[-1])
        self.attention_int = AttentionBlock(
            down_channels[-1], hidden_dim, n_heads, 4 * time_emb_dim
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
                    attention=(i == (len(down_channels) - 2 - attention_depth)),
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
        x = self.init_res(x, t)
        x = self.init_conv2(x)
        x_down_ = [x]
        for block in self.downsampling.children():
            x, h = block(x, t)
            x_down_.append(h)
        x_down_.append(x)
        x = self.resint1(x, t)
        x = self.bnorm(x)
        x = self.attention_int(x, t)
        # x = self.relu(x)
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


class Unet(nn.Module):
    """ """

    def __init__(
        self,
        time_emb_dim: int = 4,
        down_channels=[
            8,
            32,
            128,
        ],
    ):
        super().__init__()
        image_channels = 1
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
            out_channels=down_channels[0] // 2,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.init_res = ResBlock(down_channels[0] // 2, 4 * time_emb_dim)

        self.init_conv2 = nn.Conv2d(
            in_channels=down_channels[0] // 2,
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
        self.attention_int = AttentionBlock(down_channels[-1], 128, 8, 4 * time_emb_dim)
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
        x = self.init_res(x, t)
        x = self.init_conv2(x)
        x_down_ = [x]
        for block in self.downsampling.children():
            x, h = block(x, t)
            x_down_.append(h)

        x_down_.append(x)
        h = x
        x = self.resint1(x, t)
        x = self.attention_int(x, t)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.resint2(x, t)
        x = x + h
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


class Unet2(nn.Module):
    """ """

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
        self.attention_int = AttentionBlock(down_channels[-1], 24, 4 * time_emb_dim)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()

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
        x = self.silu(x)
        x = self.resint2(x, t)
        for k, block in enumerate(self.upsampling.children(), 1):
            residual = x_down_[-k]
            x_extended = torch.cat([x, residual], dim=1)
            x, _ = block(x_extended, t)
        # add the ultimate residual from the initial convolution
        x = x + x_down_[0]
        x = self.bnorm_out(x)
        x = self.silu(x)
        x = self.out_conv(x)
        return x


class LitUnet(L.LightningModule):
    def __init__(
        self, unet, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, T, device, lr
    ):
        super().__init__()
        self.unet = unet
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.T = T
        self.dev = device
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        timestep = torch.randint(1, self.T, (x.shape[0],))
        loss = get_loss(
            self.unet,
            x,
            timestep,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.dev,
        )
        writer.add_scalar("Loss", loss, self.global_step)
        writer.add_scalar("Learning Rate", self.lr)
        writer.flush()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimiser = Adam(self.parameters(), lr=self.lr)
        stepping_batches = self.trainer.estimated_stepping_batches
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5*self.lr, total_steps=stepping_batches)
        scheduler = SequentialLR(
            optimiser,
            schedulers=[
                # LinearLR(optimiser, 0.1, 1.0, 5),
                ConstantLR(optimiser, 1.0),
                ExponentialLR(optimiser, 0.98),
            ],
            # milestones=[5, 15],
            milestones=[20],
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

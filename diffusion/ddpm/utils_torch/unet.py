import numpy as np
from typing import List
from matplotlib.pyplot import imshow
import torch
import torchvision
from torch import nn
from torch import Tensor
from torch.optim import Adam
import lightning as L
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.utilities import grad_norm

from .training import get_loss
from .diffusion import sample
from torch.optim.lr_scheduler import (
    LinearLR,
    ConstantLR,
    ExponentialLR,
    SequentialLR,
)

from .modules import *


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(
        self,
        down_channels: List[int],
        time_emb_dim: int = 16,  # 16
        hidden_dim: int = 64,
        n_heads: int = 8,  # 4
        n_heads_inter: int = 4,
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

        self.init_conv = UpConvBlock(image_channels, down_channels[0], time_emb_dim)

        attention_depth = (len(down_channels) - 1) // 2

        self.downsampling = nn.Sequential(
            *[
                Block(
                    down_channels[i],
                    down_channels[i + 1],
                    4 * time_emb_dim,
                    attention=int(i == attention_depth) * n_heads_inter,
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
        # self.resint2 = ResBlock(down_channels[-1], 4 * time_emb_dim)

        self.upsampling = nn.Sequential(
            *[
                Block(
                    up_channels[i],
                    up_channels[i + 1],
                    4 * time_emb_dim,
                    up=True,
                    attention=int(i == (len(down_channels) - 2 - attention_depth))
                    * n_heads_inter,
                )
                for i in range(len(up_channels) - 1)
            ]
        )

        # self.bnorm_out = nn.BatchNorm2d(up_channels[-1])
        self.out_conv = nn.Conv2d(
            in_channels=up_channels[-1], out_channels=1, kernel_size=1
        )

    def forward(self, x: Tensor, t: Tensor):
        t = self.pos_emb(t)
        x = self.init_conv(x, t)
        x_down_ = [x]
        for block in self.downsampling.children():
            x, h = block(x, t)
            x_down_.append(h)
        x_down_.append(x)
        x = self.resint1(x, t)
        x = self.bnorm(x)
        # x = self.relu(x)
        x = self.attention_int(x, t)
        # x = self.resint2(x, t)
        for k, block in enumerate(self.upsampling.children(), 1):
            residual = x_down_[-k]
            x_extended = torch.cat([x, residual], dim=1)
            x, _ = block(x_extended, t)
        x = x + x_down_[0]
        # x = self.bnorm_out(x)
        # x = self.relu(x)
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


class LitUnet(L.LightningModule):
    def __init__(
        self,
        unet: nn.Module,
        sqrt_alphas_cumprod: Tensor,
        sqrt_one_minus_alphas_cumprod: Tensor,
        T: int,
        device: str,
        lr: float,
        writer: SummaryWriter,
        img_size: int,
        posterior_variance: Tensor,
    ):
        super().__init__()
        self.unet = unet
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.T = T
        self.dev = device
        self.lr = lr
        self.writer = writer
        self.img_size = img_size
        self.posterior_variance = posterior_variance

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.unet, norm_type=2)
        self.log_dict(norms)

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
        self.writer.add_scalar("Loss", loss, self.global_step)

        if self.global_step % 500 == 0 and self.global_step > 0:
            self.unet.eval()
            samp = sample(
                self.unet,
                (16, 1, self.img_size, self.img_size),
                self.posterior_variance,
                self.sqrt_one_minus_alphas_cumprod,
                1.0 / torch.sqrt(1 - self.posterior_variance),
                self.T,
            )[-1]
            # normalize
            # ? clip instead
            samp = samp - samp.min(dim=0)[0]
            samp = samp / samp.max(dim=0)[0]

            # log samples to board
            img_grid = torchvision.utils.make_grid(samp)
            imshow(np.transpose(img_grid.cpu().numpy(), (1, 2, 0)), aspect="auto")
            self.writer.add_image(
                f"generated samples step={self.global_step}", img_grid
            )
            self.unet.train()

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimiser = Adam(self.parameters(), lr=self.lr)
        # stepping_batches = self.trainer.estimated_stepping_batches
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5*self.lr, total_steps=stepping_batches)
        scheduler = SequentialLR(
            optimiser,
            schedulers=[
                LinearLR(optimiser, 0.1, 1.0, 2),
                ConstantLR(optimiser, 1.0),
                ExponentialLR(optimiser, 0.99),
            ],
            # milestones=[5, 15],
            milestones=[2, 20],
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

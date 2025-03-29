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
from collections import defaultdict

from .training import get_loss
from .diffusion import sample
from torch.optim.lr_scheduler import (
    LinearLR,
    ConstantLR,
    ExponentialLR,
    SequentialLR,
)
from torch.nn import ModuleList
from .modules import *
from einops import rearrange, repeat
from functools import partial


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t):
        t = self.mlp(t)
        t = rearrange(t, "b c -> b c 1 1")

        h = self.block1(x)

        h = self.block2(h)

        return h + self.res_conv(x)


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(
        self,
        downs: List[int],
        time_emb_dim: int = 16,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_heads_inter: int = 4,
    ):
        super().__init__()
        image_channels = 1
        ups = downs[::-1]
        in_out = list(zip(downs[:-1], downs[1:]))
        self.time_emb_dim = time_emb_dim

        self.pos_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=time_emb_dim),
            nn.Linear(time_emb_dim, 4 * time_emb_dim),
            nn.GELU(),
            nn.Linear(4 * time_emb_dim, 4 * time_emb_dim),
        )

        self.init_conv = nn.Conv2d(image_channels, downs[0], 7, padding=3)

        self.downsampling = ModuleList([])
        self.upsampling = ModuleList([])

        for in_dim, out_dim in in_out:
            is_last = out_dim == downs[-1]
            self.downsampling.append(
                ModuleList(
                    [
                        ResnetBlock(in_dim, in_dim, 4 * time_emb_dim),
                        ResnetBlock(in_dim, in_dim, 4 * time_emb_dim),
                        LinearAttention(in_dim, heads=n_heads_inter),
                        nn.Conv2d(in_dim, out_dim, 4, 2, 1)
                        if not is_last
                        else nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                    ]
                )
            )

        self.resint1 = ResnetBlock(downs[-1], hidden_dim, 4 * time_emb_dim)
        self.attention_int = LinearAttention(hidden_dim, heads=n_heads)
        self.resint2 = ResnetBlock(hidden_dim, downs[-1], 4 * time_emb_dim)

        for in_dim, out_dim in in_out[::-1]:
            is_last = in_dim == ups[-1]
            self.upsampling.append(
                ModuleList(
                    [
                        ResnetBlock(in_dim + out_dim, out_dim, 4 * time_emb_dim),
                        ResnetBlock(in_dim + out_dim, out_dim, 4 * time_emb_dim),
                        LinearAttention(out_dim, heads=n_heads_inter),
                        nn.ConvTranspose2d(
                            out_dim, in_dim, kernel_size=4, stride=2, padding=1
                        )
                        if not is_last
                        else nn.Conv2d(out_dim, in_dim, 3, 1, 1),
                    ]
                )
            )

        self.end_res = ResnetBlock(2 * ups[-1], ups[-1], 4 * time_emb_dim)
        self.out_conv = nn.Conv2d(in_channels=ups[-1], out_channels=1, kernel_size=1)

        assert len(self.upsampling) == len(self.downsampling), (
            f"up and down channels should have the same length, got {len(self.downsampling)} and {len(self.upsampling)}"
        )

    def forward(self, x: Tensor, t: Tensor):
        t = self.pos_emb(t)
        x = self.init_conv(x)
        h = [x.clone()]

        for block1, block2, attn, downsample in self.downsampling:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.resint1(x, t)
        x = self.attention_int(x) + x
        x = self.resint2(x, t)

        for block1, block2, attn, upsample in self.upsampling:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)

            x = attn(x) + x

            x = upsample(x)

        x = torch.cat([x, h.pop()], dim=1)
        assert len(h) == 0, f"all residuals should be used, but remains {len(h)}"
        x = self.end_res(x, t)
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
        x = self.bnorm_out(x)  # remove this
        x = self.relu(x)  # remove this
        x = self.out_conv(x)
        # can also try some soft non linearity
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

        self.timestep_losses = defaultdict(int)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.unet, norm_type=2)
        self.log_dict(norms)

    def training_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        timestep = torch.randint(1, self.T, (x.shape[0],))
        loss_per_pixel = get_loss(
            self.unet,
            x,
            timestep,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.dev,
            reduction="none",
        )
        loss_per_image = torch.mean(loss_per_pixel, dim=(1, 2, 3))

        # accumulate the image error depending on the timestep
        for i, t in enumerate(timestep):
            ts = int(t.item())

            self.timestep_losses[ts] = (
                self.timestep_losses[ts] * 0.95 + loss_per_image[i].item() * 0.1
            )

        loss = torch.mean(loss_per_image)

        self.writer.add_scalar("Loss", loss, self.global_step)
        self.writer.add_scalar("Loss 200", self.timestep_losses[200], self.global_step)
        self.writer.add_scalar("Loss 500", self.timestep_losses[500], self.global_step)
        self.writer.add_scalar("Loss 850", self.timestep_losses[850], self.global_step)

        if self.global_step % 200 == 0 and self.global_step > 0:
            self.unet.eval()
            samp = sample(
                self.unet,
                (16, 1, self.img_size, self.img_size),
                self.posterior_variance,
                self.sqrt_one_minus_alphas_cumprod,
                1.0 / torch.sqrt(1 - self.posterior_variance),
                self.T,
            )[-1]

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
        scheduler = SequentialLR(
            optimiser,
            schedulers=[
                LinearLR(optimiser, 0.1, 1.0, 2),
                ConstantLR(optimiser, 1.0),
                ExponentialLR(optimiser, 0.95),
            ],
            milestones=[2, 8],  # faster decrease
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

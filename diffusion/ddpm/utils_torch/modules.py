import torch
from torch import nn, Tensor
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
        attention: int = 0,
    ):
        super().__init__()
        self.use_attention = attention > 0
        self.n_heads = attention
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
            self.attention = AttentionBlock(in_ch, in_ch, self.n_heads, time_emb_dim)

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

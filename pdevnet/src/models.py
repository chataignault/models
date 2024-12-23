from torch import nn, Tensor, cat
from typing import List

from .attention_development import (
    AttentionDevelopmentConfig,
    MultiheadAttentionDevelopment,
)


class PDevBaggingClaffifier(nn.Module):
    def __init__(
        self,
        dropout: float,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        multidev_config: AttentionDevelopmentConfig,
    ):
        super().__init__()
        self.atdev = MultiheadAttentionDevelopment(
            dropout=dropout,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            multidev_config=multidev_config,
        )
        head_sizes = [g.channels * g.dim**2 for g in multidev_config.groups]
        inter_dim = sum(head_sizes)
        self.lin1 = nn.Linear(inter_dim, out_dim)

    def forward_partial(self, x: Tensor):
        sx = self.atdev(x)
        return sx

    def forward(self, x: Tensor):
        sx = self.atdev(x)
        sx_flat = [s.view(len(s), -1) for s in sx]
        sc = cat(sx_flat, axis=-1)
        y = self.lin1(sc)
        return y


class PDevBaggingClaffifierL1(nn.Module):
    """
    Include L1 regularization support to effectively learn the manifold structure
    """

    def __init__(
        self,
        dropout: float,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        multidev_config: AttentionDevelopmentConfig,
    ):
        super().__init__()
        self.atdev = MultiheadAttentionDevelopment(
            dropout=dropout,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            multidev_config=multidev_config,
        )
        head_sizes = [g.channels * g.dim**2 for g in multidev_config.groups]
        inter_dim = sum(head_sizes)
        self.lin1 = nn.Linear(inter_dim, inter_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(inter_dim, out_dim)

    def forward_partial(self, x: Tensor):
        sx = self.atdev(x)
        return sx

    def lie_algebra_coefficients(self) -> List[Tensor]:
        return [
            development.projection.A for development in self.atdev.development_layers
        ]

    def forward(self, x: Tensor):
        sx = self.atdev(x)
        sx_flat = [s.view(len(s), -1) for s in sx]
        sc = cat(sx_flat, axis=-1)
        y = self.lin1(sc)
        y = self.relu(y)
        y = self.lin2(y)
        return y


class PDevBaggingBiLSTM(nn.Module):
    def __init__(
        self,
        dropout: float,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        multidev_config: AttentionDevelopmentConfig,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        self.atdev = MultiheadAttentionDevelopment(
            dropout=dropout,
            input_dim=(1 + int(bidirectional)) * hidden_dim,
            hidden_dim=hidden_dim,
            multidev_config=multidev_config,
        )
        head_sizes = [g.channels * g.dim**2 for g in multidev_config.groups]
        inter_dim = sum(head_sizes)
        self.lin1 = nn.Linear(inter_dim, out_dim)

    def forward_partial(self, x: Tensor):
        x, _ = self.lstm(x)
        sx = self.atdev(x)
        return sx

    def lie_algebra_coefficients(self) -> List[Tensor]:
        return [
            development.projection.A for development in self.atdev.development_layers
        ]

    def forward(self, x: Tensor):
        x, _ = self.lstm(x)
        sx = self.atdev(x)
        sx_flat = [s.view(len(s), -1) for s in sx]
        sc = cat(sx_flat, axis=-1)
        y = self.lin1(sc)
        return y

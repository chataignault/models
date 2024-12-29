from pydantic import BaseModel
from typing import List, Any
from torch import nn
from torch import Tensor
from torch.nn import MultiheadAttention
from development.nn import development_layer

# from development.param import param
from development.so import so


class GroupConfig(BaseModel):
    group: Any
    dim: int
    channels: int

    # @field_validator('group')
    # def check_group(cls, v):
    #     if not issubclass(v, param):
    #         raise ValueError('group must be a param')
    #     return v

    # model_config = ConfigDict(arbitrary_types_allowed=True)


class AttentionDevelopmentConfig(BaseModel):
    n_heads: int
    groups: List[GroupConfig]


class AttentionDevelopment(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        num_heads: int,
        input_size: int,
        hidden_size: int,
        channels: int,
        # param: param,
        param: Any,
    ):
        super(AttentionDevelopment, self).__init__()
        self.attention = MultiheadAttention(
            embed_dim=embed_dim,
            dropout=dropout,
            num_heads=num_heads,
        )
        self.development = development_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            channels=channels,
            param=param,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Apply self-attention
        # MultiheadAttention expects input of shape (seq_len, batch, embed_dim)
        x = x.transpose(
            0, 1
        )  # Change from (batch, seq_len, embed_dim) to (seq_len, batch, embed_dim)

        # In self-attention, Q, K, and V are all derived from the input x
        # We don't need to explicitly create Q, K, V as MultiheadAttention does this internally
        attn_output, _ = self.attention(query=x, key=x, value=x)

        attn_output = attn_output.transpose(
            0, 1
        )  # Change back to (batch, seq_len, embed_dim)

        # Apply development layer
        x = self.development(attn_output)
        return x


# have as many groups as heads, with dimension for each
class MultiheadAttentionDevelopment(nn.Module):
    def __init__(
        self,
        dropout: float,
        input_dim: int,
        hidden_dim: int,
        multidev_config: AttentionDevelopmentConfig,
        bnorm: bool = True,
    ):
        super(MultiheadAttentionDevelopment, self).__init__()

        self.head_dim = hidden_dim // multidev_config.n_heads

        self.q = nn.Linear(input_dim, hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(input_dim, hidden_dim)

        self.attention = MultiheadAttention(
            embed_dim=hidden_dim,
            dropout=dropout,
            num_heads=multidev_config.n_heads,
            batch_first=True,
        )

        self.development_layers = nn.ModuleList(
            [
                development_layer(
                    input_size=hidden_dim // multidev_config.n_heads,
                    hidden_size=grp_config.dim,
                    channels=grp_config.channels,
                    param=grp_config.group,
                )
                for grp_config in multidev_config.groups
            ]
        )
        if bnorm:
            self.bnorm = nn.BatchNorm2d()

    def forward(self, x: Tensor) -> List[Tensor]:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # We don't need to explicitly create Q, K, V as MultiheadAttention does this internally
        x, _ = self.attention(query=q, key=k, value=v)

        # Apply development layers
        xs = []
        for i in range(len(self.development_layers)):
            xs.append(
                self.development_layers[i](
                    x[..., i * self.head_dim : (i + 1) * self.head_dim]
                )
            )
        return xs


if __name__ == "__main__":
    # TODO run some tests to instantiate and forward / backward of the model
    model = AttentionDevelopment(
        dropout=0.1,
        embed_dim=4,
        num_heads=2,
        input_size=3,
        hidden_size=2,
        channels=2,
        param=so,
    )

import jax
from jax.lax import reshape
import jax.numpy as jnp
from flax import linen as nn


class SinusoidalPositionEmbeddings(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, time: jnp.ndarray) -> jnp.ndarray:
        n = len(time)
        position = jnp.arange(n).reshape(-1, 1)

        div_term_even = jnp.exp(
            jnp.log(position)
            + jnp.arange(0, self.dim, 2) * (-jnp.log(10000.0) / self.dim)
        )
        div_term_odd = jnp.exp(
            jnp.log(position)
            + jnp.arange(1, self.dim, 2) * (-jnp.log(10000.0) / self.dim)
        )

        pe = jnp.zeros((n, self.dim))
        pe = pe.at[:, 0::2].set(jnp.sin(div_term_even))
        if self.dim > 1:
            pe = pe.at[:, 1::2].set(jnp.cos(div_term_odd))

        return pe


class UNetBlock(nn.Module):
    in_ch: int
    out_ch: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(self.out_ch, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_ch, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return x


class BlockAttention(nn.Module):
    in_ch: int
    out_ch: int
    n_heads: int

    @nn.compact
    def __call__(self, x, train: bool):
        shape = x.shape
        x = reshape(x, (shape[0], shape[1] * shape[2], shape[3]))
        q = nn.Dense(self.in_ch)(x)
        k = nn.Dense(self.in_ch)(x)
        v = nn.Dense(self.in_ch)(x)
        x = nn.MultiHeadDotProductAttention(
            self.n_heads, jnp.float32, jnp.float32, self.in_ch, self.out_ch
        )(q, k, v)
        newshape = (shape[0], shape[1], shape[2], self.out_ch)
        x = reshape(x, newshape)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x


class UNet(nn.Module):
    channels: int
    base_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        t = SinusoidalPositionEmbeddings(16)(t)
        t = nn.Dense(32)(t)
        t = nn.Dense(self.channels)(nn.relu(t))
        t = jnp.permute_dims(t[:, :, None, None], (0, 2, 3, 1))

        x = x + t

        # initial convolution
        x = nn.Conv(self.base_dim, (5, 5), padding="SAME")(x)

        # Downsampling path
        x1 = UNetBlock(self.base_dim, 2 * self.base_dim)(x, train)
        p1 = nn.max_pool(x1, (2, 2), strides=(2, 2))

        x2 = UNetBlock(2 * self.base_dim, 4 * self.base_dim)(p1, train)
        p2 = nn.max_pool(x2, (2, 2), strides=(2, 2), padding=((1, 1), (1, 1)))

        x3 = UNetBlock(4 * self.base_dim, 8 * self.base_dim)(p2, train)
        p3 = nn.max_pool(x3, (2, 2), strides=(2, 2))

        # Bottleneck
        b = UNetBlock(8 * self.base_dim, 8 * self.base_dim)(p3, train) + p3
        b = UNetBlock(8 * self.base_dim, 8 * self.base_dim)(b, train) + b

        # Upsampling path
        u3 = nn.ConvTranspose(8 * self.base_dim, (2, 2), strides=(2, 2))(b)
        c3 = jnp.concatenate([u3, x3], axis=-1)
        x3 = UNetBlock(16 * self.base_dim, 8 * self.base_dim)(c3, train)

        u2 = nn.ConvTranspose(8 * self.base_dim, (2, 2), strides=(2, 2))(x3)
        c2 = jnp.concatenate(
            [jax.image.resize(u2, x2.shape, jax.image.ResizeMethod.NEAREST), x2],
            axis=-1,
        )
        cc2 = nn.Conv(8 * self.base_dim, (1, 1))(c2)
        x2 = UNetBlock(8 * self.base_dim, 4 * self.base_dim)(cc2, train)

        u1 = nn.ConvTranspose(2 * self.base_dim, (2, 2), strides=(2, 2))(x2)
        c1 = jnp.concatenate([u1, x1], axis=-1)
        x1 = UNetBlock(4 * self.base_dim, 2 * self.base_dim)(c1, train)

        out = UNetBlock(2 * self.base_dim, self.base_dim)(x1, train)

        # out convolution
        out = nn.Conv(self.channels, (1, 1))(x)

        return out


class UNetConv(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        t = SinusoidalPositionEmbeddings(16)(t)
        t = nn.Dense(32)(t)
        t = nn.Dense(self.channels)(nn.relu(t))

        x = x + jnp.permute_dims(t[:, :, None, None], (0, 2, 3, 1))

        # Downsampling path
        x1 = UNetBlock(self.channels, 16)(x, train)
        # p1 = nn.Conv(16, (2, 2), strides=(2, 2))(x1)
        p1 = nn.max_pool(x1, (2, 2), strides=(2, 2))

        x2 = UNetBlock(16, 32)(p1, train)
        p2 = nn.max_pool(x2, (2, 2), strides=(2, 2), padding=((1, 1), (1, 1)))

        x3 = UNetBlock(32, 64)(p2, train)
        # p3 = nn.Conv(64, (4, 4), 2, 1)(x3)
        p3 = nn.max_pool(x3, (2, 2), strides=(2, 2))

        # Bottleneck
        b = UNetBlock(64, 64)(p3, train)

        # Upsampling path
        u3 = nn.ConvTranspose(64, (2, 2), strides=(2, 2))(b)
        c3 = jnp.concatenate([u3, x3], axis=-1)
        x3 = UNetBlock(128, 64)(c3, train)

        u2 = nn.ConvTranspose(32, (2, 2), strides=(2, 2))(x3)
        c2 = jnp.concatenate(
            [jax.image.resize(u2, x2.shape, jax.image.ResizeMethod.NEAREST), x2],
            axis=-1,
        )
        x2 = UNetBlock(64, 32)(c2, train)

        u1 = nn.ConvTranspose(16, (2, 2), strides=(2, 2))(x2)
        c1 = jnp.concatenate([u1, x1], axis=-1)
        x1 = UNetBlock(32, 16)(c1, train)

        out = UNetBlock(16, self.channels)(x1, train)

        return out


class UNetAttention(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        t = SinusoidalPositionEmbeddings(16)(t)
        t = nn.Dense(32)(t)
        t = nn.Dense(self.channels)(nn.relu(t))

        x = x + jnp.permute_dims(t[:, :, None, None], (0, 2, 3, 1))

        # Downsampling path
        x1 = BlockAttention(self.channels, 16, 1)(x, train)
        # p1 = nn.max_pool(x1, (2, 2), strides=(2, 2))
        p1 = nn.Conv(16, (2, 2), strides=(2, 2))(x1)

        x2 = BlockAttention(16, 32, 2)(p1, train)
        p2 = nn.max_pool(x2, (2, 2), strides=(2, 2), padding=((1, 1), (1, 1)))

        x3 = BlockAttention(32, 64, 4)(p2, train)
        p3 = nn.max_pool(x3, (2, 2), strides=(2, 2))

        # Bottleneck
        b = BlockAttention(64, 128, 8)(p3, train)

        # Upsampling path
        u3 = nn.ConvTranspose(64, (2, 2), strides=(2, 2))(b)
        c3 = jnp.concatenate([u3, x3], axis=-1)
        x3 = BlockAttention(128, 64, 4)(c3, train)

        u2 = nn.ConvTranspose(32, (2, 2), strides=(2, 2))(x3)
        c2 = jnp.concatenate(
            [jax.image.resize(u2, x2.shape, jax.image.ResizeMethod.NEAREST), x2],
            axis=-1,
        )
        x2 = BlockAttention(64, 32, 2)(c2, train)

        u1 = nn.ConvTranspose(16, (2, 2), strides=(2, 2))(x2)
        c1 = jnp.concatenate([u1, x1], axis=-1)
        x1 = BlockAttention(32, 16, 1)(c1, train)

        out = nn.Conv(self.channels, kernel_size=1)(x1)

        return out


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    in_ch: int
    time_emb_dim: int
    num_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        h = x

        # Time embedding
        t_emb = nn.silu(nn.Dense(self.in_ch)(t))
        t_emb = jnp.reshape(t_emb, (t_emb.shape[0], 1, 1, self.in_ch))

        # First conv block
        x = nn.Conv(self.in_ch, (3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=min(self.num_groups, self.in_ch))(x)
        x = nn.silu(x)

        # Add time embedding
        x = x + t_emb

        # Second conv block
        x = nn.Conv(self.in_ch, (3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=min(self.num_groups, self.in_ch))(x)
        x = nn.silu(x)

        return x + h


class AttentionBlockSimple(nn.Module):
    """Multi-head attention block with time embedding."""

    in_ch: int
    hidden_dim: int
    n_heads: int
    time_emb_dim: int
    num_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        # Store residual
        residual = x

        # Time embedding
        t_emb = nn.silu(nn.Dense(self.in_ch)(t))
        t_emb = jnp.reshape(t_emb, (t_emb.shape[0], 1, 1, self.in_ch))

        # Group norm
        x = nn.GroupNorm(num_groups=min(self.num_groups, self.in_ch))(x)
        x = x + t_emb

        # Reshape for attention: (N, H, W, C) -> (N, H*W, C)
        N, H, W, C = x.shape
        x_reshaped = jnp.reshape(x, (N, H * W, C))

        # Q, K, V projections
        query = nn.Dense(self.hidden_dim)(x_reshaped)
        key = nn.Dense(self.hidden_dim)(x_reshaped)
        value = nn.Dense(self.hidden_dim)(x_reshaped)

        # Multi-head attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.hidden_dim
        )(query, key, value)

        # Project back to input channels
        x_attn = nn.Dense(self.in_ch)(attn_output)

        # Reshape back to spatial dimensions
        x_attn = jnp.reshape(x_attn, (N, H, W, C))

        # Add residual connection
        return x_attn + residual


class InterBlock(nn.Module):
    """Intermediate block for downsampling or upsampling."""

    in_ch: int
    out_ch: int
    time_emb_dim: int
    up: bool = False
    num_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        h = x

        # Handle upsampling path differently
        if self.up:
            # For upsampling, input has 2*in_ch from concatenation
            x = nn.Conv(2 * self.in_ch, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(self.num_groups, 2 * self.in_ch))(x)
            x = nn.silu(x)

            # Project time embedding to match conv output (2*in_ch)
            t_emb = nn.silu(nn.Dense(2 * self.in_ch)(t))
            t_emb = jnp.reshape(t_emb, (t_emb.shape[0], 1, 1, 2 * self.in_ch))

            # Add time embedding
            x = x + t_emb

            # Second conv
            x = nn.Conv(self.in_ch, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(self.num_groups, self.in_ch))(x)
            x = nn.silu(x)

            # Residual connection with squish conv (2*in_ch -> in_ch)
            h = nn.Conv(self.in_ch, (3, 3), padding="SAME")(h)
            x = x + h

            # Upsample transform
            x_out = nn.ConvTranspose(
                self.out_ch, (4, 4), strides=(2, 2), padding="SAME"
            )(x)
        else:
            # First conv
            x = nn.Conv(self.in_ch, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(self.num_groups, self.in_ch))(x)
            x = nn.silu(x)

            # Project time embedding to match conv output (in_ch)
            t_emb = nn.silu(nn.Dense(self.in_ch)(t))
            t_emb = jnp.reshape(t_emb, (t_emb.shape[0], 1, 1, self.in_ch))

            # Add time embedding
            x = x + t_emb

            # Second conv
            x = nn.Conv(self.in_ch, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(self.num_groups, self.in_ch))(x)
            x = nn.silu(x)

            # Residual connection
            x = x + h

            # Downsample transform
            x_out = nn.Conv(self.out_ch, (4, 4), strides=(2, 2), padding="SAME")(x)

        return x_out, x


class SimpleUnet(nn.Module):
    """A simplified variant of the UNet architecture in JAX/Flax."""

    time_emb_dim: int = 8
    downs: tuple = (16, 32, 64, 128)
    channels: int = 1
    num_groups: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        # Store input shape for final output
        input_shape = x.shape
        ups = self.downs[::-1]

        # Time embedding
        t_emb = SinusoidalPositionEmbeddings(self.time_emb_dim)(t)
        t_emb = nn.Dense(4 * self.time_emb_dim)(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(4 * self.time_emb_dim)(t_emb)

        # Initial convolutions
        x = nn.Conv(self.downs[0] // 2, (3, 3), padding="SAME")(x)
        x = ResBlock(self.downs[0] // 2, 4 * self.time_emb_dim)(x, t_emb, train)
        x = nn.Conv(self.downs[0], (3, 3), padding="SAME")(x)

        # Store residuals for skip connections
        x_down = [x]

        # Downsampling path
        for i in range(len(self.downs) - 1):
            x, h = InterBlock(
                self.downs[i], self.downs[i + 1], 4 * self.time_emb_dim, up=False
            )(x, t_emb, train)
            x_down.append(h)

        x_down.append(x)

        # Bottleneck
        h = x
        x = ResBlock(self.downs[-1], 4 * self.time_emb_dim)(x, t_emb, train)
        x = AttentionBlockSimple(self.downs[-1], 128, 8, 4 * self.time_emb_dim)(
            x, t_emb, train
        )
        x = nn.GroupNorm(num_groups=min(self.num_groups, self.downs[-1]))(x)
        x = nn.silu(x)
        x = ResBlock(self.downs[-1], 4 * self.time_emb_dim)(x, t_emb, train)
        x = x + h

        # Upsampling path
        for k in range(len(ups) - 1):
            residual = x_down[-(k + 1)]
            # Resize residual to match x shape if needed (for odd dimensions)
            if residual.shape[1:3] != x.shape[1:3]:
                residual = jax.image.resize(
                    residual, x.shape, method=jax.image.ResizeMethod.NEAREST
                )
            x_extended = jnp.concatenate([x, residual], axis=-1)
            x, _ = InterBlock(ups[k], ups[k + 1], 4 * self.time_emb_dim, up=True)(
                x_extended, t_emb, train
            )

        # Add the ultimate residual from the initial convolution
        residual_initial = x_down[0]
        if residual_initial.shape[1:3] != x.shape[1:3]:
            residual_initial = jax.image.resize(
                residual_initial, x.shape, method=jax.image.ResizeMethod.NEAREST
            )
        x = x + residual_initial
        # Removed final GroupNorm and ReLU for better output
        x = nn.Conv(self.channels, (1, 1))(x)

        # Ensure output matches input spatial dimensions
        if x.shape[1:3] != input_shape[1:3]:
            x = jax.image.resize(x, input_shape, method=jax.image.ResizeMethod.NEAREST)

        return x

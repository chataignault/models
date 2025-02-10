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
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Conv(self.out_ch, (3, 3), padding="SAME")(x)
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
    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        shape = x.shape
        t = SinusoidalPositionEmbeddings(shape[1] * shape[2])(t)
        t = t.reshape((shape[0], shape[1], shape[2], 1))
        x = x + t

        # Downsampling path
        x1 = UNetBlock(1, 16)(x, train)
        p1 = nn.max_pool(x1, (2, 2), strides=(2, 2))

        x2 = UNetBlock(16, 32)(p1, train)
        p2 = nn.max_pool(x2, (2, 2), strides=(2, 2), padding=((1, 1), (1, 1)))

        x3 = UNetBlock(32, 64)(p2, train)
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

        out = UNetBlock(16, 1)(x1, train)

        return out


class UNetConv(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        shape = x.shape
        t = SinusoidalPositionEmbeddings(shape[1] * shape[2])(t)
        t = t.reshape((shape[0], shape[1], shape[2], 1))
        x = x + t
        # Downsampling path
        x1 = UNetBlock(1, 16)(x, train)
        p1 = nn.Conv(16, (4, 4), 2, 1)(x1)

        x2 = UNetBlock(16, 32)(p1, train)
        p2 = nn.max_pool(x2, (2, 2), strides=(2, 2), padding=((1, 1), (1, 1)))

        x3 = UNetBlock(32, 64)(p2, train)
        p3 = nn.Conv(64, (4, 4), 2, 1)(x3)

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

        out = UNetBlock(16, 1)(x1, train)

        return out


class UNetAttention(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool):
        shape = x.shape
        t = SinusoidalPositionEmbeddings(shape[1] * shape[2])(t)
        t = t.reshape((shape[0], shape[1], shape[2], 1))
        x = x + t
        # Downsampling path
        x1 = BlockAttention(1, 16, 1)(x, train)
        p1 = nn.max_pool(x1, (2, 2), strides=(2, 2))

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

        out = nn.Conv(1, 1)(x1)

        return out


if __name__ == "__main__":
    unet = UNet()

    print(unet)

import jax
import jax.numpy as jnp
from flax import linen as nn


class UNetBlock(nn.Module):
    in_ch: int
    out_ch: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_ch, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_ch, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        return x


class UNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Downsampling path
        x1 = UNetBlock(1, 16)(x)
        p1 = nn.max_pool(x1, (2, 2), strides=(2, 2))

        x2 = UNetBlock(16, 32)(p1)
        p2 = nn.max_pool(x2, (2, 2), strides=(2, 2), padding=((1, 1), (1, 1)))

        x3 = UNetBlock(32, 64)(p2)
        p3 = nn.max_pool(x3, (2, 2), strides=(2, 2))
        # x4 = UNetBlock(64, 128)(p3)
        # p4 = nn.max_pool(x4, (2, 2), strides=(2, 2))

        # Bottleneck
        b = UNetBlock(64, 128)(p3)

        # Upsampling path
        # u4 = nn.ConvTranspose(128, (2, 2), strides=(2, 2))(b)
        # c4 = jnp.concatenate([u4, x4], axis=-1)
        # x4 = UNetBlock(256, 128)(c4)

        u3 = nn.ConvTranspose(64, (2, 2), strides=(2, 2))(b)
        c3 = jnp.concatenate([u3, x3], axis=-1)
        x3 = UNetBlock(128, 64)(c3)

        u2 = nn.ConvTranspose(32, (2, 2), strides=(2, 2))(x3)
        c2 = jnp.concatenate(
            [jax.image.resize(u2, x2.shape, jax.image.ResizeMethod.NEAREST), x2], axis=-1
        )
        x2 = UNetBlock(64, 32)(c2)

        u1 = nn.ConvTranspose(16, (2, 2), strides=(2, 2))(x2)
        c1 = jnp.concatenate([u1, x1], axis=-1)
        x1 = UNetBlock(32, 16)(c1)

        out = UNetBlock(16, 1)(c1)

        return out


if __name__ == "__main__":
    unet = UNet()

    print(unet)

import pytest
import jax.numpy as jnp
from jax import random

from ..classes import UNet, UNetConv, UNetAttention


@pytest.mark.parametrize("b, c, d", [(64, 1, 28), (128, 3, 32), (256, 5, 28)])
def test_unet_forward(b: int, c: int, d: int):
    s = (b, d, d, c)
    rng = random.PRNGKey(0)
    rng_init, rng_apply = random.split(rng)

    unet = UNet(channels=c)
    t = jnp.zeros(b)
    x = jnp.zeros(s)

    # z = unet(x, t)

    variables = unet.init(rng_init, jnp.ones(s), jnp.ones([1]), train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    z = unet.apply(
        {"params": params, "batch_stats": batch_stats},
        x,
        t,
        rngs=rng_apply,
        train=False,
    )

    assert z.shape == s


# @pytest.mark.parametrize("b, c, d", [(64, 1, 28), (128, 3, 32), (256, 5, 28)])
# def test_unet_conv_forward(b: int, c: int, d: int):
#     s = (b, c, d, d)
#     rng = random.PRNGKey(0)

#     unet = UNetConv(channels=c)

#     t = jnp.zeros(b)
#     x = jnp.zeros(s)

#     z = unet(x, t)
#     assert z.shape == s

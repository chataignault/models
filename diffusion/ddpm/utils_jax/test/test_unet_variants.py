import pytest
import jax.numpy as jnp
from jax import random
import optax

from ..classes import UNet, UNetConv, UNetAttention
from ..diffusion import sample
from ..training import TrainState


@pytest.mark.parametrize("b, c, d", [(64, 1, 28), (128, 3, 28), (256, 5, 28)])
def test_unet_forward(b: int, c: int, d: int):
    s = (b, d, d, c)
    rng = random.PRNGKey(0)
    rng_init, rng_apply = random.split(rng)

    unet = UNet(channels=c)
    t = jnp.zeros(b)
    x = jnp.zeros(s)

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


@pytest.mark.parametrize("b, c, d", [(64, 1, 28), (128, 3, 28), (256, 5, 28)])
def test_unet_conv_forward(b: int, c: int, d: int):
    s = (b, d, d, c)
    rng = random.PRNGKey(0)
    rng_init, rng_apply = random.split(rng)

    unet = UNetConv(channels=c)

    t = jnp.zeros(b)
    x = jnp.zeros(s)

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


@pytest.mark.parametrize("b, c, d", [(64, 1, 28), (128, 3, 28), (256, 5, 28)])
def test_unet_att_forward(b: int, c: int, d: int):
    s = (b, d, d, c)
    rng = random.PRNGKey(0)
    rng_init, rng_apply = random.split(rng)

    unet = UNetAttention(channels=c)

    t = jnp.zeros(b)
    x = jnp.zeros(s)

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


@pytest.mark.parametrize("b, c, d", [(16, 1, 28), (32, 3, 28)])
def test_simple_unet_sampling(b: int, c: int, d: int):
    s = (b, d, d, c)
    rng = random.PRNGKey(0)
    rng_init, rng_sample = random.split(rng)
    unet = UNet(channels=c)
    T = 100  # faster sampling
    t = jnp.ones(T) * 0.1

    variables = unet.init(rng_init, jnp.ones(s), jnp.ones([1]), train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    tx = optax.adam(1e-3)
    state = TrainState.create(
        apply_fn=unet.apply, params=params, tx=tx, batch_stats=batch_stats
    )

    x = sample(state, s, rng_sample, T, t, t, t, t)

    assert len(x) == 1

    x = x[0]

    assert x.shape == s

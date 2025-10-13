from jax import numpy as jnp
from jax import random
from flax import linen as nn


class ScaledDotProductAttentionBlock(nn.Module):
    h_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        q = nn.Dense(self.h_dim, name="dense_query")(x)
        k = nn.Dense(self.h_dim, name="dense_key")(x)
        v = nn.Dense(self.out_dim, name="dense_value")(x)

        qk = jnp.einsum("NLH,NlH->NLl", q, k)

        s = softmax(qk / jnp.sqrt(qk.shape[-1]))

        return s @ v


def softmax(x: jnp.array) -> jnp.array:
    return jnp.divide(jnp.exp(x), jnp.sum(jnp.exp(x), axis=-1)[:, :, jnp.newaxis])


def main():
    print("Test attention block :")

    x = jnp.ones((100, 1000, 50))
    rng = random.key(0)

    print(f"inpute shape : {x.shape}")

    att = ScaledDotProductAttentionBlock(30, 30)
    variables = att.init(rng, x)

    print(att)

    y = att.apply(variables, x)

    print(y.shape)

    print("Forward pass successful.")


if __name__ == "__main__":
    main()

from jax import numpy as jnp
from flax import nn, compact

class SelfAttentionBlock(nn.Module):

    @compact
    def forward(x:jnp.array)->jnp.array:
        q = nn.Dense()(x)
        k = nn.Dense()(x)
        v = nn.Dense()(x)

        return softmax(q @ k) @ v


def softmax(x:jnp.array)->jnp.array:
    return x


def main():

    print("Test attention block :")

    x = jnp.ones((100, 1000, 50))

    print(f"inpute shape : {x.shape}")
    
    att = SelfAttentionBlock()

    y = att(x)

    print("Forward pass successful.")

if __name__ == "__main__":
    main()

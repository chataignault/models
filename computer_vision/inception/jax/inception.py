import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


class InceptionBlock(nn.Module):
    out_1x1: int
    reduce_3x3: int
    out_3x3: int
    reduce_5x5: int
    out_5x5: int
    pool_proj: int
    
    @nn.compact
    def __call__(self, x):
        branch1 = nn.Conv(self.out_1x1, kernel_size=(1, 1))(x)
        branch1 = nn.relu(branch1)
        
        branch2 = nn.Conv(self.reduce_3x3, kernel_size=(1, 1))(x)
        branch2 = nn.relu(branch2)
        branch2 = nn.Conv(self.out_3x3, kernel_size=(3, 3), padding=1)(branch2)
        branch2 = nn.relu(branch2)
        
        branch3 = nn.Conv(self.reduce_5x5, kernel_size=(1, 1))(x)
        branch3 = nn.relu(branch3)
        branch3 = nn.Conv(self.out_5x5, kernel_size=(5, 5), padding=2)(branch3)
        branch3 = nn.relu(branch3)
        
        branch4 = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        branch4 = nn.Conv(self.pool_proj, kernel_size=(1, 1))(branch4)
        branch4 = nn.relu(branch4)
        
        return jnp.concatenate([branch1, branch2, branch3, branch4], axis=-1)


class SimpleInceptionNet(nn.Module):
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(64, kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = InceptionBlock(16, 24, 32, 4, 8, 8)(x)
        x = InceptionBlock(32, 48, 64, 8, 16, 16)(x)
        
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dropout(rate=0.4, deterministic=not training)(x)
        x = nn.Dense(self.num_classes)(x)
        
        return x
    
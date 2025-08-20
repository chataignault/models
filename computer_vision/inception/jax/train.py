import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from flax.core import freeze
import flax.serialization
import time
from typing import Tuple, Any


class TrainState(train_state.TrainState):
    pass


def create_train_state(model: nn.Module, rng: jax.random.PRNGKey, learning_rate: float, 
                      input_shape: Tuple[int, ...]) -> TrainState:
    params = model.init(rng, jnp.ones(input_shape))['params']
    
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=5 * 782,  # 5 epochs * ~782 steps per epoch (50000/64)
        decay_rate=0.5
    )
    
    tx = optax.chain(
        optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    )
    
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state: TrainState, batch, rng: jax.random.PRNGKey):
    images, labels = batch
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images, training=True, rngs={'dropout': rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, accuracy


@jax.jit
def eval_step(state: TrainState, batch):
    images, labels = batch
    logits = state.apply_fn({'params': state.params}, images, training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, accuracy


def train_model(model: nn.Module, train_loader, test_loader, num_epochs: int = 20, 
               learning_rate: float = 0.001) -> Tuple[list, list]:
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(model, init_rng, learning_rate, (1, 32, 32, 3))
    
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_steps = 0
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= 782:  # Approximate steps per epoch for CIFAR-10
                break
                
            batch_data = (images, labels)
            
            rng, step_rng = jax.random.split(rng)
            state, loss, accuracy = train_step(state, batch_data, step_rng)
            
            train_loss_sum += loss
            train_acc_sum += accuracy
            train_steps += 1
            
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss:.3f}')
        
        train_acc = (train_acc_sum / train_steps) * 100
        train_accuracies.append(train_acc)
        
        test_acc = evaluate_model(state, test_loader)
        test_accuracies.append(test_acc)
        
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}% - Time: {epoch_time:.2f}s')
    
    return train_accuracies, test_accuracies, state


def evaluate_model(state: TrainState, test_loader) -> float:
    test_loss_sum = 0.0
    test_acc_sum = 0.0
    test_steps = 0
    
    for i, (images, labels) in enumerate(test_loader):
        if i >= 157:  # Approximate steps for test set (10000/64)
            break
            
        batch_data = (images, labels)
        
        loss, accuracy = eval_step(state, batch_data)
        test_loss_sum += loss
        test_acc_sum += accuracy
        test_steps += 1
    
    return (test_acc_sum / test_steps) * 100


def save_model(state: TrainState, filepath: str):
    with open(filepath, 'wb') as f:
        f.write(flax.serialization.to_bytes(state.params))
    print(f'Model saved to {filepath}')


def load_model(model: nn.Module, filepath: str, input_shape: Tuple[int, ...]) -> TrainState:
    rng = jax.random.PRNGKey(42)
    state = create_train_state(model, rng, 0.001, input_shape)
    
    with open(filepath, 'rb') as f:
        params = flax.serialization.from_bytes(state.params, f.read())
    
    state = state.replace(params=params)
    print(f'Model loaded from {filepath}')
    return state
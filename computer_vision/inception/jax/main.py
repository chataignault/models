import jax
import jax.numpy as jnp
from inception import SimpleInceptionNet
from dataset import get_cifar10_dataloaders, get_classes
from train import train_model, evaluate_model, save_model, create_train_state


def main():
    print(f'JAX devices: {jax.devices()}')
    print(f'Default backend: {jax.default_backend()}')
    
    model = SimpleInceptionNet(num_classes=10)
    rng = jax.random.PRNGKey(42)
    
    dummy_input = jnp.ones((1, 32, 32, 3))
    params = model.init(rng, dummy_input)
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f'Model created with {param_count:,} parameters')
    
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=64)
    
    print('Starting training...')
    train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, 
        num_epochs=20, learning_rate=0.001
    )
    
    state = create_train_state(model, rng, 0.001, (1, 32, 32, 3))
    final_test_acc = evaluate_model(state, test_loader)
    print(f'\nFinal test accuracy: {final_test_acc:.2f}%')
    
    save_model(state, 'inception_cifar10_jax.pkl')
    
    classes = get_classes()
    print(f'\nCIFAR-10 classes: {classes}')


if __name__ == "__main__":
    main()

# Inception CIFAR-10 Classifier (Rust/Burn)

This is a Rust implementation of an Inception-based neural network for CIFAR-10 classification using the Burn deep learning framework. 
This project is transcoded from the PyTorch implementation in the `../torch` folder.

## Architecture

The model consists of:
- Initial convolutional layer (3 → 64 channels)
- Two Inception blocks with different channel configurations
- Adaptive average pooling and dropout
- Final linear classifier

### Inception Block Architecture

Each Inception block contains four parallel branches:
1. **Branch 1**: 1×1 convolution
2. **Branch 2**: 1×1 convolution → 3×3 convolution  
3. **Branch 3**: 1×1 convolution → 5×5 convolution
4. **Branch 4**: 3×3 max pooling → 1×1 convolution

The outputs are concatenated along the channel dimension.

## Requirements

- Rust (latest stable)
- Cargo

## Usage

### Training

```bash
cargo run --release
```

This will:
1. Load the CIFAR-10 dataset
2. Train the model for 20 epochs
3. Save the trained model to `inception_cifar10.mpk`

### Configuration

The training configuration can be modified in `src/main.rs`:

```rust
let config = TrainingConfig {
    num_epochs: 20,
    learning_rate: 0.001,
    train_batch_size: 64,
    // ... other parameters
};
```

## Dataset

The implementation expects CIFAR-10 data to be available. 
The Burn framework will automatically download and prepare the dataset if not present.

## Files

- `src/model.rs` - Inception block and network architecture
- `src/dataset.rs` - CIFAR-10 data loading and preprocessing  
- `src/training.rs` - Training loop and configuration
- `src/main.rs` - Entry point

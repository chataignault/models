use burn::backend::Autodiff;
use burn_cuda::{Cuda, CudaDevice};
use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, Optimizer, GradientsParams},
};

mod model;
mod dataset;

use model::{SimpleInceptionNet, ModelConfig};
use dataset::{get_cifar10_dataloaders, Cifar10Config, get_cifar10_classes};

type Backend = Cuda<f32>;
type AutodiffBackend = Autodiff<Backend>;

fn main() {
    // Initialize the CUDA device
    let device = CudaDevice::default();
    
    // Create model configuration
    let model_config = ModelConfig::default();
    
    // Create the model with autodiff backend for training
    let mut model = SimpleInceptionNet::<AutodiffBackend>::new(&device, model_config.num_classes);
    
    println!("Model created successfully!");
    println!("Model config: {:?}", model_config);
    println!("Device: {:?}", device);
    
    // Training configuration (increased for GPU)
    let learning_rate = 0.001;
    let num_epochs = 20; // Restored for proper training
    let batch_size = 32; // Increased batch size for GPU
    
    // Initialize optimizer and loss function
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    
    // Load CIFAR-10 dataset
    let dataset_config = Cifar10Config {
        train_batch_size: batch_size,
        test_batch_size: batch_size,
        num_workers: 4,
    };
    
    let (_train_dataloader, test_dataloader) = match get_cifar10_dataloaders(&device, &dataset_config) {
        Ok(dataloaders) => dataloaders,
        Err(e) => {
            eprintln!("Failed to load CIFAR-10 dataset: {}", e);
            eprintln!("Make sure the CIFAR-10 data is available in data/cifar-10-batches-py/");
            return;
        }
    };
    
    let classes = get_cifar10_classes();
    println!("CIFAR-10 classes: {:?}", classes);
    
    println!("Starting training for {} epochs...", num_epochs);
    
    // Training loop
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Need to recreate dataloaders each epoch
        let (epoch_train_dataloader, _) = match get_cifar10_dataloaders(&device, &dataset_config) {
            Ok(dataloaders) => dataloaders,
            Err(e) => {
                eprintln!("Failed to recreate CIFAR-10 dataloader for epoch {}: {}", epoch + 1, e);
                return;
            }
        };
        
        for batch in epoch_train_dataloader.iter() {
            let images = batch.image;
            let labels = batch.label;
            
            // Forward pass
            let output = model.forward(images);
            let loss = loss_fn.forward(output, labels);
            
            // Store loss value before backward pass
            let loss_value = loss.clone().into_scalar();
            
            // Backward pass
            let grads = loss.backward();
            let params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, params);
            
            total_loss += loss_value;
            num_batches += 1;
            
            if num_batches % 100 == 0 { // Print every 100 batches
                println!("Epoch {}/{}, Batch {}, Loss: {:.4}", 
                        epoch + 1, num_epochs, num_batches, loss_value);
            }
        }
        
        let avg_loss = total_loss / num_batches as f32;
        println!("✅ Epoch {}/{} completed - Average Loss: {:.4} ({} batches)", 
                epoch + 1, num_epochs, avg_loss, num_batches);
    }
    
    println!("Training completed!");
    
    // Test the trained model on real test data
    println!("Testing trained model on CIFAR-10 test data...");
    let mut correct = 0;
    let mut total = 0;
    
    for batch in test_dataloader.iter() {
        let images = batch.image;
        let targets = batch.label;
        
        let output = model.forward(images);
        let predictions = output.argmax(1);
        
        // Convert to CPU for comparison
        let targets_data = targets.to_data();
        let predictions_data = predictions.to_data();
        
        // Count correct predictions
        for (pred, target) in predictions_data.iter::<i64>().zip(targets_data.iter::<i64>()) {
            if pred == target {
                correct += 1;
            }
            total += 1;
        }
    }
    
    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("Test Accuracy: {:.2}% ({}/{} correct)", accuracy, correct, total);
    println!("Model testing completed successfully!");
}
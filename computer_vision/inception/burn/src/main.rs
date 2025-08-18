use burn::backend::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, Optimizer, GradientsParams},
    tensor::Tensor,
};

mod model;

use model::{SimpleInceptionNet, ModelConfig};

type Backend = NdArray<f32>;
type AutodiffBackend = Autodiff<Backend>;

fn main() {
    // Initialize the device
    let device = NdArrayDevice::default();
    
    // Create model configuration
    let model_config = ModelConfig::default();
    
    // Create the model with autodiff backend for training
    let mut model = SimpleInceptionNet::<AutodiffBackend>::new(&device, model_config.num_classes);
    
    println!("Model created successfully!");
    println!("Model config: {:?}", model_config);
    println!("Device: {:?}", device);
    
    // Training configuration
    let learning_rate = 0.001;
    let num_epochs = 2; // Reduced for faster testing
    let batch_size = 4; // Reduced batch size
    let height = 32;
    let width = 32;
    let channels = 3;
    
    // Initialize optimizer and loss function
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    
    println!("Starting training for {} epochs...", num_epochs);
    
    // Training loop
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let num_batches = 5; // Reduced number of batches per epoch for faster testing
        
        for batch_idx in 0..num_batches {
            // Generate random training data
            let input_data: Vec<f32> = (0..(batch_size * channels * height * width))
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect();
            
            let input_tensor = Tensor::<AutodiffBackend, 1>::from_floats(
                input_data.as_slice(), &device
            ).reshape([batch_size, channels, height, width]);
            
            // Generate random labels (0-9 for CIFAR-10)
            let label_data: Vec<i64> = (0..batch_size)
                .map(|_| (rand::random::<u32>() % 10) as i64)
                .collect();
            
            let labels = Tensor::<AutodiffBackend, 1, burn::tensor::Int>::from_ints(
                label_data.as_slice(), &device
            );
            
            // Debug: Print tensor shapes and label values for first batch
            if epoch == 0 && batch_idx == 0 {
                println!("Debug - Input shape: {:?}", input_tensor.dims());
                println!("Debug - Label values: {:?}", label_data);
                println!("Debug - Label tensor shape: {:?}", labels.dims());
            }
            
            // Forward pass
            let output = model.forward(input_tensor);
            
            // Debug: Print output shape for first batch
            if epoch == 0 && batch_idx == 0 {
                println!("Debug - Output shape: {:?}", output.dims());
            }
            
            let loss = loss_fn.forward(output, labels);
            
            // Store loss value before backward pass
            let loss_value = loss.clone().into_scalar();
            
            // Backward pass
            let grads = loss.backward();
            let params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, params);
            
            total_loss += loss_value;
            
            if batch_idx % 2 == 0 { // Print more frequently with fewer batches
                println!("Epoch {}/{}, Batch {}/{}, Loss: {:.4}", 
                        epoch + 1, num_epochs, batch_idx + 1, num_batches, loss_value);
            }
        }
        
        let avg_loss = total_loss / num_batches as f32;
        println!("Epoch {}/{} completed - Average Loss: {:.4}", epoch + 1, num_epochs, avg_loss);
    }
    
    println!("Training completed!");
    
    // Test the trained model
    println!("Testing trained model...");
    let test_input_data: Vec<f32> = (0..(4 * channels * height * width))
        .map(|_| rand::random::<f32>() * 2.0 - 1.0)
        .collect();
    
    let test_input = Tensor::<AutodiffBackend, 1>::from_floats(
        test_input_data.as_slice(), &device
    ).reshape([4, channels, height, width]);
    
    let test_output = model.forward(test_input);
    println!("Test output shape: {:?}", test_output.dims());
    println!("Model testing completed successfully!");
}
use burn::backend::Autodiff;
use burn_wgpu::{Wgpu, WgpuDevice};

mod model;

use model::{SimpleInceptionNet, ModelConfig};

type Backend = Wgpu<f32, i32>;
type AutodiffBackend = Autodiff<Backend>;

fn main() {
    // Initialize the device
    let device = WgpuDevice::default();
    
    // Create model configuration
    let model_config = ModelConfig::default();
    
    // Create the model
    let model = SimpleInceptionNet::<Backend>::new(&device, model_config.num_classes);
    
    println!("✅ Model created successfully!");
    println!("Model config: {:?}", model_config);
    println!("Device: {:?}", device);
    
    // Test forward pass with random data
    use burn::tensor::Tensor;
    
    let batch_size = 4;
    let height = 32;
    let width = 32;
    let channels = 3;
    
    // Create random input data
    let input_data: Vec<f32> = (0..(batch_size * channels * height * width))
        .map(|_| rand::random::<f32>() * 2.0 - 1.0) // Random values between -1 and 1
        .collect();
    
    let input_tensor = Tensor::<Backend, 1>::from_floats(
        input_data.as_slice(), &device
    ).reshape([batch_size, channels, height, width]);
    
    println!("Input tensor shape: {:?}", input_tensor.dims());
    
    // Run forward pass
    let output = model.forward(input_tensor);
    
    println!("Output tensor shape: {:?}", output.dims());
    println!("✅ Forward pass completed successfully!");
    
    // Print some sample outputs
    let output_data = output.to_data();
    println!("Output shape: {:?}", output_data.shape);
    println!("Output dtype: {:?}", output_data.dtype);
}
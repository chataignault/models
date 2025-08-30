use burn::{
    config::Config,
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    tensor::{backend::Backend, Tensor},
};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Cifar10Batch<B: Backend> {
    pub image: Tensor<B, 4>,
    pub label: Tensor<B, 1, burn::tensor::Int>,
}

#[derive(Clone)]
pub struct Cifar10Batcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Cifar10Batcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct Cifar10Item {
    pub image: Vec<f32>, // 3072 elements (32x32x3)
    pub label: usize,
}

impl<B: Backend> Batcher<B, Cifar10Item, Cifar10Batch<B>> for Cifar10Batcher<B> {
    fn batch(&self, items: Vec<Cifar10Item>, _device: &B::Device) -> Cifar10Batch<B> {
        let batch_size = items.len();
        
        // Collect image data
        let mut image_data = Vec::with_capacity(batch_size * 3072);
        let mut label_data = Vec::with_capacity(batch_size);
        
        for item in items {
            image_data.extend_from_slice(&item.image);
            label_data.push(item.label as i64);
        }
        
        let images = Tensor::<B, 1>::from_floats(image_data.as_slice(), &self.device)
            .reshape([batch_size, 3, 32, 32]);
        
        let labels = Tensor::<B, 1, burn::tensor::Int>::from_ints(
            label_data.as_slice(), &self.device
        );

        Cifar10Batch {
            image: images,
            label: labels,
        }
    }
}

#[derive(Config, Debug)]
pub struct Cifar10Config {
    pub train_batch_size: usize,
    pub test_batch_size: usize,
    pub num_workers: usize,
}

impl Default for Cifar10Config {
    fn default() -> Self {
        Self {
            train_batch_size: 32,
            test_batch_size: 32,
            num_workers: 4,
        }
    }
}

fn load_cifar10_batch_python(file_path: &str) -> Result<Vec<Cifar10Item>, Box<dyn std::error::Error>> {
    println!("Loading CIFAR-10 batch from: {}", file_path);
    
    // Use Python to unpickle the file and convert to JSON
    let output = std::process::Command::new("python")
        .arg("-c")
        .arg(&format!(
            r#"
import pickle
import json
import sys
import numpy as np

try:
    with open('{}', 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    # Convert keys from bytes to strings and handle all byte values
    data = {{}}
    for key, value in batch.items():
        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
        if key_str == 'data':
            # Convert data to list and normalize to 0-1 range
            data[key_str] = (np.array(value).astype(np.float32) / 255.0).tolist()
        elif key_str == 'labels':
            data[key_str] = value if isinstance(value, list) else value.tolist()
        elif key_str == 'filenames':
            # Handle list of byte strings
            data[key_str] = [f.decode('utf-8') if isinstance(f, bytes) else f for f in value]
        elif key_str == 'batch_label':
            data[key_str] = value.decode('utf-8') if isinstance(value, bytes) else value
        else:
            # Handle any other byte values
            if isinstance(value, bytes):
                data[key_str] = value.decode('utf-8', errors='ignore')
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], bytes):
                data[key_str] = [v.decode('utf-8', errors='ignore') for v in value]
            else:
                data[key_str] = value
    
    print(json.dumps(data))
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"#, file_path
        ))
        .output()?;
    
    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python script failed: {}", error_msg).into());
    }
    
    let json_str = String::from_utf8(output.stdout)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    
    let data_array = parsed["data"].as_array()
        .ok_or("Missing 'data' field")?;
    let labels_array = parsed["labels"].as_array()
        .ok_or("Missing 'labels' field")?;
    
    let mut items = Vec::new();
    
    for (data_row, label) in data_array.iter().zip(labels_array.iter()) {
        let image_data: Vec<f32> = data_row.as_array()
            .ok_or("Invalid image data format")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        
        let label_val = label.as_u64().unwrap_or(0) as usize;
        
        items.push(Cifar10Item {
            image: image_data,
            label: label_val,
        });
    }
    
    println!("Loaded {} samples from {}", items.len(), file_path);
    Ok(items)
}

pub fn load_cifar10_dataset() -> Result<(Vec<Cifar10Item>, Vec<Cifar10Item>), Box<dyn std::error::Error>> {
    let data_dir = "data/cifar-10-batches-py";
    
    // Check if data directory exists
    if !Path::new(data_dir).exists() {
        return Err(format!("CIFAR-10 data directory not found: {}", data_dir).into());
    }
    
    println!("Loading CIFAR-10 dataset from: {}", data_dir);
    
    // Load training batches
    let mut train_data = Vec::new();
    for i in 1..=5 {
        let batch_file = format!("{}/data_batch_{}", data_dir, i);
        let batch_data = load_cifar10_batch_python(&batch_file)?;
        train_data.extend(batch_data);
    }
    
    // Load test batch
    let test_file = format!("{}/test_batch", data_dir);
    let test_data = load_cifar10_batch_python(&test_file)?;
    
    println!("Loaded {} training samples and {} test samples", 
             train_data.len(), test_data.len());
    
    Ok((train_data, test_data))
}

pub fn get_cifar10_dataloaders<B: Backend>(
    device: &B::Device,
    config: &Cifar10Config,
) -> Result<(
    std::sync::Arc<dyn burn::data::dataloader::DataLoader<B, Cifar10Batch<B>>>,
    std::sync::Arc<dyn burn::data::dataloader::DataLoader<B, Cifar10Batch<B>>>,
), Box<dyn std::error::Error>> {
    // Load the actual CIFAR-10 data
    let (train_data, test_data) = load_cifar10_dataset()?;
    
    // Create datasets
    let train_dataset = InMemDataset::new(train_data);
    let test_dataset = InMemDataset::new(test_data);
    
    // Create batchers
    let batcher_train = Cifar10Batcher::<B>::new(device.clone());
    let batcher_test = Cifar10Batcher::<B>::new(device.clone());

    // Create data loaders
    let train_dataloader = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.train_batch_size)
        .shuffle(42)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let test_dataloader = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.test_batch_size)
        .num_workers(config.num_workers)
        .build(test_dataset);

    Ok((train_dataloader, test_dataloader))
}

pub fn get_cifar10_classes() -> Vec<&'static str> {
    vec![
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]
}
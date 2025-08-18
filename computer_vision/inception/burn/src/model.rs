use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
    },
    tensor::{activation::relu, backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct InceptionBlock<B: Backend> {
    // Branch 1: 1x1 conv
    branch1: Conv2d<B>,

    // Branch 2: 1x1 conv -> 3x3 conv
    branch2_1x1: Conv2d<B>,
    branch2_3x3: Conv2d<B>,

    // Branch 3: 1x1 conv -> 5x5 conv
    branch3_1x1: Conv2d<B>,
    branch3_5x5: Conv2d<B>,

    // Branch 4: 3x3 maxpool -> 1x1 conv
    branch4_pool: MaxPool2d,
    branch4_1x1: Conv2d<B>,
}

impl<B: Backend> InceptionBlock<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_1x1: usize,
        reduce_3x3: usize,
        out_3x3: usize,
        reduce_5x5: usize,
        out_5x5: usize,
        pool_proj: usize,
    ) -> Self {
        let branch1 = Conv2dConfig::new([in_channels, out_1x1], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init(device);

        let branch2_1x1 = Conv2dConfig::new([in_channels, reduce_3x3], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init(device);
        let branch2_3x3 = Conv2dConfig::new([reduce_3x3, out_3x3], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let branch3_1x1 = Conv2dConfig::new([in_channels, reduce_5x5], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init(device);
        let branch3_5x5 = Conv2dConfig::new([reduce_5x5, out_5x5], [5, 5])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);

        let branch4_pool = MaxPool2dConfig::new([3, 3])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();
        let branch4_1x1 = Conv2dConfig::new([in_channels, pool_proj], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init(device);

        Self {
            branch1,
            branch2_1x1,
            branch2_3x3,
            branch3_1x1,
            branch3_5x5,
            branch4_pool,
            branch4_1x1,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Branch 1: 1x1 conv
        let branch1 = relu(self.branch1.forward(x.clone()));

        // Branch 2: 1x1 -> 3x3
        let branch2 = relu(self.branch2_1x1.forward(x.clone()));
        let branch2 = relu(self.branch2_3x3.forward(branch2));

        // Branch 3: 1x1 -> 5x5
        let branch3 = relu(self.branch3_1x1.forward(x.clone()));
        let branch3 = relu(self.branch3_5x5.forward(branch3));

        // Branch 4: maxpool -> 1x1
        let branch4 = self.branch4_pool.forward(x);
        let branch4 = relu(self.branch4_1x1.forward(branch4));

        // Concatenate along channel dimension
        Tensor::cat(vec![branch1, branch2, branch3, branch4], 1)
    }
}

#[derive(Module, Debug)]
pub struct SimpleInceptionNet<B: Backend> {
    conv1: Conv2d<B>,
    pool1: MaxPool2d,
    inception1: InceptionBlock<B>,
    inception2: InceptionBlock<B>,
    pool2: MaxPool2d,
    avgpool: AdaptiveAvgPool2d,
    dropout: Dropout,
    fc: Linear<B>,
}

impl<B: Backend> SimpleInceptionNet<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        let conv1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        
        let pool1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        // Inception block 1: 64 -> 64 channels (16+32+8+8)
        let inception1 = InceptionBlock::new(device, 64, 16, 24, 32, 4, 8, 8);
        
        // Inception block 2: 64 -> 128 channels (32+64+16+16)
        let inception2 = InceptionBlock::new(device, 64, 32, 48, 64, 8, 16, 16);

        let pool2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        
        let dropout = DropoutConfig::new(0.4).init();
        
        let fc = LinearConfig::new(128, num_classes).init(device);

        Self {
            conv1,
            pool1,
            inception1,
            inception2,
            pool2,
            avgpool,
            dropout,
            fc,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.conv1.forward(x));
        let x = self.pool1.forward(x);

        let x = self.inception1.forward(x);
        let x = self.inception2.forward(x);

        let x = self.pool2.forward(x);
        let x = self.avgpool.forward(x);
        
        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);
        
        let x = self.dropout.forward(x);
        self.fc.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub num_classes: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self { num_classes: 10 }
    }
}
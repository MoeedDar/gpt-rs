use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, GELU};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

struct MultilayerPeceptronConfig {
    dim: usize,
    layers: usize,
    dropout: f64,
}

impl MultilayerPeceptronConfig {
    pub fn init<B: Backend>(&self) -> MultilayerPeceptron<B> {
        MultilayerPeceptron {
            channel: LinearConfig::new(self.dim, self.layers * self.dim).init(),
            projection: LinearConfig::new(self.layers * self.dim, self.dim).init(),
            activation: GELU::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct MultilayerPeceptron<B: Backend> {
    channel: Linear<B>,
    projection: Linear<B>,
    activation: GELU,
    dropout: Dropout,
}

impl<B: Backend> MultilayerPeceptron<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = self.channel.forward(x);
        x = self.activation.forward(x);
        x = self.projection.forward(x);
        self.dropout.forward(x)
    }
}

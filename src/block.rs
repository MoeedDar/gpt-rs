use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor};

use crate::attention::Attention;
use crate::mlp::MultilayerPeceptron;

struct BlockConfig {}

// Transformer block.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    linear1: Linear<B>,
    attention: Attention<B>,
    linear2: Linear<B>,
    multilayer_peceptron: MultilayerPeceptron<B>,
    size: usize,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self
            .attention
            .forward(self.linear1.forward(x.clone()), mask(self.size))
            .add(x.clone());

        self.multilayer_peceptron
            .forward(self.linear2.forward(x.clone()))
            .add(x)
    }
}

fn mask<B: Backend>(size: usize) -> Tensor<B, 4, burn::tensor::Bool> {
    Tensor::<_, 4, burn::tensor::Float>::ones(Shape::new([1, 1, size, size]))
        .lower(Tensor::zeros(Shape::new([1, 1, size, size])))
}

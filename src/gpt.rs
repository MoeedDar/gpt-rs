use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::transformer::Transformer;

#[derive(Config)]
pub struct GPTConfig {

}

impl GPTConfig {
    pub fn from_pretrained() {
    }
}

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    transformer: Transformer<B>,
    head: Linear<B>,
    block_size: usize,
}

impl<B: Backend> GPT<B> {
    fn forward(&self, indices: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.head.forward(self.transformer.forward(indices))
    }

    pub fn generate(
        &self,
        indices: Tensor<B, 2, burn::tensor::Int>,
        max_tokens: usize,
        temperature: f64,
    ) -> Tensor<B, 2, burn::tensor::Int> {
        let mut indices = indices;

        for _ in 0..max_tokens {
            let indices_shape = indices.shape().dims;

            if indices_shape[1] > self.block_size {
                indices =
                    indices.index([0..indices_shape[0], 0..indices_shape[1] - self.block_size]);
            }

            let logits = self.forward(indices.clone());
            let [x, y, z] = logits.shape().dims;
            let logits = logits.index([0..x, 0..y - 1, 0..z]) / temperature;

            let probabilities = softmax(logits, 2);

            let next = probabilities.argmax(2).unsqueeze();

            Tensor::cat(vec![indices.clone(), next], 2);
        }

        indices
    }
}

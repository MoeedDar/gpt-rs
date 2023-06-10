use burn::{
    module::Module,
    nn::{Dropout, Embedding, LayerNorm},
    tensor::{backend::Backend, Tensor},
};

use crate::block::Block;

struct TransformerConfig {}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    token_embedding: Embedding<B>,
    word_embedding: Embedding<B>,
    dropout: Dropout,
    blocks: Vec<Block<B>>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> Transformer<B> {
    pub fn forward(
        &self,
        indices: Tensor<B, 2, burn::tensor::Int>,
    ) -> Tensor<B, 3, burn::tensor::Float> {
        let [_, size] = indices.shape().dims;

        let positions = Tensor::arange(0..size).unsqueeze();

        let token_embedding = self.token_embedding.forward(indices.clone());
        let word_embedding = self.word_embedding.forward(positions);

        let mut x = self.dropout.forward(token_embedding + word_embedding);
        self.blocks.iter().for_each(|b| x = b.forward(&x));

        self.layer_norm.forward(x)
    }
}

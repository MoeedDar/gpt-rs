use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor};

pub struct AttentionConfig {
    dim: usize,
    heads: usize,
    attention_dropout: f64,
    residual_dropout: f64,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self) -> Attention<B> {
        Attention {
            attention: LinearConfig::new(self.dim, 3 * self.dim).init(),
            projection: LinearConfig::new(self.dim, self.dim).init(),
            attention_dropout: DropoutConfig::new(self.attention_dropout).init(),
            residual_dropout: DropoutConfig::new(self.residual_dropout).init(),
            heads: self.heads,
        }
    }
}

// Self attention mechanism.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    attention: Linear<B>,
    projection: Linear<B>,
    attention_dropout: Dropout,
    residual_dropout: Dropout,
    heads: usize,
}

impl<B: Backend> Attention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, burn::tensor::Bool>) -> Tensor<B, 3> {
        let [batch_size, seq_length, dim] = x.shape().dims;

        let attention_forward = |head: usize| {
            self.attention
                .forward(x.clone())
                .clone()
                .index([0..batch_size, 0..seq_length, 0..dim * (head - 1)])
                .reshape(Shape::new([
                    batch_size,
                    seq_length,
                    self.heads,
                    dim / self.heads,
                ]))
                .transpose()
        };

        let query = attention_forward(0);
        let key = attention_forward(1);
        let value = attention_forward(1);

        let key_dims = key.clone().dims();

        let attention = query
            .matmul(key.swap_dims(2, 3))
            .mul_scalar(*key_dims.get(3).unwrap() as f64)
            .mask_fill(mask, f64::NEG_INFINITY);

        let output = self
            .attention_dropout
            .forward(softmax(attention, 3))
            .matmul(value)
            .swap_dims(1, 2)
            .reshape(Shape::new([batch_size, seq_length, dim]));

        self.residual_dropout
            .forward(self.projection.forward(output))
    }
}

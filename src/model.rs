// model.rs
use crate::constants::model_dims;
use burn::prelude::*;
use burn::{
    nn::{Linear, LinearConfig, Relu, LayerNorm, LayerNormConfig}, // LayerNorm と LayerNormConfig を追加
    tensor::activation::softplus,
};

#[derive(Module, Debug)]
pub struct TuningForkPINN<B: Backend> {
    layer_1: Linear<B>,
    activation_1: Relu,
    norm_1: LayerNorm<B>, // 追加
    layer_2: Linear<B>,
    activation_2: Relu,
    norm_2: LayerNorm<B>, // 追加
    layer_3: Linear<B>,
    activation_3: Relu,
    norm_3: LayerNorm<B>, // 追加
    output_layer: Linear<B>,
}

impl<B: Backend> TuningForkPINN<B> {
    pub fn new(device: &B::Device) -> Self {
        let hidden_size = 128;
        Self {
            layer_1: LinearConfig::new(1, hidden_size).init(device),
            activation_1: Relu::new(),
            norm_1: LayerNormConfig::new(hidden_size).init(device), // 初期化
            layer_2: LinearConfig::new(hidden_size, hidden_size).init(device),
            activation_2: Relu::new(),
            norm_2: LayerNormConfig::new(hidden_size).init(device), // 初期化
            layer_3: LinearConfig::new(hidden_size, hidden_size).init(device),
            activation_3: Relu::new(),
            norm_3: LayerNormConfig::new(hidden_size).init(device), // 初期化
            output_layer: LinearConfig::new(hidden_size, model_dims::NUM_DIMS).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_1.forward(input);
        let x = self.activation_1.forward(x);
        let x = self.norm_1.forward(x); // 適用
        let x = self.layer_2.forward(x);
        let x = self.activation_2.forward(x);
        let x = self.norm_2.forward(x); // 適用
        let x = self.layer_3.forward(x);
        let x = self.activation_3.forward(x);
        let x = self.norm_3.forward(x); // 適用
        let x = self.output_layer.forward(x);

        softplus(x, 1.0)
    }
}
//! # ニューラルネットワークモデル
//!
//! 周波数から音叉の寸法を予測するための、シンプルな多層パーセプトロン (MLP) モデルを定義します。

use crate::constants::model_dims;
use burn::prelude::*;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    tensor::activation::softplus,
};

/// 音叉の寸法を予測するPINNモデル。
///
/// ## アーキテクチャ
/// - 入力: 周波数 (1次元)
/// - 隠れ層: 3層の全結合層 (活性化関数: ReLU)
/// - 出力: 音叉の寸法 (5次元)
///   - [柄の長さ, 柄の直径, プロングの長さ, プロングの直径, プロングの間隔]
///
/// ## 活性化関数
/// - 隠れ層には`ReLU`を使用。
/// - 出力層には`softplus`を使用し、寸法が必ず正の値になるように制約をかけます。
#[derive(Module, Debug)]
pub struct TuningForkPINN<B: Backend> {
    layer_1: Linear<B>,
    activation_1: Relu,
    layer_2: Linear<B>,
    activation_2: Relu,
    layer_3: Linear<B>,
    activation_3: Relu,
    output_layer: Linear<B>,
}

impl<B: Backend> TuningForkPINN<B> {
    /// 新しい `TuningForkPINN` モデルを初期化します。
    pub fn new(device: &B::Device) -> Self {
        let hidden_size = 128;
        Self {
            layer_1: LinearConfig::new(1, hidden_size).init(device),
            activation_1: Relu::new(),
            layer_2: LinearConfig::new(hidden_size, hidden_size).init(device),
            activation_2: Relu::new(),
            layer_3: LinearConfig::new(hidden_size, hidden_size).init(device),
            activation_3: Relu::new(),
            output_layer: LinearConfig::new(hidden_size, model_dims::NUM_DIMS).init(device),
        }
    }

    /// モデルのフォワードパス。
    ///
    /// # Arguments
    /// * `input` - 周波数のテンソル。形状は `[batch_size, 1]`。
    ///
    /// # Returns
    /// 予測された寸法のテンソル。形状は `[batch_size, 5]`。
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_1.forward(input);
        let x = self.activation_1.forward(x);
        let x = self.layer_2.forward(x);
        let x = self.activation_2.forward(x);
        let x = self.layer_3.forward(x);
        let x = self.activation_3.forward(x);
        let x = self.output_layer.forward(x);

        // softplusを適用して出力が必ず正の値になるようにする
        softplus(x, 1.0)
    }
}

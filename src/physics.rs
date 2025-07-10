//! # 物理情報に基づいた損失関数
//!
//! このモジュールは、PINNの核となるカスタム損失関数を定義します。

use crate::constants::physics::*;
use burn::prelude::*;
use burn::tensor::activation::relu;

/// 音叉の物理法則と制約に基づいた損失を計算します。
///
/// この関数は、ニューラルネットワークが予測した寸法から周波数を計算し、
/// 目標周波数との誤差（損失）を算出します。
/// さらに、物理的に不適切な寸法に対するペナルティを追加します。

pub fn tuning_fork_loss<B: Backend>(
    predicted_dims: Tensor<B, 2>,
    target_freqs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let batch_size = predicted_dims.dims()[0];

    let prong_length = predicted_dims.clone().slice([0..batch_size, 2..3]);
    let prong_diameter = predicted_dims.clone().slice([0..batch_size, 3..4]);

    let pi = std::f32::consts::PI;

    // 修正: prong_diameterが複数回使われるため、都度clone()する
    let area = prong_diameter.clone().powf_scalar(2.0) * (pi / 4.0);
    let moment_of_inertia = prong_diameter.clone().powf_scalar(4.0) * (pi / 64.0);

    let stiffness = (moment_of_inertia * YOUNGS_MODULUS).sqrt();
    let density_mass = (area * DENSITY).sqrt();

    // 修正: prong_length.clone()で所有権を渡す
    let predicted_freqs =
        (stiffness / density_mass) * FREQ_K_CONSTANT / prong_length.clone().powf_scalar(2.0);
    let frequency_loss = (predicted_freqs - target_freqs).powf_scalar(2.0);

    let handle_length = predicted_dims.slice([0..batch_size, 0..1]);

    // 修正: ここでprong_lengthの所有権がムーブされる
    let ratio_penalty = relu(prong_length - handle_length).powf_scalar(2.0);

    // 修正: ここでprong_diameterの所有権がムーブされる
    let range_penalty = relu((prong_diameter.clone() - 0.002).neg()).powf_scalar(2.0)
        + relu(prong_diameter - 0.02).powf_scalar(2.0);

    let total_loss = (frequency_loss
        + ratio_penalty * PENALTY_WEIGHT_RATIO
        + range_penalty * PENALTY_WEIGHT_RANGE)
        .mean();

    total_loss
}

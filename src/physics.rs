//! # 物理情報に基づいた損失関数
//!
//! このモジュールは、PINNの核となるカスタム損失関数を定義します。

use crate::constants::{model_dims, physics::*};
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
    let pi = std::f32::consts::PI;

    // --- 各次元のテンソルを最初にスライス ---
    // .slice()は所有権を消費するため、都度.clone()が必要
    let predicted_dims_cloned = predicted_dims.clone();

    let handle_length = predicted_dims_cloned.clone().slice([
        0..batch_size,
        model_dims::HANDLE_LENGTH_IDX..(model_dims::HANDLE_LENGTH_IDX + 1),
    ]);
    let handle_diameter = predicted_dims_cloned.clone().slice([
        0..batch_size,
        model_dims::HANDLE_DIAMETER_IDX..(model_dims::HANDLE_DIAMETER_IDX + 1),
    ]);
    let prong_length = predicted_dims_cloned.clone().slice([
        0..batch_size,
        model_dims::PRONG_LENGTH_IDX..(model_dims::PRONG_LENGTH_IDX + 1),
    ]);
        let prong_diameter = predicted_dims_cloned.clone().slice([
        0..batch_size,
        model_dims::PRONG_DIAMETER_IDX..(model_dims::PRONG_DIAMETER_IDX + 1),
    ]);
    let prong_gap = predicted_dims_cloned.slice([
        0..batch_size,
        model_dims::PRONG_GAP_IDX..(model_dims::PRONG_GAP_IDX + 1),
    ]);

    // --- 1. 周波数損失の計算 ---
    let area = prong_diameter.clone().powf_scalar(2.0) * (pi / 4.0);
    let moment_of_inertia = prong_diameter.clone().powf_scalar(4.0) * (pi / 64.0);

    let stiffness = moment_of_inertia * YOUNGS_MODULUS; // E * I
    let density_mass = area * DENSITY; // rho * A

    // ゼロ除算を防ぐために微小な値を加算
    let epsilon = 1e-8;

    let sqrt_term = (stiffness / (density_mass + epsilon)).sqrt();
    let length_term = prong_length.clone().powf_scalar(2.0);

    let predicted_freqs =
        sqrt_term.mul_scalar(K_FACTOR / (2.0 * pi)) / length_term;

    let frequency_loss = (predicted_freqs - target_freqs).powf_scalar(2.0);

    // --- 2. 物理的制約に対するペナルティの計算 ---
    // ペナルティ1: プロングはハンドルより長くてはならない
    let ratio_penalty = relu(prong_length.clone() - handle_length.clone()).powf_scalar(2.0);

    // ペナルティ2: 各寸法が物理的に妥当な範囲にあるか
    // relu((x - min_val).neg()) は x < min_val の時にペナルティ
    // relu(x - max_val) は x > max_val の時にペナルティ
    let prong_diameter_penalty = relu((prong_diameter.clone() - 0.002).neg()).powf_scalar(2.0)
        + relu(prong_diameter - 0.02).powf_scalar(2.0);

    let prong_length_penalty = relu((prong_length.clone() - 0.01).neg()).powf_scalar(2.0)
        + relu(prong_length - 0.2).powf_scalar(2.0);

    let handle_length_penalty = relu((handle_length.clone() - 0.03).neg()).powf_scalar(2.0)
        + relu(handle_length - 0.15).powf_scalar(2.0);

    let handle_diameter_penalty =
        relu((handle_diameter.clone() - 0.005).neg()).powf_scalar(2.0)
            + relu(handle_diameter - 0.02).powf_scalar(2.0);

    let prong_gap_penalty = relu((prong_gap.clone() - 0.002).neg()).powf_scalar(2.0)
        + relu(prong_gap - 0.02).powf_scalar(2.0);


    // --- 3. 合計損失の計算 ---
    let total_loss = (frequency_loss
        + ratio_penalty * PENALTY_WEIGHT_RATIO
        // 重要な寸法（周波数に直接影響）には高い重み
        + (prong_diameter_penalty + prong_length_penalty) * PENALTY_WEIGHT_RANGE
        // その他の物理的制約には低い重み
        + (handle_length_penalty + handle_diameter_penalty + prong_gap_penalty)
            * PENALTY_WEIGHT_OTHER)
        .mean();

    total_loss
}

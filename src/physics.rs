//! # 物理情報に基づいた損失関数
//!
//! このモジュールは、PINNの核となるカスタム損失関数を定義します。

use crate::constants::{model_dims, physics::*};
use burn::prelude::*;
use burn::tensor::{activation::relu, Tensor};

/// 音叉の物理法則と制約に基づいた損失を計算します。
///
/// この関数は、ニューラルネットワークが予測した寸法から周波数を計算し、
/// 目標周波数との誤差（損失）を算出します。
/// さらに、物理的に不適切な寸法に対するペナルティを追加します。
/// 音叉の物理法則と制約に基づいた損失を計算します。
///
/// この関数は、ニューラルネットワークが予測した寸法から周波数を計算し、
/// 目標周波数との誤差（損失）を算出します。
/// さらに、物理的に不適切な寸法に対するペナルティを追加します。
///
/// # Note
///
/// 計算効率を向上させるため、中間テンソルの生成と`.clone()`の呼び出しを
/// 最小限に抑えるように最適化されています。
pub fn tuning_fork_loss<B: Backend>(
    predicted_dims: Tensor<B, 2>,
    target_freqs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let pi = std::f32::consts::PI;
    let epsilon = 1e-8;

    // --- 各次元のテンソルへの参照を取得 ---
    let dim_tensors = predicted_dims.split(1, 1);
    let handle_length = &dim_tensors[model_dims::HANDLE_LENGTH_IDX];
    let handle_diameter = &dim_tensors[model_dims::HANDLE_DIAMETER_IDX];
    let prong_length = &dim_tensors[model_dims::PRONG_LENGTH_IDX];
    let prong_diameter = &dim_tensors[model_dims::PRONG_DIAMETER_IDX];
    let prong_gap = &dim_tensors[model_dims::PRONG_GAP_IDX];

    // --- 1. 周波数損失の計算 (中間テンソルの削減) ---
    let prong_d2 = prong_diameter.clone().powf_scalar(2.0);
    let area = prong_d2.clone() * (pi / 4.0);
    let moment_of_inertia = prong_d2.powf_scalar(2.0) * (pi / 64.0);

    let stiffness = moment_of_inertia * YOUNGS_MODULUS;
    let density_mass = area * DENSITY;

    let sqrt_term = (stiffness / (density_mass + epsilon)).sqrt();
    let length_term = prong_length.clone().powf_scalar(2.0);

    let predicted_freqs = sqrt_term.mul_scalar(K_FACTOR / (2.0 * pi)) / length_term;
    let frequency_loss = (predicted_freqs - target_freqs).powf_scalar(2.0);

    // --- 2. 物理的制約に対するペナルティの計算 ---
    let ratio_penalty = relu(prong_length.clone() - handle_length.clone()).powf_scalar(2.0);

    let prong_diameter_penalty = relu(0.002 - prong_diameter.clone()).powf_scalar(2.0)
        + relu(prong_diameter.clone() - 0.02).powf_scalar(2.0);

    let prong_length_penalty = relu(0.01 - prong_length.clone()).powf_scalar(2.0)
        + relu(prong_length.clone() - 0.2).powf_scalar(2.0);

    let handle_length_penalty = relu(0.03 - handle_length.clone()).powf_scalar(2.0)
        + relu(handle_length.clone() - 0.15).powf_scalar(2.0);

    let handle_diameter_penalty =
        relu(0.005 - handle_diameter.clone()).powf_scalar(2.0)
            + relu(handle_diameter.clone() - 0.02).powf_scalar(2.0);

    let prong_gap_penalty = relu(0.002 - prong_gap.clone()).powf_scalar(2.0)
        + relu(prong_gap.clone() - 0.02).powf_scalar(2.0);

    // --- 3. 合計損失の計算 ---
    let total_loss = (frequency_loss
        + ratio_penalty * PENALTY_WEIGHT_RATIO
        + (prong_diameter_penalty + prong_length_penalty) * PENALTY_WEIGHT_RANGE
        + (handle_length_penalty + handle_diameter_penalty + prong_gap_penalty)
            * PENALTY_WEIGHT_OTHER)
        .mean();

    total_loss
}

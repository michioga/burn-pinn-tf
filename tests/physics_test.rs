//! 物理法則の損失関数に対するユニットテスト

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Tensor;
use burn_tuningfork_pinn::constants::physics::*;
use burn_tuningfork_pinn::physics::tuning_fork_loss;

type B = NdArray<f32>;

#[test]
fn test_loss_calculation_no_penalty() {
    let device: NdArrayDevice = Default::default();
    let predicted_dims = Tensor::<B, 2>::from_floats(
        [[0.10, 0.01, 0.08, 0.005, 0.01]], // [m]
        &device,
    );
    let target_freq = Tensor::<B, 2>::from_floats([[440.0]], &device); // [Hz]

    let loss = tuning_fork_loss(predicted_dims.clone(), target_freq);

    let prong_length: f32 = 0.08;
    let prong_diameter: f32 = 0.005;
    let pi: f32 = std::f32::consts::PI;

    let area = prong_diameter.powi(2) * (pi / 4.0);
    let moment_of_inertia = prong_diameter.powi(4) * (pi / 64.0);
    let stiffness = (moment_of_inertia * YOUNGS_MODULUS).sqrt();
    let density_mass = (area * DENSITY).sqrt();
    let predicted_freq = (stiffness / density_mass) * FREQ_K_CONSTANT / prong_length.powi(2);
    let expected_freq_loss = (predicted_freq - 440.0).powi(2);
    let expected_total_loss = expected_freq_loss;

    let loss_value = loss.into_data().into_vec::<f32>().unwrap()[0];
    assert!((loss_value - expected_total_loss).abs() < 1e-6);
}

#[test]
fn test_loss_calculation_with_penalties() {
    let device: NdArrayDevice = Default::default();
    let predicted_dims = Tensor::<B, 2>::from_floats([[0.05, 0.01, 0.08, 0.001, 0.01]], &device);
    let target_freq = Tensor::<B, 2>::from_floats([[440.0]], &device);

    let loss = tuning_fork_loss(predicted_dims.clone(), target_freq);

    let prong_length: f32 = 0.08;
    let prong_diameter: f32 = 0.001;
    let pi: f32 = std::f32::consts::PI;
    let area = prong_diameter.powi(2) * (pi / 4.0);
    let moment_of_inertia = prong_diameter.powi(4) * (pi / 64.0);
    let stiffness = (moment_of_inertia * YOUNGS_MODULUS).sqrt();
    let density_mass = (area * DENSITY).sqrt();
    let predicted_freq = (stiffness / density_mass) * FREQ_K_CONSTANT / prong_length.powi(2);
    let freq_loss = (predicted_freq - 440.0).powi(2);

    let ratio_penalty = (0.08_f32 - 0.05_f32).powi(2) * PENALTY_WEIGHT_RATIO;
    let range_penalty = (0.001_f32 - 0.002_f32).powi(2) * PENALTY_WEIGHT_RANGE;
    let expected_total_loss = freq_loss + ratio_penalty + range_penalty;

    let loss_value = loss.into_data().into_vec::<f32>().unwrap()[0];
    assert!((loss_value - expected_total_loss).abs() < 1e-6);
}

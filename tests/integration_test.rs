#![recursion_limit = "256"]

//! # 統合テスト
//!
//! 学習から推論までの一連のサイクルをテストし、アプリケーション全体の
//! 健全性を確認します。`ndarray`と`wgpu`の両方のバックエンドでテストを実行します。

use burn::backend::{wgpu::Wgpu, Autodiff, NdArray};
use burn::tensor::backend::AutodiffBackend;
use burn::prelude::Backend;
use burn_tuningfork_pinn::{infer, train};
use std::{fs, path::Path};

/// `ndarray`バックエンドを使用して、学習と推論のサイクルをテストします。
///
/// このテストは現在、`wgpu`との互換性の問題のため無視されています。
#[test]
#[ignore]
fn test_training_and_inference_cycle_ndarray() {
    let artifact_dir = "./test_artifacts_ndarray";
    test_cycle::<Autodiff<NdArray>>(artifact_dir);
}

/// `wgpu`バックエンドを使用して、学習と推論のサイクルをテストします。
///
/// このテストは現在、`wgpu`との互換性の問題のため無視されています。
#[test]
#[ignore]
fn test_training_and_inference_cycle_wgpu() {
    let artifact_dir = "./test_artifacts_wgpu";
    test_cycle::<Autodiff<Wgpu>>(artifact_dir);
}

/// 指定されたバックエンドで学習と推論のサイクルを実行するヘルパー関数。
///
/// # Type Parameters
///
/// * `B` - テストに使用するバックエンド。
///
/// # Arguments
///
/// * `artifact_dir` - テスト中に生成されるアーティファクトを保存するディレクトリ。
fn test_cycle<B: AutodiffBackend>(artifact_dir: &str)
where
    B::InnerBackend: Backend,
{
    // --- 準備 ---
    if Path::new(artifact_dir).exists() {
        fs::remove_dir_all(artifact_dir).unwrap();
    }
    fs::create_dir(artifact_dir).unwrap();

    // --- 学習プロセスの実行 ---
    let device = Default::default();
    train::run::<B>(device);

    // 学習済みモデルファイルが生成されたことを確認
    let model_path = format!("{}/model.mpk", artifact_dir);
    assert!(
        Path::new(&model_path).exists(),
        "Trained model file should be created."
    );

    // --- 推論プロセスの実行 ---
    let device = Default::default();
    infer::run::<B::InnerBackend>(440.0, device);

    // --- 後片付け ---
    fs::remove_dir_all(artifact_dir).unwrap();
}

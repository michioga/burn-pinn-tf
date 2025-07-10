//! # 推論ロジック
//! 
//! 学習済みのモデルを読み込み、指定された周波数に対する音叉の寸法を推論します。
//! このモジュールは、任意のバックエンドで動作するようにジェネリックになっています。

use crate::model::TuningForkPINN;
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};

/// 推論プロセスを実行します。
///
/// # Arguments
///
/// * `freq` - 推論したい音叉の周波数 (Hz)。
/// * `device` - 推論に使用するデバイス（例: `WgpuDevice`、`NdArrayDevice`）。
///
/// # Panics
///
/// モデルファイルの読み込みに失敗した場合にパニックします。
pub fn run<B: Backend>(freq: f32, device: B::Device) {
    let artifact_dir = "./artifacts";
    let model_path = format!("{artifact_dir}/model");

    // 保存されたモデルのレコードを読み込む
    let record = CompactRecorder::new()
        .load(model_path.into(), &device)
        .expect(
            "Failed to load model record. Run training first via `cargo run --release -- train`",
        );

    // レコードからモデルを復元
    let model: TuningForkPINN<B> = TuningForkPINN::new(&device).load_record(record);

    // 入力テンソルを作成
    let input = Tensor::<B, 2>::from_floats([[freq]], &device);

    // 推論を実行
    let dims = model.forward(input);
    let dims_values: Vec<f32> = dims.into_data().convert::<f32>().into_vec().unwrap();

    // 結果を表示
    println!("\n--- Predicted Dimensions (in meters) ---");
    println!("  - Handle Length:     {:.6}", dims_values[0]);
    println!("  - Handle Diameter:   {:.6}", dims_values[1]);
    println!("  - Prong Length:      {:.6}", dims_values[2]);
    println!("  - Prong Diameter:    {:.6}", dims_values[3]);
    println!("  - Prong Gap:         {:.6}", dims_values[4]);
    println!("----------------------------------------");
}

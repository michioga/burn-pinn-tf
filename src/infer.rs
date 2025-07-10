//! # 推論ロジック
//!
//! 学習済みのモデルを読み込み、指定された周波数に対する
//! 音叉の寸法を推論して表示します。

use crate::model::TuningForkPINN;
use burn::{
    backend::ndarray::NdArray,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

/// 推論プロセスを実行します。
pub fn run(freq: f32) {
    // バックエンドの型をndarrayに固定
    type AppBackend = NdArray<f32>;

    let device = Default::default();
    let artifact_dir = "./artifacts";
    let model_path = format!("{artifact_dir}/model");

    // 保存されたモデルのレコードを読み込む
    let record = CompactRecorder::new()
        .load(model_path.into(), &device)
        .expect(
            "Failed to load model record. Run training first via `cargo run --release -- train`",
        );

    // レコードからモデルを復元
    let model: TuningForkPINN<AppBackend> = TuningForkPINN::new(&device).load_record(record);

    // 入力テンソルを作成
    let input = Tensor::<AppBackend, 2>::from_floats([[freq]], &device);

    // 推論を実行
    let dims = model.forward(input);
    let dims_values = dims.into_data().into_vec::<f32>().unwrap();

    // 結果を表示
    println!("\n--- Predicted Dimensions (in meters) ---");
    println!("  - Handle Length:     {:.6}", dims_values[0]);
    println!("  - Handle Diameter:   {:.6}", dims_values[1]);
    println!("  - Prong Length:      {:.6}", dims_values[2]);
    println!("  - Prong Diameter:    {:.6}", dims_values[3]);
    println!("  - Prong Gap:         {:.6}", dims_values[4]);
    println!("----------------------------------------");
}

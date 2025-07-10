//! 学習から推論までの一連の流れをテストする統合テスト

use burn::record::{CompactRecorder, Recorder};
use burn_tuningfork_pinn::{model::TuningForkPINN, train};
use std::{fs, path::Path};

#[test]
fn test_training_and_inference_cycle() {
    // --- 準備 ---
    let artifact_dir = "./test_artifacts";
    if Path::new(artifact_dir).exists() {
        fs::remove_dir_all(artifact_dir).unwrap();
    }
    fs::create_dir(artifact_dir).unwrap();

    // --- 学習プロセスの実行 ---
    // ▼▼▼【ここを修正】▼▼▼
    let mut config = train::TrainingConfig::new(burn::optim::AdamConfig::new());
    config.num_epochs = 1;
    config.batch_size = 8;

    train_for_test(&config, artifact_dir);

    // 学習済みモデルファイルが生成されたことを確認
    let model_path = format!("{}/model.mpk", artifact_dir);
    assert!(
        Path::new(&model_path).exists(),
        "Trained model file should be created."
    );

    // --- 推論プロセスの実行 ---
    infer_for_test(440.0, artifact_dir);

    // --- 後片付け ---
    fs::remove_dir_all(artifact_dir).unwrap();
}

/// テスト用に簡略化した学習関数
// ▼▼▼【ここを修正】▼▼▼
fn train_for_test(config: &train::TrainingConfig, artifact_dir: &str) {
    use burn::backend::{Autodiff, NdArray};
    use burn::data::dataloader::DataLoaderBuilder;
    use burn::lr_scheduler::constant::ConstantLr;
    use burn::module::Module;
    use burn::optim::AdamConfig;
    use burn::train::{LearnerBuilder, metric::LossMetric};
    use burn_tuningfork_pinn::train::{TuningForkBatcher, TuningForkDataset};

    type AppBackend = NdArray<f32>;
    type AppAutodiffBackend = Autodiff<AppBackend>;
    let device = Default::default();

    let dataloader_train = DataLoaderBuilder::new(TuningForkBatcher::new(device))
        .batch_size(config.batch_size)
        .num_workers(0)
        .build(TuningForkDataset {
            size: config.batch_size * 2,
            freq_range: (200.0, 1800.0),
        });

    let dataloader_valid = DataLoaderBuilder::new(TuningForkBatcher::new(device))
        .batch_size(config.batch_size)
        .num_workers(0)
        .build(TuningForkDataset {
            size: config.batch_size * 2,
            freq_range: (1800.0, 2000.0),
        });

    let learner = LearnerBuilder::new(artifact_dir)
        .num_epochs(config.num_epochs)
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .build(
            TuningForkPINN::<AppAutodiffBackend>::new(&device),
            AdamConfig::new().init(),
            ConstantLr::new(config.learning_rate),
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    CompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{}/model", artifact_dir).into(),
        )
        .unwrap();
}

/// テスト用に簡略化した推論関数
fn infer_for_test(freq: f32, artifact_dir: &str) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::tensor::Tensor;

    type AppBackend = NdArray<f32>;
    let device = Default::default();
    let model_path = format!("{}/model", artifact_dir);

    let record = CompactRecorder::new()
        .load(model_path.into(), &device)
        .expect("Failed to load model record for inference test.");

    let model: TuningForkPINN<AppBackend> = TuningForkPINN::new(&device).load_record(record);
    let input = Tensor::<AppBackend, 2>::from_floats([[freq]], &device);

    let _dims = model.forward(input);
}

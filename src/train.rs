//! # 学習ロジック
//!
//! `burn`の`Learner` APIを使用して、物理情報ニューラルネットワーク（PINN）の学習プロセスを管理します。
//! このモジュールは、任意のバックエンドで動作するようにジェネリックになっています。

use crate::model::TuningForkPINN;
use crate::physics::tuning_fork_loss;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataloader::batcher::Batcher, dataset::Dataset},
    lr_scheduler::constant::ConstantLr,
    module::Module,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use rand::{Rng, thread_rng};

/// 学習データをオンザフライで生成するデータセット。
///
/// 物理シミュレーションであるため、事前にデータファイルを用意する必要がなく、
/// 必要になるたびにランダムな周波数を生成します。
#[derive(Clone, Debug)]
pub struct TuningForkDataset {
    /// データセットの見かけ上のサイズ。
    pub size: usize,
    /// 生成する周波数の範囲 (min, max)。
    pub freq_range: (f32, f32),
}

impl Dataset<f32> for TuningForkDataset {
    /// データセットから一つのアイテム（周波数）を取得します。
    ///
    /// この実装では、呼ばれるたびに新しいランダムな周波数を返します。
    fn get(&self, _index: usize) -> Option<f32> {
        let mut rng = thread_rng();
        let frequency = rng.gen_range(self.freq_range.0..=self.freq_range.1);
        Some(frequency)
    }

    /// データセットの長さを返します。
    fn len(&self) -> usize {
        self.size
    }
}

/// データセットから取得したアイテムをミニバッチにまとめるバッチャ。
///
/// `f32`のスライスを、指定されたバックエンドのテンソルに変換します。
pub struct TuningForkBatcher<B: Backend> {
    _device: B::Device,
}

impl<B: Backend> TuningForkBatcher<B> {
    /// 新しいバッチャを作成します。
    pub fn new(device: B::Device) -> Self {
        Self { _device: device }
    }
}

impl<B: Backend> Batcher<B, f32, Tensor<B, 2>> for TuningForkBatcher<B> {
    /// `f32`のVecを`[batch_size, 1]`形状のテンソルに変換します。
    fn batch(&self, items: Vec<f32>, device: &B::Device) -> Tensor<B, 2> {
        let tensors: Vec<_> = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([*item], device))
            .collect();
        Tensor::cat(tensors, 0).reshape([-1, 1])
    }
}

/// モデルの学習ステップを定義します。
impl<B: AutodiffBackend> TrainStep<Tensor<B, 2>, RegressionOutput<B>> for TuningForkPINN<B> {
    /// 1回の学習ステップを実行します。
    ///
    /// 1. モデルによる予測
    /// 2. 物理法則に基づいた損失の計算
    /// 3. 勾配の計算と逆伝播
    fn step(&self, item: Tensor<B, 2>) -> TrainOutput<RegressionOutput<B>> {
        let predicted_dims = self.forward(item.clone());
        let loss = tuning_fork_loss(predicted_dims.clone(), item.clone());
        let output = RegressionOutput {
            loss: loss.clone(),
            output: predicted_dims,
            targets: item,
        };
        TrainOutput::new(self, loss.backward(), output)
    }
}

/// モデルの検証ステップを定義します。
impl<B: Backend> ValidStep<Tensor<B, 2>, RegressionOutput<B>> for TuningForkPINN<B> {
    /// 1回の検証ステップを実行します。
    ///
    /// 損失を計算し、学習の進捗をモニタリングします。
    fn step(&self, item: Tensor<B, 2>) -> RegressionOutput<B> {
        let predicted_dims = self.forward(item.clone());
        let loss = tuning_fork_loss(predicted_dims.clone(), item.clone());
        RegressionOutput {
            loss,
            output: predicted_dims,
            targets: item,
        }
    }
}

/// 学習プロセス全体の設定を保持します。
#[derive(Config)]
pub struct TrainingConfig {
    /// オプティマイザの設定。
    pub optimizer: AdamConfig,
    /// 学習率。
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    /// 学習エポック数。
    #[config(default = 10000)]
    pub num_epochs: usize,
    /// バッチサイズ。
    #[config(default = 1024)]
    pub batch_size: usize,
}

/// 学習プロセスを実行します。
///
/// # Type Parameters
///
/// * `B` - 学習に使用するバックエンド（例: `Autodiff<Wgpu>`、`Autodiff<NdArray>`）。
///
/// # Arguments
///
/// * `device` - 学習に使用するデバイス。
pub fn run<B: AutodiffBackend>(device: B::Device)
where
    B::InnerBackend: Backend,
{
    let config = TrainingConfig::new(AdamConfig::new());
    let artifact_dir = "./artifacts";

    // 学習用データローダー
    let batcher_train = TuningForkBatcher::<B>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(TuningForkDataset {
            size: config.batch_size * 100,
            freq_range: (200.0, 1800.0), // 学習用の周波数範囲
        });

    // 検証用データローダー
    let batcher_valid = TuningForkBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(TuningForkDataset {
            size: config.batch_size * 20,
            freq_range: (1800.0, 2000.0), // 検証用の周波数範囲
        });

    let scheduler = ConstantLr::new(config.learning_rate);

    // Learnerを構築
    let learner = LearnerBuilder::new(artifact_dir)
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            TuningForkPINN::<B>::new(&device),
            config.optimizer.init(),
            scheduler,
        );

    println!("🚀 Starting training on {:?}...", device);
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // 学習済みモデルを保存
    let model_record = model_trained.into_record();
    CompactRecorder::new()
        .record(model_record, format!("{artifact_dir}/model").into())
        .expect("Failed to save trained model");

    println!("\n✅ Model saved to '{artifact_dir}/model.mpk'");
}


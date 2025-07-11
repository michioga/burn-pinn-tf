//! # メインエントリーポイント
//!
//! コマンドライン引数を解析し、選択されたバックエンド（`ndarray`または`wgpu`）で
//! 学習または推論プロセスを開始します。

#![recursion_limit = "256"]

use burn::backend::{Autodiff, NdArray, wgpu::Wgpu};
use burn_tuningfork_pinn::{infer, train};
use clap::{Parser, Subcommand};

// デフォルトのバックエンド定義は不要になります

/// コマンドラインインターフェースの定義
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// 使用するバックエンドを指定します。
    ///
    /// `ndarray`または`wgpu`を選択できます。
    #[arg(long, default_value = "wgpu")]
    backend: String,
}

/// サブコマンド (`train` または `infer`)
#[derive(Subcommand, Debug)]
enum Commands {
    /// モデルを学習させます。
    Train,
    /// 学習済みモデルを使って推論します。
    Infer {
        /// 推論したい音叉の周波数 (Hz)
        #[arg(short, long)]
        freq: f32,
    },
}

/// 指定されたバックエンドでアクション（学習または推論）を実行するためのマクロ
macro_rules! run_action {
    ($backend:ty, $device:expr, $command:expr) => {
        match $command {
            Commands::Train => {
                println!("🚀 Starting training on {:?}...", $device);
                train::run::<Autodiff<$backend>>($device);
            }
            Commands::Infer { freq } => {
                println!(
                    "🔍 Inferring for frequency: {} Hz on {:?}...",
                    freq, $device
                );
                infer::run::<$backend>(freq, $device);
            }
        }
    };
}

/// アプリケーションのエントリーポイント
///
/// コマンドライン引数を解析し、指定されたサブコマンドとバックエンドに基づいて
/// 適切なアクション（学習または推論）を実行します。
fn main() {
    let cli = Cli::parse();

    match cli.backend.as_str() {
        "wgpu" => {
            let device = burn::backend::wgpu::WgpuDevice::default();
            run_action!(Wgpu, device, cli.command);
        }
        "ndarray" => {
            let device = burn::backend::ndarray::NdArrayDevice::default();
            run_action!(NdArray, device, cli.command);
        }
        _ => {
            panic!("❌ Invalid backend specified. Use 'wgpu' or 'ndarray'.");
        }
    }
}

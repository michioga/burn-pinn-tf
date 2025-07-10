//! # メインエントリーポイント
//!
//! コマンドライン引数を解析し、選択されたバックエンド（`ndarray`または`wgpu`）で
//! 学習または推論プロセスを開始します。

#![recursion_limit = "256"]

use burn::backend::{wgpu::Wgpu, Autodiff, NdArray};
use burn_tuningfork_pinn::{infer, train};
use clap::{Parser, Subcommand};

/// デフォルトのバックエンドをWGPUに設定します。
type DefaultBackend = Wgpu;

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

/// アプリケーションのエントリーポイント
///
/// コマンドライン引数を解析し、指定されたサブコマンドとバックエンドに基づいて
/// 適切なアクション（学習または推論）を実行します。
fn main() {
    let cli = Cli::parse();

    let device = burn::backend::wgpu::WgpuDevice::default();
    match cli.command {
        Commands::Train => {
            println!("🚀 Starting training...");
            train::run::<Autodiff<Wgpu>>(device);
        }
        Commands::Infer { freq } => {
            println!("🔍 Inferring for frequency: {} Hz ", freq);
            infer::run::<Wgpu>(freq, device);
        }
    }
}
//! # 音叉の物理情報ニューラルネットワーク (PINN)
//!
//! このプロジェクトは、[Burn](https://burn.dev/)フレームワークを使用して、
//! 物理情報ニューラルネットワーク（PINN）を実装するデモンストレーションです。
//!
//! 具体的には、音叉の目標周波数（例: 440Hz）を入力すると、
//! その周波数を実現するための物理的な寸法（柄の長さ、プロングの長さなど）を
//! ニューラルネットワークが出力します。
//!
//! 学習プロセスでは、物理法則を損失関数に組み込むことで、
//! モデルが物理的に妥当な解を学習するように誘導します。
//!
//! ## バックエンド
//!
//! このプロジェクトはCPUで動作する`ndarray`バックエンドを使用します。

#![recursion_limit = "256"]

use clap::{Parser, Subcommand};

mod constants;
mod infer;
mod model;
mod physics;
mod train;

/// コマンドラインインターフェースの定義
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train => {
            println!("🚀 Starting training...");
            train::run();
        }
        Commands::Infer { freq } => {
            println!("🔍 Inferring for frequency: {} Hz", freq);
            infer::run(freq);
        }
    }
}

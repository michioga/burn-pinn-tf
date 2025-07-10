use clap::{Parser, Subcommand};
// プロジェクト名（ライブラリ名）を使ってモジュールをインポート
use burn_tuningfork_pinn::{infer, train};

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

//! # 音叉の物理情報ニューラルネットワーク (PINN)
//!
//! このプロジェクトは、[Burn](https://burn.dev/)フレームワークを使用して、
//! 物理情報ニューラルネットワーク（PINN）を実装するデモンストレーションです。
//! (以下、ドキュメントコメントは省略)

// 各モジュールをライブラリの公開APIとして定義
pub mod constants;
pub mod infer;
pub mod model;
pub mod physics;
pub mod train;

//! # 定数モジュール
//!
//! このモジュールは、物理計算やモデル定義で使用される定数を定義します。

/// 物理計算に関する定数
pub mod physics {
    // 材質: ステンレス鋼 (SUS304) を想定
    /// ヤング率 (Pa)。材料の硬さを示す指標。
    pub const YOUNGS_MODULUS: f32 = 193.0e9;
    /// 密度 (kg/m^3)。
    pub const DENSITY: f32 = 8000.0;
    /// ポアソン比。材料が引張られた際の横方向の縮みを示す。今回は未使用。
    pub const POISSON_RATIO: f32 = 0.29; // ポアソン比

    /// 振動モードの係数
    pub const K_FACTOR: f32 = 3.5160;

    

    /// 損失関数におけるペナルティ項の重み。
    /// `ratio_penalty`（プロング長 > 柄長）に対する重み。
    pub const PENALTY_WEIGHT_RATIO: f32 = 0.5;
    /// `range_penalty`（プロング直径の範囲）に対する重み。
    pub const PENALTY_WEIGHT_RANGE: f32 = 10.0;
    /// `range_penalty`（その他の寸法の範囲）に対する重み。
    pub const PENALTY_WEIGHT_OTHER: f32 = 5.0;
}

/// モデルの寸法に関する定数
pub mod model_dims {
    /// 出力次元の総数
    pub const NUM_DIMS: usize = 5;
    /// 柄の長さのインデックス
    pub const HANDLE_LENGTH_IDX: usize = 0;
    /// 柄の直径のインデックス
    pub const HANDLE_DIAMETER_IDX: usize = 1;
    /// プロングの長さのインデックス
    pub const PRONG_LENGTH_IDX: usize = 2;
    /// プロングの直径のインデックス
    pub const PRONG_DIAMETER_IDX: usize = 3;
    /// プロングの間隔のインデックス
    pub const PRONG_GAP_IDX: usize = 4;
}

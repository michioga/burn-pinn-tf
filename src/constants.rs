//! # 物理定数モジュール
//!
//! このモジュールは、物理計算で使用される定数を定義します。
//! 主にステンレス鋼 (SUS304) の物性を想定しています。

pub mod physics {
    // 材質: ステンレス鋼 (SUS304) を想定
    /// ヤング率 (Pa)。材料の硬さを示す指標。
    pub const YOUNGS_MODULUS: f32 = 193.0e9;
    /// 密度 (kg/m^3)。
    pub const DENSITY: f32 = 8000.0;
    /// ポアソン比。材料が引張られた際の横方向の縮みを示す。今回は未使用。
    pub const POISSON_RATIO: f32 = 0.29; // ポアソン比

    /// 周波数計算式の係数 K。
    /// プロング（振動する腕）の形状や振動モード、境界条件によって決まる経験的な定数。
    /// この値はシミュレーションや実験を通じて調整される重要なハイパーパラメータです
    pub const FREQ_K_CONSTANT: f32 = 0.1615;

    /// 損失関数におけるペナルティ項の重み。
    /// `ratio_penalty`（プロング長 > 柄長）に対する重み。
    pub const PENALTY_WEIGHT_RATIO: f32 = 0.5;
    /// `range_penalty`（プロング直径の範囲）に対する重み。
    pub const PENALTY_WEIGHT_RANGE: f32 = 1.0;
}

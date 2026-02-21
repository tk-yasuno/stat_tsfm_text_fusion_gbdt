"""
Configuration file for HVAC Range Deviation Forecast v1.1 (Production)
空調設備レンジ逸脱予測システムの設定ファイル (64設備 + 最適構成)
v1.3実験の結果、v1.1構成が最良であることを確認
"""

import os
from pathlib import Path

# ===== プロジェクトパス =====
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"

# データディレクトリ
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
RANGES_DATA_DIR = DATA_ROOT / "ranges"

# ソースデータパス (v1.1: 全設備データ)
SOURCE_CSV_PATH = PROJECT_ROOT.parent / "data_source" / "251217_チェック項目_実施結果.csv"

# ===== データ処理パラメータ =====
# エンコーディング
CSV_ENCODING = "cp932"  # Windows Shift-JIS

# カラム名（日本語） - 全設備データ構造
COLUMNS = {
    "inspection_id": "点検ID",
    "equipment_category": "共通分類コード",  # 1:機械, 2:ポンプ, 3:空調
    "tenant_id": "テナントid",
    "equipment_id": "設備id",
    "check_item_id": "チェック項目id",
    "upper_limit": "上限値",
    "lower_limit": "下限値",
    "datetime": "実施日時",
    "value": "実施結果の値"  # 測定値（時系列データ）
}

# v1.1設備選定情報
SELECTED_EQUIPMENT_INFO = PROCESSED_DATA_DIR / "selected_64_equipment.json"

# 時系列パラメータ
LOOKBACK_DAYS = 90  # 過去参照長 D
FORECAST_HORIZONS = [30, 60, 90]  # 予測ホライズン h (日)
MIN_DATA_POINTS = 180  # 最小データポイント数（6ヶ月）

# 日次集計方法
AGGREGATION_METHOD = "mean"  # "mean", "median", "last"

# 設備フィルタ（v1.1: 全320設備の20%に相当する64設備）
# データ品質と変動性に基づいてTOP 64設備を選定
# Quality Score = CV × log(data_points) × log(date_range_days)
TARGET_EQUIPMENT_IDS = [
    265702, 265703, 265695, 315210, 265697, 265693, 265696, 265694,
    316620, 317132, 265698, 384187, 327244, 270352, 317136, 409052,
    409053, 327243, 317143, 376721, 327241, 327280, 322203, 315294,
    386663, 322220, 322034, 327242, 322196, 327279, 322219, 327869,
    327239, 327240, 327864, 265715, 386665, 322212, 327863, 327245,
    270330, 327246, 327875, 389359, 265769, 327883, 386664, 327855,
    265699, 327856, 267651, 265709, 265707, 265708, 265706, 265710,
    265777, 383648, 265775, 327881, 265776, 327857, 315153, 383643
]  # v1.0の5設備を含む

# ===== 正常レンジ定義 =====
# 分位点ベース（より厳格な範囲）
LOWER_PERCENTILE = 10  # 10th percentile
UPPER_PERCENTILE = 90  # 90th percentile

# 正常期間フィルタ（オプション）
# 明らかに正常だった期間を指定する場合
NORMAL_PERIOD_START = None  # "2023-01-01"
NORMAL_PERIOD_END = None    # "2023-12-31"

# ===== モデルパラメータ =====
# Granite Time Series
GRANITE_MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r1"  # TinyTimeMixer-512-96
CONTEXT_LENGTH = 512  # モデルの入力系列長
PREDICTION_LENGTH = 96  # モデルの予測長

# LoRA設定 (v1.1 最適構成 - 本番採用)
# v1.3でr=16を試したが過学習により性能悪化。r=8が最適。
LORA_CONFIG = {
    "r": 8,              # LoRAのランク (v1.3のr=16より優れている)
    "lora_alpha": 16,    # スケーリングファクター
    "target_modules": ["q_proj", "v_proj"],  # 適用するモジュール
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "SEQ_CLS"  # Sequence Classification
}

# ===== トレーニングパラメータ =====
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_epochs": 50,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # データ分割
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    
    # Early Stopping
    "patience": 5,
    "min_delta": 0.001,
    
    # クラス不均衡対応
    "use_class_weights": True,
    "focal_loss_gamma": 3.0,  # Focal Loss使用時（2.0→3.0に強化）
}

# ===== 推論パラメータ =====
INFERENCE_CONFIG = {
    "alert_threshold_warning": 0.7,   # 要注視
    "alert_threshold_critical": 0.9,  # 要対策
    "batch_size": 64,
}

# ===== 評価パラメータ =====
EVALUATION_METRICS = [
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1_score",
    "accuracy"
]

# リードタイム評価
LEADTIME_THRESHOLDS = [0.5, 0.7, 0.9]

# ===== ロギング =====
LOG_LEVEL = "INFO"
SAVE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 5  # エポックごと

# ===== 再現性 =====
RANDOM_SEED = 42

# ===== GPU設定 =====
USE_GPU = True
GPU_ID = 0

# ===== ディレクトリ作成 =====
def create_directories():
    """必要なディレクトリを作成"""
    dirs = [
        DATA_ROOT,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RANGES_DATA_DIR,
        MODEL_ROOT,
        RESULTS_ROOT,
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✓ プロジェクトディレクトリを作成しました")

if __name__ == "__main__":
    create_directories()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Source CSV: {SOURCE_CSV_PATH}")
    print(f"Lookback: {LOOKBACK_DAYS} days")
    print(f"Forecast Horizons: {FORECAST_HORIZONS} days")

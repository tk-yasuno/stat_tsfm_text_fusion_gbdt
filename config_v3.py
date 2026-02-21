"""
Configuration for MVP v3-0: Triplet Feature Learning for Equipment Anomaly Prediction
設備異常予測 トリプレット特徴学習 設定ファイル

v2-0との差分:
  + テキスト埋め込みモデル設定（設備名 / チェック項目）
  + FAISS インデックス設定
  + トリプレット特徴次元の管理
  + テキスト埋め込みキャッシュパス
"""

from config import *  # v2-0 の全設定を継承

# =====================================================================
# v3-0 追加: テキスト埋め込み設定
# =====================================================================

# ---- マスタCSVパス (v3-0) ------------------------------------------
# 設備id × チェック項目id → 設備名 / 設備分類 / チェック項目 のルックアップに使用
EQUIP_MASTER_CSV = (
    PROJECT_ROOT.parent / "data_source" / "251217_CSV_チェック項目_数値結果100件以上.csv"
)
EQUIP_MASTER_ENCODING = "cp932"

# マスタCSV内のカラム名
MASTER_COLS = {
    "equip_id":        "設備id",
    "equip_name":      "設備名",
    "equip_category":  "設備分類",
    "check_item_id":   "チェック項目id",
    "check_item_name": "チェック項目",
}

# enriched CSV 側の結合キーカラム名
COLUMNS_V3 = {
    **COLUMNS,  # v2-0 継承
    "equipment_id":    "equipment_id",   # enriched CSV の設備ID列
    "check_item_id":   "check_item_id",  # enriched CSV のチェック項目ID列
}

# ---- テキスト埋め込みモデル設定 ------------------------------------
# 'local'  : intfloat/multilingual-e5-large (1024 dim) — オフライン可
# 'openai' : text-embedding-3-large (1024 dim) — OpenAI API キー必須
TEXT_EMBED_BACKEND = "local"  # "local" | "openai"

# ローカルモデル名（HuggingFace Hub）
TEXT_EMBED_LOCAL_MODEL = "intfloat/multilingual-e5-large"

# OpenAI モデル名
TEXT_EMBED_OPENAI_MODEL   = "text-embedding-3-large"
OPENAI_EMBED_DIMENSIONS   = 1024  # text-embedding-3-large の次元数を削減可

# ローカルモデルの埋め込み次元
TEXT_EMBED_DIM = 1024  # multilingual-e5-large: 1024

# テキスト結合方法
#   'concat'   : [equip_embed; sensor_embed] → 2048 dim → PCA で 1024 に削減
#   'separate' : equip_embed (1024) と sensor_embed (1024) を別々にキャッシュ
#   'joint'    : "設備名 [SEP] チェック項目" をひとつのテキストとして埋め込み → 1024 dim
TEXT_COMBINE_STRATEGY = "joint"  # MVP は joint が最も単純

# ---- FAISS インデックス設定 ----------------------------------------
FAISS_N_NEIGHBORS = 5    # 近傍検索の k（MVP では参照用にのみ使用）
FAISS_USE_GPU     = False  # True にすると faiss-gpu が必要

# ---- キャッシュパス -------------------------------------------------
TEXT_EMBED_CACHE_DIR = PROCESSED_DATA_DIR / "text_embeddings"
FAISS_INDEX_PATH     = TEXT_EMBED_CACHE_DIR / "equip_category.faiss"
TEXT_EMBED_TRAIN_NPZ = TEXT_EMBED_CACHE_DIR / "train_text_embeddings.npz"
TEXT_EMBED_TEST_NPZ  = TEXT_EMBED_CACHE_DIR / "test_text_embeddings.npz"
TEXT_EMBED_UNIQUE_NPZ = TEXT_EMBED_CACHE_DIR / "unique_text_embeddings.npz"
TTM_EMBED_TRAIN_NPZ   = TEXT_EMBED_CACHE_DIR / "train_ttm_embeddings.npz"
TTM_EMBED_TEST_NPZ    = TEXT_EMBED_CACHE_DIR / "test_ttm_embeddings.npz"

# =====================================================================
# v3-0 特徴次元サマリー
# =====================================================================
STAT_FEATURE_DIM   = 28   # x ∈ ℝ²⁸  （統計特徴 — create_enriched_features.py）
TTM_EMBED_DIM      = 64   # y ∈ ℝ⁶⁴  （TinyTimeMixer + LoRA — granite_ts_model.py）
TEXT_Z_DIM         = TEXT_EMBED_DIM  # z ∈ ℝ¹⁰²⁴ （テキスト埋め込み）
TRIPLET_TOTAL_DIM  = STAT_FEATURE_DIM + TTM_EMBED_DIM + TEXT_Z_DIM  # 1116 dim

# =====================================================================
# v3-0 モデルパス
# =====================================================================
TRIPLET_MODEL_DIR  = MODEL_ROOT / "triplet_model"
TRIPLET_RESULTS_DIR = RESULTS_ROOT / "triplet_model"

# =====================================================================
# v3-0 LightGBM (Triplet Fusion Boosting Classifier) パラメータ
# =====================================================================
TRIPLET_LGBM_PARAMS = {
    'objective':       'binary',
    'metric':          'auc',
    'boosting_type':   'gbdt',
    'num_leaves':      63,       # v2-0 の 31 より増加（特徴量増加対応）
    'learning_rate':   0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':    5,
    'verbose':        -1,
    'min_child_samples': 20,
    'reg_alpha':       0.1,
    'reg_lambda':      0.1,
    # scale_pos_weight はクラス比率から実行時に決定
}
TRIPLET_NUM_BOOST_ROUND    = 1000
TRIPLET_EARLY_STOPPING     = 50
TRIPLET_LOG_EVAL_PERIOD    = 100
TRIPLET_EXTRACT_BATCH_SIZE = 128  # テキスト埋め込み抽出バッチサイズ

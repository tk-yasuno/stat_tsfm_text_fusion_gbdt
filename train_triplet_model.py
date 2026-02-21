"""
train_triplet_model.py  ―  MVP v3-0
Triplet Fusion Boosting Classifier for Equipment Anomaly Prediction
設備異常予測 トリプレット融合ブースティング分類器

━━━━ アーキテクチャ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  x ∈ ℝ²⁸    統計的特徴量（create_enriched_features.py で生成）
  y ∈ ℝ⁶⁴    TinyTimeMixer + LoRA 埋め込み（granite_ts_model.py）
  z ∈ ℝ¹⁰²⁴  テキスト埋め込み（text_embedding.py）
             └─ 設備名 × チェック項目 → multilingual-e5-large
  h = concat(x; y; z)  ∈ ℝ¹¹¹⁶  → LightGBM で {正常 / 異常} 確率出力
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

実行手順:
  1. python create_enriched_features.py   # x (統計特徴) 生成
  2. python text_embedding.py             # z (テキスト埋め込み) キャッシュ生成
  3. python train_triplet_model.py        # 本スクリプト（y + 結合 + 学習）

v2-0 との差分:
  + Text Embedding Path (z) が追加
  + 特徴量次元: 92 → 1116
  + 結果保存先: results/triplet_model/
"""

import sys
import os

# Granite TS 用の回避策：torchvision をスキップ
sys.modules["torchvision"] = None
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# v3-0 設定（v2-0 を継承）
from config_v3 import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    RANDOM_SEED,
    LOOKBACK_DAYS,
    USE_GPU,
    GPU_ID,
    TRIPLET_MODEL_DIR,
    TRIPLET_RESULTS_DIR,
    TRIPLET_LGBM_PARAMS,
    TRIPLET_NUM_BOOST_ROUND,
    TRIPLET_EARLY_STOPPING,
    TRIPLET_LOG_EVAL_PERIOD,
    STAT_FEATURE_DIM,
    TTM_EMBED_DIM,
    TEXT_Z_DIM,
    TRIPLET_TOTAL_DIM,
    TEXT_EMBED_TRAIN_NPZ,
    TEXT_EMBED_TEST_NPZ,
)

# Text Embedder
from text_embedding import TextEmbedder, precompute_all_embeddings

# Granite TS モデル（v2-0 と共通）
from granite_ts_model import GraniteTimeSeriesClassifier

# プロット設定
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


# =====================================================================
# データセット
# =====================================================================

class TripletDataset(Dataset):
    """
    Triplet Feature Dataset
    各サンプルに (時系列, 統計特徴, ラベル) を持つ。
    """

    def __init__(self, df: pd.DataFrame, stat_feature_cols: List[str]):
        self.df = df
        self.stat_feature_cols = stat_feature_cols

        # 時系列デコード
        import ast
        seqs = []
        for seq_str in df["values_sequence"].values:
            try:
                values = ast.literal_eval(str(seq_str))
            except Exception:
                values = [float(x.strip("[] ")) for x in str(seq_str).split(",") if x.strip()]

            # パディング / トリミング
            if len(values) < LOOKBACK_DAYS:
                values = [values[0]] * (LOOKBACK_DAYS - len(values)) + list(values)
            elif len(values) > LOOKBACK_DAYS:
                values = list(values)[-LOOKBACK_DAYS:]
            seqs.append(values)

        self.sequences = np.array(seqs, dtype=np.float32)
        self.stat_feats = df[stat_feature_cols].values.astype(np.float32)
        self.labels = {
            f"label_{h}d": df[f"label_{h}d"].values.astype(np.int64)
            for h in FORECAST_HORIZONS
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "sequence": torch.from_numpy(self.sequences[idx].reshape(-1, 1)),
            "stat_feats": torch.from_numpy(self.stat_feats[idx]),
            "labels": {k: v[idx] for k, v in self.labels.items()},
        }


# =====================================================================
# TripletFusionModel
# =====================================================================

class TripletFusionModel:
    """
    Triplet Fusion Boosting Classifier (v3-0)

    特徴量パイプライン:
      x = 統計特徴 (28)  ← create_enriched_features.py
      y = TTM 埋め込み (64) ← granite_ts_model.py + LoRA
      z = テキスト埋め込み (1024) ← text_embedding.py
      h = concat(x, y, z) → LightGBM
    """

    def __init__(self):
        self.use_gpu  = USE_GPU and torch.cuda.is_available()
        self.device   = torch.device(f"cuda:{GPU_ID}" if self.use_gpu else "cpu")
        self.ts_encoder: Optional[torch.nn.Module] = None
        self.lgbm_models: Dict[int, lgb.Booster]  = {}
        self.results: Dict[int, dict]              = {}
        self.stat_feature_cols: List[str]          = []

        TRIPLET_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        TRIPLET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. データロード
    # ------------------------------------------------------------------

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """enriched CSV を読み込む"""
        print("📂 Loading enriched data ...")

        train_path = PROCESSED_DATA_DIR / "training_samples_enriched.csv"
        test_path  = PROCESSED_DATA_DIR / "test_samples_enriched.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Enriched CSV が見つかりません。\n"
                "先に create_enriched_features.py を実行してください。"
            )

        self.train_df = pd.read_csv(train_path)
        self.test_df  = pd.read_csv(test_path)

        print(f"  ✓ Train: {len(self.train_df):,} samples")
        print(f"  ✓ Test : {len(self.test_df):,} samples")

        # 統計特徴カラムを特定（メタカラムを除外）
        exclude = {
            "equipment_id", "check_item_id", "date",
            "window_start", "window_end", "values_sequence",
            "label_current", "label_30d", "label_60d", "label_90d",
            "any_anomaly",
        }
        self.stat_feature_cols = [c for c in self.train_df.columns if c not in exclude]
        print(f"  ✓ Statistical feature cols: {len(self.stat_feature_cols)}")

        return self.train_df, self.test_df

    # ------------------------------------------------------------------
    # 2. テキスト埋め込み (z)
    # ------------------------------------------------------------------

    def load_text_embeddings(self, force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        テキスト埋め込みをキャッシュから読み込む。
        キャッシュが存在しない場合は text_embedding.py を使って生成する。
        """
        print("\n🔤 Loading text embeddings (z) ...")

        if (
            not force_recompute
            and TEXT_EMBED_TRAIN_NPZ.exists()
            and TEXT_EMBED_TEST_NPZ.exists()
        ):
            z_train = np.load(TEXT_EMBED_TRAIN_NPZ)["embeddings"].astype(np.float32)
            z_test  = np.load(TEXT_EMBED_TEST_NPZ)["embeddings"].astype(np.float32)
            print(f"  ✓ Loaded from cache: z_train {z_train.shape}, z_test {z_test.shape}")
        else:
            print("  ⚠ Cache not found. Computing embeddings now ...")
            print("    (初回は数分かかります)")
            _, z_train, z_test = precompute_all_embeddings(self.train_df, self.test_df)

        return z_train, z_test

    # ------------------------------------------------------------------
    # 3. TTM 埋め込み (y)
    # ------------------------------------------------------------------

    def build_ts_encoder(self):
        """Granite TS TinyTimeMixer Encoder をロード"""
        print("\n🤖 Building Granite TS TinyTimeMixer Encoder ...")
        try:
            model = GraniteTimeSeriesClassifier(
                num_horizons=len(FORECAST_HORIZONS),
                device=self.device,
            )
            if hasattr(model, "base_model"):
                self.ts_encoder = model.base_model
            elif hasattr(model, "model"):
                self.ts_encoder = model.model.base_model
            else:
                self.ts_encoder = None

            if self.ts_encoder is not None:
                self.ts_encoder.to(self.device)
                self.ts_encoder.eval()
                print(f"  ✓ Encoder ready. Device: {self.device}")
            else:
                print("  ⚠ Encoder 取得不可。ゼロベクトルで代替します。")

        except Exception as e:
            print(f"  ⚠ Encoder ロード失敗: {e}")
            print("    ゼロベクトルで代替します。")
            self.ts_encoder = None

    def extract_ttm_embeddings(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
    ) -> np.ndarray:
        """TinyTimeMixer から y ∈ ℝ⁶⁴ を抽出する"""
        if self.ts_encoder is None:
            print("  ⚠ Encoder なし → ゼロベクトルを返します。")
            return np.zeros((len(df), TTM_EMBED_DIM), dtype=np.float32)

        print(f"  Extracting TTM embeddings from {len(df):,} samples ...")
        dataset    = TripletDataset(df, self.stat_feature_cols)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        embeddings = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                seqs = batch["sequence"].to(self.device)  # (B, L, 1)
                try:
                    outputs = self.ts_encoder(
                        past_values=seqs,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    if (
                        hasattr(outputs, "backbone_hidden_state")
                        and outputs.backbone_hidden_state is not None
                    ):
                        bh = outputs.backbone_hidden_state  # (B, 1, P, D)
                        hidden = bh.squeeze(1).mean(dim=1)  # (B, D)
                    else:
                        hidden = seqs.mean(dim=1).squeeze(-1)
                        hidden = hidden.unsqueeze(-1).expand(-1, TTM_EMBED_DIM)
                except Exception:
                    hidden = torch.zeros(seqs.size(0), TTM_EMBED_DIM, device=self.device)

                embeddings.append(hidden.cpu().numpy())

                if (i + 1) % 20 == 0:
                    done = min((i + 1) * batch_size, len(df))
                    print(f"    {done:,} / {len(df):,}")

        y = np.vstack(embeddings).astype(np.float32)
        print(f"  ✓ TTM embeddings: {y.shape}")
        return y

    # ------------------------------------------------------------------
    # 4. トリプレット特徴の結合 h = [x; y; z]
    # ------------------------------------------------------------------

    def prepare_triplet_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        x (統計), y (TTM), z (テキスト) を結合してトリプレット特徴を構築。

        Returns:
            (X_train, X_test)  各 shape: (N, TRIPLET_TOTAL_DIM)
        """
        print("\n⚡ Building Triplet Feature h = [x; y; z] ...")

        # x: 統計特徴
        x_train = self.train_df[self.stat_feature_cols].values.astype(np.float32)
        x_test  = self.test_df[self.stat_feature_cols].values.astype(np.float32)
        print(f"  x (stats)  : train {x_train.shape}, test {x_test.shape}")

        # y: TTM 埋め込み
        from config_v3 import TTM_EMBED_TRAIN_NPZ, TTM_EMBED_TEST_NPZ
        TTM_EMBED_TRAIN_NPZ.parent.mkdir(parents=True, exist_ok=True)

        if TTM_EMBED_TRAIN_NPZ.exists():
            print("  📂 Loading cached TTM train embeddings ...")
            y_train = np.load(TTM_EMBED_TRAIN_NPZ)["embeddings"].astype(np.float32)
        else:
            y_train = self.extract_ttm_embeddings(self.train_df)
            np.savez_compressed(TTM_EMBED_TRAIN_NPZ, embeddings=y_train)
            print(f"  💾 Saved TTM train cache → {TTM_EMBED_TRAIN_NPZ.name}")

        if TTM_EMBED_TEST_NPZ.exists():
            print("  📂 Loading cached TTM test embeddings  ...")
            y_test = np.load(TTM_EMBED_TEST_NPZ)["embeddings"].astype(np.float32)
        else:
            y_test = self.extract_ttm_embeddings(self.test_df)
            np.savez_compressed(TTM_EMBED_TEST_NPZ, embeddings=y_test)
            print(f"  💾 Saved TTM test  cache → {TTM_EMBED_TEST_NPZ.name}")

        print(f"  y (TTM)    : train {y_train.shape}, test {y_test.shape}")

        # z: テキスト埋め込み
        z_train, z_test = self.load_text_embeddings()
        print(f"  z (text)   : train {z_train.shape}, test {z_test.shape}")

        # サイズ整合チェック
        for name, x, y, z in [("train", x_train, y_train, z_train),
                               ("test",  x_test,  y_test,  z_test)]:
            n = len(self.train_df) if name == "train" else len(self.test_df)
            assert len(x) == n, f"Stats size mismatch in {name}"
            assert len(y) == n, f"TTM   size mismatch in {name}"
            assert len(z) == n, f"Text  size mismatch in {name}: {len(z)} != {n}"

        # 結合
        self.X_train = np.hstack([x_train, y_train, z_train])
        self.X_test  = np.hstack([x_test,  y_test,  z_test])

        print(f"\n  ✓ Triplet features ready:")
        print(f"    Train : {self.X_train.shape}  (= {x_train.shape[1]} + {y_train.shape[1]} + {z_train.shape[1]})")
        print(f"    Test  : {self.X_test.shape}")
        print(f"    Total dim: {self.X_train.shape[1]} (expected {TRIPLET_TOTAL_DIM})")

        return self.X_train, self.X_test

    # ------------------------------------------------------------------
    # 5. 学習（各ホライズン）
    # ------------------------------------------------------------------

    def _get_lgbm_params(self, pos_weight: float) -> Dict:
        params = dict(TRIPLET_LGBM_PARAMS)
        params["scale_pos_weight"] = pos_weight
        params["random_state"]     = RANDOM_SEED
        return params

    def _build_feature_names(self) -> List[str]:
        """特徴量名リストを生成"""
        names = list(self.stat_feature_cols)
        names += [f"ttm_{i}" for i in range(TTM_EMBED_DIM)]
        names += [f"text_{i}" for i in range(TEXT_Z_DIM)]
        return names

    def train_horizon(self, horizon: int) -> Tuple[lgb.Booster, Dict]:
        """特定ホライズンの LightGBM を学習"""
        print(f"\n{'='*70}")
        print(f"  Training Triplet Fusion Classifier — {horizon}d horizon")
        print(f"{'='*70}")

        label_col = f"label_{horizon}d"
        y_train   = self.train_df[label_col].values
        y_test    = self.test_df[label_col].values

        pos_rate = y_train.mean()
        pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        print(f"  Train pos: {y_train.sum():,} / {len(y_train):,}  ({pos_rate*100:.1f}%)")
        print(f"  Test  pos: {y_test.sum():,}  / {len(y_test):,}   ({y_test.mean()*100:.1f}%)")
        print(f"  pos_weight: {pos_weight:.2f}")

        feature_names = self._build_feature_names()
        params = self._get_lgbm_params(pos_weight)

        train_data = lgb.Dataset(self.X_train, label=y_train, feature_name=feature_names)
        test_data  = lgb.Dataset(self.X_test,  label=y_test,  reference=train_data,
                                 feature_name=feature_names)

        print(f"\n  🚀 Training LightGBM ...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=TRIPLET_NUM_BOOST_ROUND,
            valid_sets=[train_data, test_data],
            valid_names=["train", "test"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=TRIPLET_EARLY_STOPPING),
                lgb.log_evaluation(period=TRIPLET_LOG_EVAL_PERIOD),
            ],
        )

        self.lgbm_models[horizon] = model

        # 評価
        y_pred_proba = model.predict(self.X_test, num_iteration=model.best_iteration)

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = (
            2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        )
        opt_idx   = np.argmax(f1_scores)
        opt_thr   = thresholds[opt_idx]
        y_pred    = (y_pred_proba > opt_thr).astype(int)

        metrics = {
            "horizon":           horizon,
            "model":             "TripletFusion",
            "optimal_threshold": float(opt_thr),
            "accuracy":          float(accuracy_score(y_test, y_pred)),
            "precision":         float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":            float(recall_score(y_test, y_pred, zero_division=0)),
            "f1":                float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc":           float(roc_auc_score(y_test, y_pred_proba)),
            "pr_auc":            float(average_precision_score(y_test, y_pred_proba)),
            "best_iteration":    model.best_iteration,
            "triplet_dim":       self.X_train.shape[1],
        }

        self.results[horizon] = {
            "metrics":     metrics,
            "predictions": y_pred_proba,
            "labels":      y_test,
        }

        print(f"\n  📊 {horizon}d Results:")
        print(f"    Threshold : {opt_thr:.4f}")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    F1-Score  : {metrics['f1']:.4f}")
        print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"    PR-AUC    : {metrics['pr_auc']:.4f}")

        return model, metrics

    def train_all_horizons(self) -> pd.DataFrame:
        """全ホライズンの学習"""
        print("\n" + "="*70)
        print("🚀 Triplet Fusion Boosting Classifier — Training All Horizons")
        print("="*70)
        print(f"    特徴次元 h = [x:{STAT_FEATURE_DIM} | y:{TTM_EMBED_DIM} | z:{TEXT_Z_DIM}]")

        all_metrics = []
        for h in FORECAST_HORIZONS:
            _, m = self.train_horizon(h)
            all_metrics.append(m)

        metrics_df = pd.DataFrame(all_metrics)
        print("\n" + "="*70)
        print("📊 Summary — Triplet Fusion Classifier")
        print("="*70)
        print(metrics_df[["horizon", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))

        return metrics_df

    # ------------------------------------------------------------------
    # 6. 保存
    # ------------------------------------------------------------------

    def save_models(self):
        """モデルと評価結果を保存"""
        print(f"\n💾 Saving models → {TRIPLET_MODEL_DIR}")

        for h, model in self.lgbm_models.items():
            path = TRIPLET_MODEL_DIR / f"lgbm_triplet_{h}d.txt"
            model.save_model(str(path))
            print(f"  ✓ {h}d model: {path.name}")

        # メトリクス CSV
        rows  = [self.results[h]["metrics"] for h in FORECAST_HORIZONS if h in self.results]
        m_df  = pd.DataFrame(rows)
        m_path = TRIPLET_RESULTS_DIR / "metrics_summary.csv"
        m_df.to_csv(m_path, index=False, encoding="utf-8-sig")
        print(f"  ✓ Metrics: {m_path}")

        # JSON 形式でも保存
        j_path = TRIPLET_RESULTS_DIR / "metrics_summary.json"
        with open(j_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"  ✓ JSON: {j_path}")

    # ------------------------------------------------------------------
    # 7. 可視化
    # ------------------------------------------------------------------

    def plot_results(self):
        """特徴量重要度 Top50 を可視化（horizon ごとに横並び）"""
        import matplotlib.patches as mpatches
        print(f"\nPlotting results -> {TRIPLET_RESULTS_DIR}")

        TOP_N = 50
        BAR_H = 0.55
        n_horizons = len(FORECAST_HORIZONS)
        feature_names = self._build_feature_names()

        fig_h = TOP_N * BAR_H * 0.14 + 2.5
        fig, axes = plt.subplots(1, n_horizons, figsize=(8 * n_horizons, fig_h))
        if n_horizons == 1:
            axes = [axes]

        legend_handles = [
            mpatches.Patch(color="#4CAF50", label="Text embedding"),
            mpatches.Patch(color="#2196F3", label="TTM embedding"),
            mpatches.Patch(color="#FF9800", label="Statistical feature"),
        ]

        for col_idx, h in enumerate(FORECAST_HORIZONS):
            ax    = axes[col_idx]
            model = self.lgbm_models.get(h)
            if model is None or h not in self.results:
                ax.axis("off")
                continue

            importance = model.feature_importance(importance_type="gain")
            top_idx    = np.argsort(importance)[-TOP_N:][::-1]
            top_names  = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
                          for i in top_idx]
            top_vals   = importance[top_idx]

            colors = []
            for nm in top_names:
                if nm.startswith("text_"):
                    colors.append("#4CAF50")   # green : text embedding
                elif nm.startswith("ttm_"):
                    colors.append("#2196F3")   # blue  : TTM embedding
                else:
                    colors.append("#FF9800")   # orange: statistical feature

            ax.barh(range(TOP_N), top_vals[::-1], color=colors[::-1], height=BAR_H)
            ax.set_yticks(range(TOP_N))
            ax.set_yticklabels(top_names[::-1], fontsize=7)
            ax.set_ylim(-0.5, TOP_N - 0.5)   # 上下の隙間を詰める

            m = self.results[h]["metrics"]
            ax.set_title(
                f"{h}d Feature Importance Top{TOP_N}\n"
                f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  "
                f"F1={m['f1']:.3f}  AUC={m['roc_auc']:.4f}",
                fontsize=10,
            )
            ax.set_xlabel("Gain")
            # 各サブプロットの右下に凡例
            ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

        plt.suptitle("Triplet Fusion Boosting Classifier (v3-0) - Feature Importance",
                     fontsize=13, y=1.005)
        plt.tight_layout()

        fig_path = TRIPLET_RESULTS_DIR / "triplet_model_evaluation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Feature importance figure saved: {fig_path}")

    # ------------------------------------------------------------------
    # 8. v2-0 との比較
    # ------------------------------------------------------------------

    def compare_with_v2(self) -> pd.DataFrame:
        """v2-0（Hybrid）の結果と比較"""
        print(f"\n📊 Comparing v3-0 vs v2-0 ...")

        rows = []

        # v2-0 ハイブリッドの結果を読み込み
        v2_path = RESULTS_ROOT / "hybrid_model" / "metrics_summary.csv"
        if v2_path.exists():
            v2_df = pd.read_csv(v2_path)
            for _, r in v2_df.iterrows():
                rows.append({
                    "Model":     "v2-0 Hybrid (92-dim)",
                    "Horizon":   f"{int(r['horizon'])}d",
                    "Precision": r.get("precision", float("nan")),
                    "Recall":    r.get("recall", float("nan")),
                    "F1":        r.get("f1", float("nan")),
                    "ROC-AUC":   r.get("roc_auc", float("nan")),
                })
        else:
            print("  ⚠ v2-0 metrics not found. Skipping comparison.")

        # v3-0 の結果
        for h in FORECAST_HORIZONS:
            if h not in self.results:
                continue
            m = self.results[h]["metrics"]
            rows.append({
                "Model":     f"v3-0 Triplet (1116-dim)",
                "Horizon":   f"{h}d",
                "Precision": m["precision"],
                "Recall":    m["recall"],
                "F1":        m["f1"],
                "ROC-AUC":   m["roc_auc"],
            })

        comparison_df = pd.DataFrame(rows)
        cmp_path = TRIPLET_RESULTS_DIR / "comparison_v2_vs_v3.csv"
        comparison_df.to_csv(cmp_path, index=False, encoding="utf-8-sig")

        print("\n" + "="*70)
        print("Model Comparison: v2-0 Hybrid vs v3-0 Triplet Fusion")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print(f"\n  💾 Saved: {cmp_path}")
        return comparison_df


# =====================================================================
# エントリーポイント
# =====================================================================

def main():
    print("=" * 70)
    print("MVP v3-0: Triplet Feature Learning for Equipment Anomaly Prediction")
    print("  Architecture: x(28) + y(64) + z(1024) → LightGBM")
    print("=" * 70)
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model_obj = TripletFusionModel()

    # Step 1: データロード
    model_obj.load_data()

    # Step 2: Granite TS Encoder ロード（y の抽出に使用）
    model_obj.build_ts_encoder()

    # Step 3: トリプレット特徴の構築 [x; y; z]
    model_obj.prepare_triplet_features()

    # Step 4: 全ホライズンで学習
    metrics_df = model_obj.train_all_horizons()

    # Step 5: 保存
    model_obj.save_models()

    # Step 6: 可視化
    model_obj.plot_results()

    # Step 7: v2-0 との比較
    model_obj.compare_with_v2()

    print("\n" + "=" * 70)
    print(f"✅ v3-0 Training Completed! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"  Models  → {TRIPLET_MODEL_DIR}")
    print(f"  Results → {TRIPLET_RESULTS_DIR}")


if __name__ == "__main__":
    main()

"""
evaluate_v3.py  ―  MVP v3-0 評価スクリプト
Triplet Fusion Boosting Classifier の保存済みモデルを評価する。

使い方:
  python evaluate_v3.py
  python evaluate_v3.py --horizon 30
"""

import sys
import os

sys.modules["torchvision"] = None
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from config_v3 import (
    PROCESSED_DATA_DIR,
    FORECAST_HORIZONS,
    TRIPLET_MODEL_DIR,
    TRIPLET_RESULTS_DIR,
    STAT_FEATURE_DIM,
    TTM_EMBED_DIM,
    TEXT_Z_DIM,
    TRIPLET_TOTAL_DIM,
    TEXT_EMBED_TEST_NPZ,
    TTM_EMBED_TEST_NPZ,
)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


# =====================================================================
# ヘルパー
# =====================================================================

def _load_test_features() -> tuple:
    """
    評価用の特徴量を再構築する（save_models() が保存した CSV がある場合はそれを使用）。
    """
    # まず保存済みの特徴量 CSV を探す
    features_path = PROCESSED_DATA_DIR / "test_features_triplet.csv"
    labels_path   = PROCESSED_DATA_DIR / "test_labels.csv"

    if features_path.exists() and labels_path.exists():
        print(f"  📂 Loading saved test features: {features_path.name}")
        X_test = pd.read_csv(features_path).values.astype(np.float32)
        y_dict = {f"label_{h}d": pd.read_csv(labels_path)[f"label_{h}d"].values
                  for h in FORECAST_HORIZONS}
        return X_test, y_dict

    # 保存済みがなければ、enriched CSV + text embedding キャッシュから再構築
    print("  📂 Rebuilding test features from enriched CSV + embedding cache ...")
    test_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            "test_samples_enriched.csv が見つかりません。\n"
            "create_enriched_features.py を実行してください。"
        )

    test_df = pd.read_csv(test_path)

    exclude = {
        "equipment_id", "check_item_id", "date",
        "window_start", "window_end", "values_sequence",
        "label_current", "label_30d", "label_60d", "label_90d",
        "any_anomaly",
    }
    stat_cols = [c for c in test_df.columns if c not in exclude]
    x_test = test_df[stat_cols].values.astype(np.float32)

    # TTM 埋め込み（train_triplet_model.py が保存したキャッシュを使用）
    if TTM_EMBED_TEST_NPZ.exists():
        y_test = np.load(TTM_EMBED_TEST_NPZ)["embeddings"].astype(np.float32)
        print(f"  📂 Loaded TTM test embeddings: {y_test.shape}")
    else:
        print("  ⚠ TTM埋め込みキャッシュなし → ゼロで代替。train_triplet_model.py を先に実行してください。")
        y_test = np.zeros((len(test_df), TTM_EMBED_DIM), dtype=np.float32)

    # テキスト埋め込み
    if TEXT_EMBED_TEST_NPZ.exists():
        z_test = np.load(TEXT_EMBED_TEST_NPZ)["embeddings"].astype(np.float32)
    else:
        print("  ⚠ テキスト埋め込みキャッシュなし → ゼロで代替します。")
        z_test = np.zeros((len(test_df), TEXT_Z_DIM), dtype=np.float32)

    X_test = np.hstack([x_test, y_test, z_test])
    y_dict = {f"label_{h}d": test_df[f"label_{h}d"].values for h in FORECAST_HORIZONS}
    return X_test, y_dict


def _load_model(horizon: int) -> lgb.Booster:
    model_path = TRIPLET_MODEL_DIR / f"lgbm_triplet_{horizon}d.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルなし: {model_path}\n先に train_triplet_model.py を実行してください。")
    return lgb.Booster(model_file=str(model_path))


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1  = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
    return float(thr[np.argmax(f1)])


# =====================================================================
# 評価メイン
# =====================================================================

def evaluate(horizons=None):
    if horizons is None:
        horizons = FORECAST_HORIZONS

    TRIPLET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MVP v3-0 Evaluation — Triplet Fusion Boosting Classifier")
    print("=" * 70)

    X_test, y_dict = _load_test_features()
    print(f"  Test feature shape: {X_test.shape}")
    print(f"  (expected TRIPLET_TOTAL_DIM = {TRIPLET_TOTAL_DIM})")

    all_metrics = []
    all_results = {}

    for h in horizons:
        print(f"\n{'─'*60}")
        print(f"  Evaluating {h}d horizon ...")

        model  = _load_model(h)
        y_true = y_dict[f"label_{h}d"]

        y_prob = model.predict(X_test, num_iteration=model.best_iteration)
        opt_thr = _optimal_threshold(y_true, y_prob)
        y_pred  = (y_prob > opt_thr).astype(int)

        m = {
            "horizon":           h,
            "optimal_threshold": opt_thr,
            "accuracy":          float(accuracy_score(y_true, y_pred)),
            "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc":           float(roc_auc_score(y_true, y_prob)),
            "pr_auc":            float(average_precision_score(y_true, y_prob)),
        }
        all_metrics.append(m)
        all_results[h] = {"metrics": m, "y_true": y_true, "y_prob": y_prob, "model": model}

        cm = confusion_matrix(y_true, y_pred)
        fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
        tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0
        tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
        fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
        total_neg = tn + fp

        print(f"  Threshold : {opt_thr:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1-Score  : {m['f1']:.4f}")
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"  PR-AUC    : {m['pr_auc']:.4f}")
        print(f"  False Positive Rate: {fp/total_neg*100:.1f}% ({fp} / {total_neg})" if total_neg else "")
        print(f"  True Positive Rate : {tp/(tp+fn)*100:.1f}% ({tp} / {tp+fn})" if (tp+fn) else "")

    # ---- サマリー表示 ----
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    df = pd.DataFrame(all_metrics)
    print(df[["horizon", "precision", "recall", "f1", "roc_auc", "pr_auc"]].to_string(index=False))

    # ---- 保存 ----
    m_path = TRIPLET_RESULTS_DIR / "eval_metrics.csv"
    df.to_csv(m_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Saved: {m_path}")

    # ---- 可視化 ----
    _plot_evaluation(all_results, horizons)


def _plot_evaluation(all_results: dict, horizons: list):
    fig, axes = plt.subplots(2, len(horizons), figsize=(7 * len(horizons), 12))
    if len(horizons) == 1:
        axes = axes.reshape(2, 1)

    for col, h in enumerate(horizons):
        res    = all_results[h]
        y_true = res["y_true"]
        y_prob = res["y_prob"]
        m      = res["metrics"]
        y_pred = (y_prob > m["optimal_threshold"]).astype(int)

        # 混同行列
        ax = axes[0, col]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
        ax.set_title(f"{h}d Confusion Matrix\nPrec={m['precision']:.3f} Rec={m['recall']:.3f} F1={m['f1']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        # ROC Curve
        ax = axes[1, col]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"ROC AUC={m['roc_auc']:.4f}", color="steelblue", lw=2)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_title(f"{h}d ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

    plt.suptitle("v3-0 Triplet Fusion Classifier — Evaluation", fontsize=13, y=1.01)
    plt.tight_layout()

    fig_path = TRIPLET_RESULTS_DIR / "eval_plots.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Figure saved: {fig_path}")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVP v3-0 評価スクリプト")
    parser.add_argument("--horizon", type=int, default=None,
                        help="評価するホライズン (30, 60, 90)。省略時は全て。")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else FORECAST_HORIZONS
    evaluate(horizons=horizons)

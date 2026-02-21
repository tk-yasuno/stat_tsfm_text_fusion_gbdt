"""
Text Embedding Module for MVP v3-0
テキスト埋め込みモジュール（設備名 × チェック項目 → z ∈ ℝ¹⁰²⁴）

機能:
  1. ローカルモデル (intfloat/multilingual-e5-large, 1024 dim) で日本語テキストを埋め込み
  2. OpenAI API (text-embedding-3-large) による代替バックエンド
  3. FAISS インデックス構築（設備カテゴリの意味空間の安定化）
  4. 埋め込みのキャッシュ保存／読み込み（再実行コスト削減）

利用する設定 (config_v3.py):
  TEXT_EMBED_BACKEND  : "local" or "openai"
  TEXT_EMBED_LOCAL_MODEL  : HuggingFace モデル名
  TEXT_COMBINE_STRATEGY   : "joint" | "concat" | "separate"
  TEXT_EMBED_DIM          : z の次元数 (1024)
  FAISS_N_NEIGHBORS       : FAISS 検索時の k

使い方:
  embedder = TextEmbedder()
  z_train = embedder.get_embeddings(train_df)   # (N, 1024)
  embedder.build_faiss_index(z_unique)           # 設備ユニーク埋め込みでインデックス構築
  similar_ids, distances = embedder.faiss_search(query_vec, k=5)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from config_v3 import (
    TEXT_EMBED_BACKEND,
    TEXT_EMBED_LOCAL_MODEL,
    TEXT_EMBED_OPENAI_MODEL,
    OPENAI_EMBED_DIMENSIONS,
    TEXT_EMBED_DIM,
    TEXT_COMBINE_STRATEGY,
    FAISS_N_NEIGHBORS,
    FAISS_USE_GPU,
    TEXT_EMBED_CACHE_DIR,
    FAISS_INDEX_PATH,
    TEXT_EMBED_TRAIN_NPZ,
    TEXT_EMBED_TEST_NPZ,
    TEXT_EMBED_UNIQUE_NPZ,
    TRIPLET_EXTRACT_BATCH_SIZE,
    COLUMNS_V3,
    EQUIP_MASTER_CSV,
    EQUIP_MASTER_ENCODING,
    MASTER_COLS,
)


# =====================================================================
# マスタCSV ルックアップ辞書
# =====================================================================

_MASTER_LOOKUP: dict = {}   # (equip_id, check_item_id) -> text
_UNIQUE_TEXTS:  dict = {}   # (equip_id, check_item_id) -> text  (same, used for FAISS)


def load_master_lookup(force: bool = False) -> dict:
    """
    251217_CSV_チェック項目_数値結果100件以上.csv を読み込み
    (equipment_id, check_item_id) → ユニークテキスト の辞書を返す。
    """
    global _MASTER_LOOKUP
    if _MASTER_LOOKUP and not force:
        return _MASTER_LOOKUP

    if not EQUIP_MASTER_CSV.exists():
        print(f"  ⚠ マスタCSV が見つかりません: {EQUIP_MASTER_CSV}")
        print("    プレースホルダーテキストを使用します。")
        return {}

    print(f"  📂 Loading equipment master: {EQUIP_MASTER_CSV.name}")
    mdf = pd.read_csv(EQUIP_MASTER_CSV, encoding=EQUIP_MASTER_ENCODING,
                      usecols=list(MASTER_COLS.values()),
                      dtype={MASTER_COLS['equip_id']: str,
                             MASTER_COLS['check_item_id']: str})

    lookup = {}
    for _, row in mdf.iterrows():
        eid  = str(row[MASTER_COLS['equip_id']]).strip()
        cid  = str(row[MASTER_COLS['check_item_id']]).strip()
        name = str(row.get(MASTER_COLS['equip_name'],   '') or '').strip()
        cat  = str(row.get(MASTER_COLS['equip_category'],'') or '').strip()
        item = str(row.get(MASTER_COLS['check_item_name'],'') or '').strip()
        # multilingual-e5-large 推奨フォーマット
        text = f"passage: {cat} {name} {item}".strip()
        lookup[(eid, cid)] = text

    _MASTER_LOOKUP = lookup
    print(f"  ✓ Lookup built: {len(lookup):,} (equip_id, check_item_id) pairs")
    return lookup


def _build_text_from_ids(equip_id, check_item_id, lookup: dict) -> str:
    """設備ID × チェック項目ID からテキストを生成"""
    key = (str(equip_id).strip(), str(check_item_id).strip())
    return lookup.get(key, f"passage: 空調設備 設備{equip_id} 項目{check_item_id}")


# =====================================================================
# ユーティリティ
# =====================================================================


def _normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 正規化（コサイン類似度に対応）"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return vecs / norms


# =====================================================================
# TextEmbedder クラス
# =====================================================================

class TextEmbedder:
    """
    設備テキストから意味的埋め込みを生成するクラス。

    Example:
        embedder = TextEmbedder()
        z = embedder.get_embeddings(df)   # shape (N, 1024)
        embedder.build_faiss_index(z_unique, labels=unique_ids)
        neighbor_ids, dists = embedder.faiss_search(z[0], k=3)
    """

    def __init__(self, backend: str = TEXT_EMBED_BACKEND):
        self.backend = backend
        self.model   = None
        self.tokenizer = None
        self.faiss_index = None
        self.faiss_labels: Optional[List[str]] = None
        TEXT_EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        print(f"📝 TextEmbedder backend: {self.backend}")

    # ------------------------------------------------------------------
    # モデルロード
    # ------------------------------------------------------------------

    def _load_local_model(self):
        """intfloat/multilingual-e5-large をロード"""
        if self.model is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        import torch

        print(f"  Loading local model: {TEXT_EMBED_LOCAL_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBED_LOCAL_MODEL)
        self.model     = AutoModel.from_pretrained(TEXT_EMBED_LOCAL_MODEL)
        self.model.eval()

        # GPU 利用可能なら GPU に移動
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        print(f"  Device: {self._device}")

    def _encode_local(self, texts: List[str], batch_size: int) -> np.ndarray:
        """ローカルモデルで埋め込みを生成 (average-pooling + L2-norm)"""
        import torch

        self._load_local_model()
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                output = self.model(**encoded)
                # average pooling over tokens
                attention_mask = encoded["attention_mask"]
                token_embs = output.last_hidden_state  # (B, T, H)
                mask_exp   = attention_mask.unsqueeze(-1).float()
                vecs = (token_embs * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)
                all_vecs.append(vecs.cpu().numpy())

            if (i // batch_size + 1) % 10 == 0:
                print(f"    {i + len(batch):,} / {len(texts):,} encoded")

        return _normalize(np.vstack(all_vecs).astype(np.float32))

    def _encode_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        """OpenAI API で埋め込みを生成"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai パッケージが必要です: pip install openai")

        client  = OpenAI()
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp  = client.embeddings.create(
                model=TEXT_EMBED_OPENAI_MODEL,
                input=batch,
                dimensions=OPENAI_EMBED_DIMENSIONS,
            )
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            all_vecs.append(vecs)

            if (i // batch_size + 1) % 5 == 0:
                print(f"    {i + len(batch):,} / {len(texts):,} encoded")

        return _normalize(np.vstack(all_vecs))

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def encode_texts(self, texts: List[str],
                     batch_size: int = TRIPLET_EXTRACT_BATCH_SIZE) -> np.ndarray:
        """
        テキストリストを埋め込みベクトルに変換。

        Args:
            texts     : 埋め込むテキストのリスト
            batch_size: 推論バッチサイズ

        Returns:
            vecs: shape (N, TEXT_EMBED_DIM), float32, L2 正規化済み
        """
        if self.backend == "local":
            return self._encode_local(texts, batch_size)
        elif self.backend == "openai":
            return self._encode_openai(texts, batch_size)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def get_embeddings(
        self,
        df: pd.DataFrame,
        equip_col:  str = COLUMNS_V3["equipment_id"],
        sensor_col: str = COLUMNS_V3["check_item_id"],
        cache_path: Optional[Path] = None,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """
        DataFrame の各行について (equipment_id, check_item_id) から
        マスタCSVを引いて z ∈ ℝ¹⁰²⁴ を生成。

        キャッシュ (.npz) が存在する場合はそれを返す。

        Args:
            df             : training_samples_enriched.csv などのデータフレーム
            equip_col      : 設備IDカラム名 (default: 'equipment_id')
            sensor_col     : チェック項目IDカラム名 (default: 'check_item_id')
            cache_path     : キャッシュ保存先 (.npz)
            force_recompute: True の場合キャッシュを無視して再計算

        Returns:
            z: shape (N, TEXT_EMBED_DIM)
        """
        # キャッシュ確認
        if cache_path is not None and cache_path.exists() and not force_recompute:
            print(f"  📂 Loading cached embeddings from {cache_path.name}")
            return np.load(cache_path)["embeddings"].astype(np.float32)

        # マスタルックアップテーブルをロード
        lookup = load_master_lookup()

        # (equipment_id, check_item_id) → テキスト
        print(f"  Building texts for {len(df):,} rows via master lookup ...")
        if equip_col in df.columns and sensor_col in df.columns:
            texts = [
                _build_text_from_ids(row[equip_col], row[sensor_col], lookup)
                for _, row in df[[equip_col, sensor_col]].iterrows()
            ]
        else:
            missing = [c for c in [equip_col, sensor_col] if c not in df.columns]
            print(f"  ⚠ Columns not found: {missing}. Using placeholder text.")
            texts = ["passage: 空調設備 温度センサー"] * len(df)

        # 埋め込み計算
        print(f"  Encoding {len(texts):,} texts with [{self.backend}] backend ...")
        z = self.encode_texts(texts)

        # キャッシュ保存
        if cache_path is not None:
            np.savez_compressed(cache_path, embeddings=z)
            print(f"  💾 Saved embedding cache → {cache_path}")

        return z

    # ------------------------------------------------------------------
    # FAISS インデックス
    # ------------------------------------------------------------------

    def build_faiss_index(
        self,
        vectors: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Path = FAISS_INDEX_PATH,
    ) -> None:
        """
        FAISS インデックスを構築し保存する。

        設備カテゴリの意味空間を安定化させるために使用。
        MVP では near-neighbor 検索は必須ではないが、
        将来の説明性 (どの正常設備に最も近いか) のために構築しておく。

        Args:
            vectors  : (N, D) 埋め込みベクトル
            labels   : 各ベクトルに対応するラベル（設備名など）
            save_path: インデックスの保存先
        """
        try:
            import faiss
        except ImportError:
            print("  ⚠ faiss not installed. Skipping FAISS index build.")
            print("    Install: pip install faiss-cpu")
            return

        dim = vectors.shape[1]
        vecs = _normalize(vectors)  # コサイン類似度 → 内積で代用

        print(f"  Building FAISS IVFFlat index: {len(vectors)} vectors, dim={dim}")

        # 小サイズ（< 1000）なら FlatIP、大サイズなら IVFFlat
        if len(vectors) < 1000:
            index = faiss.IndexFlatIP(dim)
        else:
            n_centroids = min(64, len(vectors) // 10)
            quantizer   = faiss.IndexFlatIP(dim)
            index       = faiss.IndexIVFFlat(quantizer, dim, n_centroids,
                                              faiss.METRIC_INNER_PRODUCT)
            index.train(vecs)

        if FAISS_USE_GPU:
            try:
                res   = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                print("  ⚠ FAISS GPU 利用不可。CPU で続行します。")

        index.add(vecs)
        self.faiss_index  = index
        self.faiss_labels = labels or [str(i) for i in range(len(vectors))]

        # 保存 (GPU → CPU に転換してから保存)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if FAISS_USE_GPU:
            index_cpu = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index_cpu, str(save_path))
        else:
            faiss.write_index(index, str(save_path))

        print(f"  💾 FAISS index saved → {save_path}")

        # labels も保存
        labels_path = save_path.with_suffix(".labels.npy")
        np.save(str(labels_path), np.array(self.faiss_labels))
        print(f"  💾 FAISS labels saved → {labels_path}")

    def load_faiss_index(self, index_path: Path = FAISS_INDEX_PATH) -> bool:
        """FAISS インデックスをロード。成功時 True を返す。"""
        try:
            import faiss
        except ImportError:
            return False

        if not index_path.exists():
            return False

        self.faiss_index  = faiss.read_index(str(index_path))
        labels_path = index_path.with_suffix(".labels.npy")
        if labels_path.exists():
            self.faiss_labels = list(np.load(str(labels_path), allow_pickle=True))
        print(f"  ✓ FAISS index loaded: {index_path.name}")
        return True

    def faiss_search(
        self,
        query: np.ndarray,
        k: int = FAISS_N_NEIGHBORS,
    ) -> Tuple[List[str], np.ndarray]:
        """
        FAISS 近傍探索。

        Args:
            query : 1D または 2D 埋め込みベクトル
            k     : 返す近傍数

        Returns:
            (neighbor_labels, distances)
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index が構築されていません。build_faiss_index() を先に呼んでください。")

        query_2d = query.reshape(1, -1).astype(np.float32)
        query_2d = _normalize(query_2d)

        distances, indices = self.faiss_index.search(query_2d, k)
        labels = [self.faiss_labels[i] for i in indices[0] if 0 <= i < len(self.faiss_labels)]
        return labels, distances[0]


# =====================================================================
# スタンドアロン実行（テスト・事前計算用）
# =====================================================================

def precompute_all_embeddings(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    学習・テストデータ全件のテキスト埋め込みを事前計算してキャッシュする。
    train_triplet_model.py の実行前に一度だけ呼ぶ。
    """
    embedder = TextEmbedder()
    lookup   = load_master_lookup()

    print("\n[1/4] Computing train text embeddings ...")
    z_train = embedder.get_embeddings(train_df, cache_path=TEXT_EMBED_TRAIN_NPZ)
    print(f"  ✓ Train embeddings: {z_train.shape}")

    print("\n[2/4] Computing test text embeddings ...")
    z_test = embedder.get_embeddings(test_df, cache_path=TEXT_EMBED_TEST_NPZ)
    print(f"  ✓ Test embeddings: {z_test.shape}")

    # ユニーク (equipment_id, check_item_id) の埋め込みで FAISS インデックスを構築
    print("\n[3/4] Building unique equipment embeddings for FAISS ...")
    all_df     = pd.concat([train_df, test_df], ignore_index=True)
    unique_df  = all_df.drop_duplicates(subset=["equipment_id", "check_item_id"])
    z_unique   = embedder.get_embeddings(unique_df, cache_path=TEXT_EMBED_UNIQUE_NPZ)
    unique_labels = (
        unique_df["equipment_id"].astype(str) + "_" + unique_df["check_item_id"].astype(str)
    ).tolist()

    print("\n[4/4] Building FAISS index ...")
    embedder.build_faiss_index(z_unique, labels=unique_labels)

    print("\n✅ 事前計算完了。train_triplet_model.py を実行してください。")
    return embedder, z_train, z_test


if __name__ == "__main__":
    from config_v3 import PROCESSED_DATA_DIR

    print("="*70)
    print("Text Embedding Pre-computation for v3-0")
    print("="*70)

    train_path = PROCESSED_DATA_DIR / "training_samples_enriched.csv"
    test_path  = PROCESSED_DATA_DIR / "test_samples_enriched.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Enriched CSVs not found.\n"
            "先に create_enriched_features.py を実行してください。"
        )

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    precompute_all_embeddings(train_df, test_df)

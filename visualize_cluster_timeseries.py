"""
Cluster Time-Series Visualiser  (v3-0)
=======================================
* Re-runs DBSCAN (same settings as analyze_text_embeddings.py) on the
  278 unique text embeddings to recover cluster assignments.
* For each cluster, picks the pseudo-centroid representative pair
  (equipment_id × check_item_id).
* Fetches the most recent 90-day window from the **test** dataset for
  that pair and plots the normalised values_sequence.
* Output: 4 PNG files  (5 clusters × 4 pages = 20 clusters total)
  results/triplet_model/cluster_ts_page{1..4}.png

Privacy: no equipment IDs are shown; labels are Category + Check-item type only.
"""

import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────
from config_v3 import (
    TEXT_EMBED_CACHE_DIR, TEXT_EMBED_UNIQUE_NPZ,
    EQUIP_MASTER_CSV, EQUIP_MASTER_ENCODING, MASTER_COLS,
    PROCESSED_DATA_DIR, TRIPLET_RESULTS_DIR,
)

DBSCAN_EPS         = 0.24
DBSCAN_MIN_SAMPLES = 3
CLUSTERS_PER_PAGE  = 5
FONT_FAMILY        = "DejaVu Sans"
LABELS_NPY         = TEXT_EMBED_CACHE_DIR / "equip_category.labels.npy"
OUT_DIR            = TRIPLET_RESULTS_DIR

# ── Japanese → English ────────────────────────────────────────────────
CATEGORY_EN = {
    "ポンプ":           "Pump",
    "空調設備":         "Air Conditioning",
    "個別分散空調設備": "Distributed AC Unit",
    "機械設備":         "Mechanical Equipment",
}

CHECK_ITEM_KEYWORDS = [
    ("運転電流",     "Operating Current"),
    ("主電動機",     "Main Motor Current"),
    ("電流値",       "Current [A]"),
    ("電流計",       "Ammeter"),
    ("電流",         "Current [A]"),
    ("電力",         "Power [kW]"),
    ("電圧",         "Voltage [V]"),
    ("電導度",       "Conductivity"),
    ("吐出圧力",     "Discharge Pressure"),
    ("吐出圧",       "Discharge Pressure"),
    ("吸込圧",       "Suction Pressure"),
    ("PIS圧",        "PIS Pressure"),
    ("差圧",         "Diff. Pressure [Pa]"),
    ("蒸気圧力",     "Steam Pressure"),
    ("蒸気圧",       "Steam Pressure"),
    ("タンク圧力",   "Tank Pressure"),
    ("凝縮器",       "Condenser Pressure"),
    ("蒸発器",       "Evaporator Pressure"),
    ("潤滑油",       "Lube Oil Pressure"),
    ("エレメント入口圧", "Element Inlet Pressure"),
    ("エレメント出口圧", "Element Outlet Pressure"),
    ("圧力",         "Pressure"),
    ("冷水入口",     "ChW Inlet [°C]"),
    ("冷水出口",     "ChW Outlet [°C]"),
    ("冷却水入口",   "CW Inlet [°C]"),
    ("冷却水出口",   "CW Outlet [°C]"),
    ("冷水往",       "ChW Supply Temp"),
    ("冷水還",       "ChW Return Temp"),
    ("冷・温水往",   "HW/CW Supply Temp"),
    ("冷・温水還",   "HW/CW Return Temp"),
    ("設定温度",     "Set Temp [°C]"),
    ("給気温度",     "Supply Air Temp"),
    ("入口水温",     "Inlet Water Temp"),
    ("出口水温",     "Outlet Water Temp"),
    ("気温",         "Ambient Temp [°C]"),
    ("温度計",       "Thermometer"),
    ("温度",         "Temperature [°C]"),
    ("主蒸気流量",   "Main Steam Flow"),
    ("蒸気流量",     "Steam Flow"),
    ("風量",         "Air Flow"),
    ("流量",         "Flow Rate"),
    ("レベル",       "Level"),
    ("高架水槽",     "Elevated Tank Level"),
    ("タンク残量",   "Tank Remaining"),
    ("残量",         "Remaining Level"),
    ("振動",         "Vibration"),
    ("回転数",       "RPM"),
    ("インバータ周波数", "Inverter Frequency"),
    ("周波数",       "Frequency"),
    ("ストローク",   "Stroke"),
    ("パルス",       "Pulse"),
    ("稼働時間",     "Operation Hours"),
    ("運転容量",     "Operating Capacity"),
    ("O2計",         "O2 Meter"),
    ("ガス濃度",     "Gas Concentration"),
    ("pH",           "pH"),
    ("薬液",         "Chemical Level"),
    ("苛性",         "Caustic Level"),
]


def _translate(name: str) -> str:
    for jp, en in CHECK_ITEM_KEYWORDS:
        if jp in name:
            return en
    return "Check Item"


def make_label(eid: str, cid: str, master: dict) -> str:
    info = master.get((eid, cid))
    if info is None:
        return "Unknown\nCheck Item"
    cat_en = CATEGORY_EN.get(info["category"], info["category"])
    chk_en = _translate(info["check_item"])
    return f"{cat_en}\n{chk_en}"


# ── 1. Load embeddings & labels ───────────────────────────────────────
print("=" * 60)
print("Cluster Time-Series Visualiser  (v3-0, test data)")
print("=" * 60)

z_all        = np.load(TEXT_EMBED_UNIQUE_NPZ)["embeddings"].astype(np.float32)
faiss_labels = list(np.load(str(LABELS_NPY), allow_pickle=True))

pair_ids: list[tuple[str, str]] = []
for lbl in faiss_labels:
    parts = lbl.split("_")
    pair_ids.append((parts[0], parts[1]))

print(f"[1] Unique embeddings: {z_all.shape}  pairs: {len(pair_ids)}")

# ── 2. Master CSV ─────────────────────────────────────────────────────
mdf = pd.read_csv(EQUIP_MASTER_CSV, encoding=EQUIP_MASTER_ENCODING,
                  dtype={MASTER_COLS["equip_id"]: str,
                         MASTER_COLS["check_item_id"]: str})
master: dict[tuple[str, str], dict] = {}
for _, row in mdf.iterrows():
    eid = str(row[MASTER_COLS["equip_id"]]).strip()
    cid = str(row[MASTER_COLS["check_item_id"]]).strip()
    master[(eid, cid)] = {
        "category":   str(row.get(MASTER_COLS["equip_category"], "") or "").strip(),
        "check_item": str(row.get(MASTER_COLS["check_item_name"], "") or "").strip(),
    }
print(f"[2] Master pairs: {len(master)}")

# ── 3. DBSCAN ─────────────────────────────────────────────────────────
z_norm = normalize(z_all, norm="l2")
db     = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                metric="euclidean", n_jobs=-1)
cluster_labels = db.fit_predict(z_norm)

unique_clusters = sorted(set(cluster_labels) - {-1})
n_clusters      = len(unique_clusters)
n_noise         = int((cluster_labels == -1).sum())
print(f"[3] DBSCAN: {n_clusters} clusters  {n_noise} noise  (eps={DBSCAN_EPS})")

# pseudo-centroid representative index per cluster
centroid_reps: list[tuple[int, int]] = []   # (cluster_id, point_idx)
for k in unique_clusters:
    mask    = cluster_labels == k
    members = np.where(mask)[0]
    centre  = normalize(z_norm[mask].mean(axis=0, keepdims=True), norm="l2")[0]
    dists   = np.linalg.norm(z_norm[mask] - centre, axis=1)
    centroid_reps.append((k, int(members[np.argmin(dists)])))

# ── 4. Load test data ─────────────────────────────────────────────────
test_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
print(f"[4] Loading test data: {test_path.name}")
test_df = pd.read_csv(test_path, dtype={"equipment_id": str, "check_item_id": str})
test_df["window_end"] = pd.to_datetime(test_df["window_end"])
print(f"    rows: {len(test_df):,}")

def get_latest_series(eid: str, cid: str) -> tuple[np.ndarray | None, str, str, int]:
    """Return (values, window_start, label_any) for the most recent test window."""
    sub = test_df[(test_df["equipment_id"] == eid) &
                  (test_df["check_item_id"] == cid)]
    if sub.empty:
        return None, "", "", 0
    row = sub.loc[sub["window_end"].idxmax()]
    vals = np.array(ast.literal_eval(row["values_sequence"]), dtype=np.float64)
    return vals, str(row["window_start"]), str(row["window_end"]), int(row["any_anomaly"])

# ── 5. Plot ───────────────────────────────────────────────────────────
plt.rcParams["font.family"]       = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False

CMAP      = plt.get_cmap("tab20")
ANOM_COL  = "#e53935"   # red for anomaly window
NORM_COL  = "#1976D2"   # blue for normal window

n_pages = (n_clusters + CLUSTERS_PER_PAGE - 1) // CLUSTERS_PER_PAGE

for page in range(n_pages):
    chunk = centroid_reps[page * CLUSTERS_PER_PAGE :
                          (page + 1) * CLUSTERS_PER_PAGE]

    fig, axes = plt.subplots(
        len(chunk), 1,
        figsize=(14, 3.2 * len(chunk)),
        sharex=False,
    )
    if len(chunk) == 1:
        axes = [axes]

    fig.suptitle(
        f"Representative Time-Series per DBSCAN Cluster  —  Test Data  (v3-0)\n"
        f"Page {page + 1} / {n_pages}  •  Clusters {chunk[0][0]}–{chunk[-1][0]}  "
        f"•  eps={DBSCAN_EPS}  min_samples={DBSCAN_MIN_SAMPLES}",
        fontsize=11, y=1.01,
    )

    for ax_i, (k, rep_idx) in enumerate(chunk):
        ax   = axes[ax_i]
        eid, cid = pair_ids[rep_idx]
        vals, ws, we, is_anom = get_latest_series(eid, cid)

        cluster_col = CMAP(list(unique_clusters).index(k) / max(n_clusters, 1))
        lbl         = make_label(eid, cid, master)
        size_k      = int((cluster_labels == k).sum())

        if vals is None:
            ax.text(0.5, 0.5, "No test data for this pair",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"Cluster {k}  ({size_k} pts)  |  {lbl.replace(chr(10), ' / ')}",
                         fontsize=9, loc="left")
            continue

        x = np.arange(len(vals))

        # background shading for anomaly flag
        bg_col = "#ffebee" if is_anom else "#e3f2fd"
        ax.set_facecolor(bg_col)

        # time-series line
        line_col = ANOM_COL if is_anom else NORM_COL
        ax.plot(x, vals, color=line_col, linewidth=1.1, alpha=0.85, zorder=3)
        ax.fill_between(x, vals, alpha=0.15, color=line_col, zorder=2)

        # cluster colour strip on left edge
        ax.axvline(x=-0.5, color=cluster_col, linewidth=6, solid_capstyle="butt",
                   clip_on=False, zorder=4)

        # zero reference
        ax.axhline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=1)

        # labels
        anom_str = "⚠ ANOMALY" if is_anom else "Normal"
        ax.set_title(
            f"Cluster {k}  ({size_k} pts)  |  {lbl.replace(chr(10), ' / ')}"
            f"  |  Window: {ws} → {we}  |  {anom_str}",
            fontsize=8.5, loc="left", pad=4,
        )
        ax.set_ylabel("Normalised value", fontsize=8)
        ax.set_xlabel("Days in 90-day window", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-1, len(vals))
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / f"cluster_ts_page{page + 1}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")

print(f"\n✓ Done: {n_pages} figures in {OUT_DIR}")
print("=" * 60)

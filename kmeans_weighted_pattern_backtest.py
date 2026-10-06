from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. 基本設定
# ============================================================

# ------------------------------------------------------------
# 上一步 find_big_move_candidates.py 的輸出資料夾。
# 這份資料有：
# - future_return_3d
# - target_3d
# - bullish / bearish / neutral 標籤
# ------------------------------------------------------------
CANDIDATE_DIR = Path("output_big_move_candidates")

TRAIN_LABELED_FILE = (
    CANDIDATE_DIR / "train_with_future_return_3d.csv"
)

# ------------------------------------------------------------
# build_tw_features_v3.py 的輸出資料夾。
# 用於取得 2026 的特徵資料。
# ------------------------------------------------------------
FEATURE_DIR = Path(
    "output_tw_stock_features_v3_ex_right_or_split"
)

TEST_FILE = FEATURE_DIR / "test_2026.csv"

# ------------------------------------------------------------
# 本程式的新輸出資料夾。
# 不覆蓋前面錯誤或簡化版本的結果。
# ------------------------------------------------------------
OUTPUT_DIR = Path("output_kmeans_weighted_backtest")

# ------------------------------------------------------------
# 題目定義：
# t+3 收盤相對 t 收盤：
# > +5% 為大漲
# < -5% 為大跌
# ------------------------------------------------------------
FORECAST_HORIZON = 3
BULLISH_THRESHOLD = 0.05
BEARISH_THRESHOLD = -0.05

# ------------------------------------------------------------
# 固定建立 10 個看漲 + 10 個看跌 K 線型態。
# 若樣本很少，程式會自動降低，但 50 檔資料通常足夠。
# ------------------------------------------------------------
N_BULLISH_CLUSTERS = 10
N_BEARISH_CLUSTERS = 10

# ------------------------------------------------------------
# KMeans 設定
# ------------------------------------------------------------
KMEANS_N_INIT = 30
RANDOM_STATE = 42

# ------------------------------------------------------------
# 二次篩選：
# 每個型態在 2026 至少要出現幾次才列入「有效型態」。
# 你可保留 10；這比頻率 1 或 2 次可靠。
# ------------------------------------------------------------
MIN_TEST_FREQUENCY = 10

# ------------------------------------------------------------
# 人工設定的加權歐氏距離權重。
#
# 與你提供的參考作業報告一致：
# - body_pct：實體線，權重 2
# - trend_5d_pct：前五日趨勢，權重 2
# - 其餘特徵：權重 1
#
# 先 StandardScaler 標準化，再套權重。
# ------------------------------------------------------------
FEATURE_WEIGHTS = {
    "upper_shadow_pct": 1.0,
    "lower_shadow_pct": 1.0,
    "body_pct": 2.0,
    "prev_upper_shadow_pct": 1.0,
    "prev_lower_shadow_pct": 1.0,
    "prev_body_pct": 1.0,
    "open_pattern_pct": 1.0,
    "close_pattern_pct": 1.0,
    "volume_vs_ma5_ratio": 1.0,
    "trend_5d_pct": 2.0
}

FEATURE_COLUMNS = list(FEATURE_WEIGHTS.keys())


# ============================================================
# 1. 讀取並清理資料
# ============================================================

def load_labeled_training_data(file_path: Path) -> pd.DataFrame:
    """
    讀取 2018~2025 訓練資料。

    此檔案必須由 find_big_move_candidates.py 產生，
    因此已包含：
    - future_close_3d
    - future_return_3d
    - target_3d
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到訓練資料：{file_path}\n"
            "請先執行 find_big_move_candidates.py。"
        )

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    required_columns = [
        "date",
        "ticker",
        "close",
        "future_return_3d",
        "target_3d"
    ] + FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"訓練資料缺少欄位：{missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = [
        "close",
        "future_return_3d"
    ] + FEATURE_COLUMNS

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # 只保留有完整未來三日資料的樣本。
    df = df[
        df["target_3d"].isin(
            ["bullish", "bearish", "neutral"]
        )
    ].copy()

    df = df.dropna(
        subset=[
            "date",
            "ticker",
            "close",
            "future_return_3d"
        ] + FEATURE_COLUMNS
    ).copy()

    df = (
        df
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"])
        .reset_index(drop=True)
    )

    return df


def load_test_data(file_path: Path) -> pd.DataFrame:
    """
    讀取 2026 測試資料。

    test_2026.csv 原本沒有 future_return_3d，
    所以下面會再由 close 自行計算三日後報酬。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到測試資料：{file_path}\n"
            "請確認 FEATURE_DIR 是否正確。"
        )

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    required_columns = [
        "date",
        "ticker",
        "close"
    ] + FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"測試資料缺少欄位：{missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = [
        "close"
    ] + FEATURE_COLUMNS

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "date",
            "ticker",
            "close"
        ] + FEATURE_COLUMNS
    ).copy()

    df = (
        df
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"])
        .reset_index(drop=True)
    )

    return df


def add_future_return(
    df: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """
    對每一檔股票計算三個交易日後報酬：

    future_return_3d = Close(t+3) / Close(t) - 1

    每檔股票的最後三筆，因沒有 t+3 收盤價，
    future_return_3d 會是 NaN，後續不納入回測統計。
    """

    result = df.copy()

    result = (
        result
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    group = result.groupby("ticker", group_keys=False)

    result["future_close_3d"] = group["close"].shift(
        -horizon
    )

    result["future_return_3d"] = (
        result["future_close_3d"]
        / result["close"]
        - 1
    )

    result["future_return_3d_pct"] = (
        result["future_return_3d"] * 100
    )

    result["actual_target_3d"] = "unknown"

    valid_future = result["future_return_3d"].notna()

    result.loc[
        valid_future,
        "actual_target_3d"
    ] = "neutral"

    result.loc[
        result["future_return_3d"] > BULLISH_THRESHOLD,
        "actual_target_3d"
    ] = "bullish"

    result.loc[
        result["future_return_3d"] < BEARISH_THRESHOLD,
        "actual_target_3d"
    ] = "bearish"

    return result


# ============================================================
# 2. 加權 KMeans 分群
# ============================================================

def get_weight_array() -> np.ndarray:
    """
    取得與 FEATURE_COLUMNS 同順序的權重向量。
    """

    return np.array(
        [
            FEATURE_WEIGHTS[column]
            for column in FEATURE_COLUMNS
        ],
        dtype=float
    )


def fit_weighted_kmeans(
    candidate_df: pd.DataFrame,
    n_clusters: int,
    direction: str,
    scaler: StandardScaler
) -> tuple[pd.DataFrame, KMeans, np.ndarray, int]:
    """
    對大漲或大跌訓練樣本做 KMeans。

    重點：
    1. 先用 scaler 將所有特徵標準化。
    2. 對標準化特徵乘上 sqrt(weight)。
       因為一般歐氏距離對應：
       sqrt(sum((x-y)^2))
       若要加權：
       sqrt(sum(w * (x-y)^2))
       所以向量應乘 sqrt(w)。
    3. KMeans 在加權空間中建立群中心。
    """

    candidates = candidate_df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    if len(candidates) < 2:
        raise RuntimeError(
            f"{direction} 候選樣本不足，無法分群。"
        )

    # 每群至少希望有 10 個訓練候選樣本。
    maximum_reasonable_clusters = max(
        2,
        len(candidates) // 10
    )

    actual_clusters = min(
        n_clusters,
        maximum_reasonable_clusters
    )

    actual_clusters = min(
        actual_clusters,
        len(candidates)
    )

    X_scaled = scaler.transform(
        candidates[FEATURE_COLUMNS]
    )

    sqrt_weights = np.sqrt(
        get_weight_array()
    )

    X_weighted = X_scaled * sqrt_weights

    kmeans = KMeans(
        n_clusters=actual_clusters,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE
    )

    candidates["cluster_id"] = kmeans.fit_predict(
        X_weighted
    )

    # 到所屬中心的加權歐氏距離
    assigned_centers = kmeans.cluster_centers_[
        candidates["cluster_id"].to_numpy()
    ]

    distances = np.sqrt(
        np.sum(
            (X_weighted - assigned_centers) ** 2,
            axis=1
        )
    )

    candidates["distance_to_centroid"] = distances

    candidates["similarity_to_centroid"] = (
        1 / (1 + distances)
    )

    candidates["direction"] = direction

    return (
        candidates,
        kmeans,
        sqrt_weights,
        actual_clusters
    )


# ============================================================
# 3. 對 2026 所有交易日做群中心匹配
# ============================================================

def assign_nearest_pattern(
    test_df: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
    sqrt_weights: np.ndarray,
    direction: str
) -> pd.DataFrame:
    """
    對 2026 每一筆可用資料：

    - 計算與每個群中心的加權歐氏距離。
    - 指派至最近的群中心。
    - 每個交易日都會獲得一個 pattern_id。

    這是與簡化版最大不同處：
    不再只取前 30 個最像的日期，而是讓 2026 全部資料
    都被分配到某一種看漲或看跌型態。
    """

    result = test_df.copy()

    X_scaled = scaler.transform(
        result[FEATURE_COLUMNS]
    )

    X_weighted = X_scaled * sqrt_weights

    # shape:
    # (n_test_rows, n_clusters, n_features)
    differences = (
        X_weighted[:, np.newaxis, :]
        - kmeans.cluster_centers_[np.newaxis, :, :]
    )

    distance_matrix = np.sqrt(
        np.sum(differences ** 2, axis=2)
    )

    nearest_cluster_id = np.argmin(
        distance_matrix,
        axis=1
    )

    nearest_distance = distance_matrix[
        np.arange(len(result)),
        nearest_cluster_id
    ]

    result["direction"] = direction
    result["cluster_id"] = nearest_cluster_id
    result["pattern_id"] = [
        f"{direction}_cluster_{cluster_id + 1:02d}"
        for cluster_id in nearest_cluster_id
    ]

    result["weighted_euclidean_distance"] = (
        nearest_distance
    )

    result["similarity"] = (
        1 / (1 + nearest_distance)
    )

    return result


# ============================================================
# 4. 建立群中心的可讀特徵描述
# ============================================================

def build_cluster_profiles(
    clustered_train_df: pd.DataFrame,
    direction: str
) -> pd.DataFrame:
    """
    對每一群訓練樣本建立摘要。

    因為 KMeans 中心在標準化且加權後的空間，
    為了讓報告較容易閱讀，這裡直接統計各群原始特徵平均。
    """

    profile = (
        clustered_train_df
        .groupby("cluster_id")
        .agg(
            train_candidate_count=("cluster_id", "count"),
            train_mean_future_return_3d_pct=(
                "future_return_3d",
                lambda series: series.mean() * 100
            ),
            train_median_future_return_3d_pct=(
                "future_return_3d",
                lambda series: series.median() * 100
            ),
            train_mean_similarity_to_centroid=(
                "similarity_to_centroid",
                "mean"
            ),
            avg_upper_shadow_pct=(
                "upper_shadow_pct",
                "mean"
            ),
            avg_lower_shadow_pct=(
                "lower_shadow_pct",
                "mean"
            ),
            avg_body_pct=("body_pct", "mean"),
            avg_open_pattern_pct=(
                "open_pattern_pct",
                "mean"
            ),
            avg_close_pattern_pct=(
                "close_pattern_pct",
                "mean"
            ),
            avg_volume_vs_ma5_ratio=(
                "volume_vs_ma5_ratio",
                "mean"
            ),
            avg_trend_5d_pct=(
                "trend_5d_pct",
                "mean"
            )
        )
        .reset_index()
        .sort_values("cluster_id")
    )

    profile["direction"] = direction

    profile["pattern_id"] = [
        f"{direction}_cluster_{cluster_id + 1:02d}"
        for cluster_id in profile["cluster_id"]
    ]

    return profile


def make_auto_description(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    依每群平均特徵，產生可在報告中初步使用的描述。

    這是輔助說明；最終可再配合 GUI K 線圖人工命名。
    """

    result = profile_df.copy()

    descriptions = []

    for _, row in result.iterrows():
        parts = []

        trend = row["avg_trend_5d_pct"]
        volume = row["avg_volume_vs_ma5_ratio"]
        upper = row["avg_upper_shadow_pct"]
        lower = row["avg_lower_shadow_pct"]
        body = row["avg_body_pct"]

        if trend >= 5:
            parts.append("前五日上漲")
        elif trend <= -5:
            parts.append("前五日下跌")

        if volume >= 0.30:
            parts.append("量能放大")
        elif volume <= -0.20:
            parts.append("量能縮減")

        if upper >= 1.5:
            parts.append("上影線偏長")

        if lower >= 1.5:
            parts.append("下影線偏長")

        if body >= 1.0:
            parts.append("偏紅K")
        elif body <= -1.0:
            parts.append("偏黑K")

        if not parts:
            parts.append("中性量價結構")

        descriptions.append("、".join(parts))

    result["auto_description"] = descriptions

    return result


# ============================================================
# 5. 2026 回測統計
# ============================================================

def calculate_2026_pattern_summary(
    assigned_test_df: pd.DataFrame,
    direction: str
) -> pd.DataFrame:
    """
    對被分配到每個 pattern 的 2026 測試資料統計：

    - 出現頻率
    - 預測命中數
    - 命中率
    - 平均三日後報酬
    - 相似度
    - 綜合排序分數
    """

    valid_df = assigned_test_df.dropna(
        subset=["future_return_3d"]
    ).copy()

    if direction == "bullish":
        valid_df["is_direction_hit"] = (
            valid_df["future_return_3d"]
            > BULLISH_THRESHOLD
        )
    elif direction == "bearish":
        valid_df["is_direction_hit"] = (
            valid_df["future_return_3d"]
            < BEARISH_THRESHOLD
        )
    else:
        raise ValueError(
            "direction 必須是 bullish 或 bearish。"
        )

    summary = (
        valid_df
        .groupby(
            ["pattern_id", "cluster_id", "direction"],
            as_index=False
        )
        .agg(
            test_frequency=("pattern_id", "count"),
            test_hit_count=("is_direction_hit", "sum"),
            test_hit_rate=("is_direction_hit", "mean"),
            test_mean_future_return_3d_pct=(
                "future_return_3d",
                lambda series: series.mean() * 100
            ),
            test_median_future_return_3d_pct=(
                "future_return_3d",
                lambda series: series.median() * 100
            ),
            test_std_future_return_3d_pct=(
                "future_return_3d",
                lambda series: series.std() * 100
            ),
            test_mean_similarity=("similarity", "mean"),
            test_min_similarity=("similarity", "min"),
            test_max_similarity=("similarity", "max"),
            test_mean_distance=(
                "weighted_euclidean_distance",
                "mean"
            )
        )
    )

    summary["test_hit_rate_pct"] = (
        summary["test_hit_rate"] * 100
    )

    if direction == "bullish":
        directional_return = (
            summary["test_mean_future_return_3d_pct"]
        )
    else:
        directional_return = (
            -summary["test_mean_future_return_3d_pct"]
        )

    # 排名分數：
    # 命中率 × 正向方向平均報酬 × log(1+出現次數) × 平均相似度。
    summary["pattern_score"] = (
        summary["test_hit_rate"]
        * directional_return.clip(lower=0)
        * np.log1p(summary["test_frequency"])
        * summary["test_mean_similarity"]
    )

    summary["is_valid_pattern"] = (
        summary["test_frequency"]
        >= MIN_TEST_FREQUENCY
    )

    return summary.sort_values(
        "pattern_score",
        ascending=False
    ).reset_index(drop=True)


def calculate_overall_direction_summary(
    assigned_test_df: pd.DataFrame,
    direction: str
) -> pd.DataFrame:
    """
    將某方向全部 2026 訊號合併統計。

    注意：
    因為本方法將每個測試日指派到最近型態，
    在同一方向內的有效測試日會各出現一次。
    """

    valid_df = assigned_test_df.dropna(
        subset=["future_return_3d"]
    ).copy()

    if direction == "bullish":
        is_hit = (
            valid_df["future_return_3d"]
            > BULLISH_THRESHOLD
        )
    else:
        is_hit = (
            valid_df["future_return_3d"]
            < BEARISH_THRESHOLD
        )

    result = pd.DataFrame(
        {
            "direction": [direction],
            "test_sample_count": [len(valid_df)],
            "hit_count": [int(is_hit.sum())],
            "hit_rate_pct": [is_hit.mean() * 100],
            "mean_future_return_3d_pct": [
                valid_df["future_return_3d"].mean() * 100
            ],
            "median_future_return_3d_pct": [
                valid_df["future_return_3d"].median() * 100
            ]
        }
    )

    return result


def calculate_baseline_summary(
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    計算 2026 所有可用樣本的自然基準率。
    """

    valid_df = test_df.dropna(
        subset=["future_return_3d"]
    ).copy()

    bullish_rate = (
        valid_df["future_return_3d"]
        > BULLISH_THRESHOLD
    ).mean()

    bearish_rate = (
        valid_df["future_return_3d"]
        < BEARISH_THRESHOLD
    ).mean()

    return pd.DataFrame(
        {
            "metric": [
                "test_total_samples",
                "baseline_bullish_rate_pct",
                "baseline_bearish_rate_pct",
                "baseline_mean_future_return_3d_pct"
            ],
            "value": [
                len(valid_df),
                bullish_rate * 100,
                bearish_rate * 100,
                valid_df["future_return_3d"].mean() * 100
            ]
        }
    )


# ============================================================
# 6. 合併訓練型態與測試結果
# ============================================================

def combine_train_test_results(
    cluster_profile: pd.DataFrame,
    test_summary: pd.DataFrame,
    baseline_rate_pct: float,
    direction: str
) -> pd.DataFrame:
    """
    合併：

    - 訓練期：群內大漲／大跌樣本資訊
    - 測試期：出現頻率、命中率、平均報酬
    - Lift：型態命中率／2026 自然基準率
    """

    result = cluster_profile.merge(
        test_summary,
        on=["pattern_id", "cluster_id", "direction"],
        how="left"
    )

    if direction == "bullish":
        result["lift_vs_2026_baseline"] = (
            result["test_hit_rate_pct"] / baseline_rate_pct
        )
    else:
        result["lift_vs_2026_baseline"] = (
            result["test_hit_rate_pct"] / baseline_rate_pct
        )

    result = result.sort_values(
        "pattern_score",
        ascending=False
    ).reset_index(drop=True)

    result.insert(
        0,
        "overall_rank",
        range(1, len(result) + 1)
    )

    return result


# ============================================================
# 7. 輸出 CSV
# ============================================================

def save_all_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bullish_clustered_train: pd.DataFrame,
    bearish_clustered_train: pd.DataFrame,
    bullish_profile: pd.DataFrame,
    bearish_profile: pd.DataFrame,
    bullish_assigned_test: pd.DataFrame,
    bearish_assigned_test: pd.DataFrame,
    bullish_results: pd.DataFrame,
    bearish_results: pd.DataFrame,
    overall_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame
) -> None:
    """
    輸出完整結果，供報告、Excel、GUI 使用。
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. 設定與權重
    # --------------------------------------------------------
    settings = pd.DataFrame(
        {
            "setting": [
                "train_file",
                "test_file",
                "forecast_horizon",
                "bullish_threshold",
                "bearish_threshold",
                "n_bullish_clusters",
                "n_bearish_clusters",
                "min_test_frequency",
                "kmeans_n_init",
                "random_state"
            ],
            "value": [
                str(TRAIN_LABELED_FILE),
                str(TEST_FILE),
                FORECAST_HORIZON,
                BULLISH_THRESHOLD,
                BEARISH_THRESHOLD,
                N_BULLISH_CLUSTERS,
                N_BEARISH_CLUSTERS,
                MIN_TEST_FREQUENCY,
                KMEANS_N_INIT,
                RANDOM_STATE
            ]
        }
    )

    settings.to_csv(
        OUTPUT_DIR / "settings_used.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        {
            "feature_name": FEATURE_COLUMNS,
            "weight": [
                FEATURE_WEIGHTS[column]
                for column in FEATURE_COLUMNS
            ]
        }
    ).to_csv(
        OUTPUT_DIR / "feature_weights.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 2. 訓練期分群結果
    # --------------------------------------------------------
    bullish_train_output = bullish_clustered_train.copy()
    bearish_train_output = bearish_clustered_train.copy()

    bullish_train_output["date"] = (
        bullish_train_output["date"].dt.strftime("%Y-%m-%d")
    )

    bearish_train_output["date"] = (
        bearish_train_output["date"].dt.strftime("%Y-%m-%d")
    )

    bullish_train_output.to_csv(
        OUTPUT_DIR / "train_bullish_candidates_clustered.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_train_output.to_csv(
        OUTPUT_DIR / "train_bearish_candidates_clustered.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bullish_profile.to_csv(
        OUTPUT_DIR / "bullish_cluster_profiles.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_profile.to_csv(
        OUTPUT_DIR / "bearish_cluster_profiles.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 3. 2026 全量型態分配結果
    # --------------------------------------------------------
    test_output = test_df.copy()
    test_output["date"] = (
        test_output["date"].dt.strftime("%Y-%m-%d")
    )

    test_output.to_csv(
        OUTPUT_DIR / "test_2026_with_future_return.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bullish_assigned_output = bullish_assigned_test.copy()
    bearish_assigned_output = bearish_assigned_test.copy()

    bullish_assigned_output["date"] = (
        bullish_assigned_output["date"]
        .dt.strftime("%Y-%m-%d")
    )

    bearish_assigned_output["date"] = (
        bearish_assigned_output["date"]
        .dt.strftime("%Y-%m-%d")
    )

    bullish_assigned_output.to_csv(
        OUTPUT_DIR / "test_2026_assigned_bullish_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_assigned_output.to_csv(
        OUTPUT_DIR / "test_2026_assigned_bearish_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 4. 最重要：訓練與測試比較、Top 10
    # --------------------------------------------------------
    bullish_results.to_csv(
        OUTPUT_DIR / "bullish_patterns_train_test_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_results.to_csv(
        OUTPUT_DIR / "bearish_patterns_train_test_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bullish_top10 = bullish_results.head(10).copy()
    bearish_top10 = bearish_results.head(10).copy()

    bullish_top10.to_csv(
        OUTPUT_DIR / "top10_bullish_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_top10.to_csv(
        OUTPUT_DIR / "top10_bearish_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    overall_summary.to_csv(
        OUTPUT_DIR / "overall_2026_direction_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    baseline_summary.to_csv(
        OUTPUT_DIR / "baseline_2026_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 8. 主程式
# ============================================================

def main() -> None:
    print("=" * 72)
    print("K-Means + 加權歐氏距離：K 線型態探勘與 2026 回測")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. 讀取資料
    # --------------------------------------------------------
    train_df = load_labeled_training_data(
        TRAIN_LABELED_FILE
    )

    test_df = load_test_data(TEST_FILE)

    test_df = add_future_return(
        test_df,
        horizon=FORECAST_HORIZON
    )

    print(f"訓練期資料列數：{len(train_df):,}")
    print(f"訓練期股票數：{train_df['ticker'].nunique()}")
    print(f"2026 測試資料列數：{len(test_df):,}")
    print(f"2026 測試股票數：{test_df['ticker'].nunique()}")

    # --------------------------------------------------------
    # 2. 對整個訓練集 fit scaler
    #
    # 非常重要：
    # scaler 不可以用 2026 fit，否則有資料洩漏。
    # --------------------------------------------------------
    scaler = StandardScaler()

    scaler.fit(
        train_df[FEATURE_COLUMNS]
    )

    # --------------------------------------------------------
    # 3. 分出歷史大漲／大跌候選樣本
    # --------------------------------------------------------
    bullish_candidates = train_df[
        train_df["target_3d"] == "bullish"
    ].copy()

    bearish_candidates = train_df[
        train_df["target_3d"] == "bearish"
    ].copy()

    print(
        f"\n訓練期大漲候選樣本數："
        f"{len(bullish_candidates):,}"
    )

    print(
        f"訓練期大跌候選樣本數："
        f"{len(bearish_candidates):,}"
    )

    # --------------------------------------------------------
    # 4. 大漲樣本 K-Means 分 10 群
    # --------------------------------------------------------
    print("\n開始建立 10 個看漲 K 線型態...")

    (
        bullish_clustered_train,
        bullish_kmeans,
        bullish_sqrt_weights,
        actual_bullish_clusters
    ) = fit_weighted_kmeans(
        candidate_df=bullish_candidates,
        n_clusters=N_BULLISH_CLUSTERS,
        direction="bullish",
        scaler=scaler
    )

    print(
        f"看漲型態群數：{actual_bullish_clusters}"
    )

    # --------------------------------------------------------
    # 5. 大跌樣本 K-Means 分 10 群
    # --------------------------------------------------------
    print("開始建立 10 個看跌 K 線型態...")

    (
        bearish_clustered_train,
        bearish_kmeans,
        bearish_sqrt_weights,
        actual_bearish_clusters
    ) = fit_weighted_kmeans(
        candidate_df=bearish_candidates,
        n_clusters=N_BEARISH_CLUSTERS,
        direction="bearish",
        scaler=scaler
    )

    print(
        f"看跌型態群數：{actual_bearish_clusters}"
    )

    # --------------------------------------------------------
    # 6. 建立每個群的可讀輪廓
    # --------------------------------------------------------
    bullish_profile = build_cluster_profiles(
        clustered_train_df=bullish_clustered_train,
        direction="bullish"
    )

    bearish_profile = build_cluster_profiles(
        clustered_train_df=bearish_clustered_train,
        direction="bearish"
    )

    bullish_profile = make_auto_description(
        bullish_profile
    )

    bearish_profile = make_auto_description(
        bearish_profile
    )

    # --------------------------------------------------------
    # 7. 2026 所有交易日都指派到最近的看漲型態
    # --------------------------------------------------------
    print("\n將 2026 所有交易日分配到最近看漲型態...")

    bullish_assigned_test = assign_nearest_pattern(
        test_df=test_df,
        kmeans=bullish_kmeans,
        scaler=scaler,
        sqrt_weights=bullish_sqrt_weights,
        direction="bullish"
    )

    # --------------------------------------------------------
    # 8. 2026 所有交易日都指派到最近的看跌型態
    # --------------------------------------------------------
    print("將 2026 所有交易日分配到最近看跌型態...")

    bearish_assigned_test = assign_nearest_pattern(
        test_df=test_df,
        kmeans=bearish_kmeans,
        scaler=scaler,
        sqrt_weights=bearish_sqrt_weights,
        direction="bearish"
    )

    # --------------------------------------------------------
    # 9. 計算每個型態的 2026 回測結果
    # --------------------------------------------------------
    bullish_test_summary = calculate_2026_pattern_summary(
        assigned_test_df=bullish_assigned_test,
        direction="bullish"
    )

    bearish_test_summary = calculate_2026_pattern_summary(
        assigned_test_df=bearish_assigned_test,
        direction="bearish"
    )

    baseline_summary = calculate_baseline_summary(
        test_df=test_df
    )

    baseline_bullish_rate_pct = float(
        baseline_summary.loc[
            baseline_summary["metric"]
            == "baseline_bullish_rate_pct",
            "value"
        ].iloc[0]
    )

    baseline_bearish_rate_pct = float(
        baseline_summary.loc[
            baseline_summary["metric"]
            == "baseline_bearish_rate_pct",
            "value"
        ].iloc[0]
    )

    # --------------------------------------------------------
    # 10. 合併訓練與測試結果，計算 Lift 與排名
    # --------------------------------------------------------
    bullish_results = combine_train_test_results(
        cluster_profile=bullish_profile,
        test_summary=bullish_test_summary,
        baseline_rate_pct=baseline_bullish_rate_pct,
        direction="bullish"
    )

    bearish_results = combine_train_test_results(
        cluster_profile=bearish_profile,
        test_summary=bearish_test_summary,
        baseline_rate_pct=baseline_bearish_rate_pct,
        direction="bearish"
    )

    # --------------------------------------------------------
    # 11. 全體方向統計
    #
    # 注意：
    # 每一個 2026 日期都被分配一次到 bullish 群、
    # 也被分配一次到 bearish 群。
    #
    # 因此「全部群合併」的方向命中率會接近 2026 自然基準率。
    # 真正要比較的是：
    # 每個群自己的 hit rate 與 Lift。
    # --------------------------------------------------------
    overall_bullish = calculate_overall_direction_summary(
        assigned_test_df=bullish_assigned_test,
        direction="bullish"
    )

    overall_bearish = calculate_overall_direction_summary(
        assigned_test_df=bearish_assigned_test,
        direction="bearish"
    )

    overall_summary = pd.concat(
        [overall_bullish, overall_bearish],
        ignore_index=True
    )

    # --------------------------------------------------------
    # 12. 輸出結果
    # --------------------------------------------------------
    save_all_outputs(
        train_df=train_df,
        test_df=test_df,
        bullish_clustered_train=bullish_clustered_train,
        bearish_clustered_train=bearish_clustered_train,
        bullish_profile=bullish_profile,
        bearish_profile=bearish_profile,
        bullish_assigned_test=bullish_assigned_test,
        bearish_assigned_test=bearish_assigned_test,
        bullish_results=bullish_results,
        bearish_results=bearish_results,
        overall_summary=overall_summary,
        baseline_summary=baseline_summary
    )

    # --------------------------------------------------------
    # 13. 終端機摘要
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("K-Means 型態探勘與 2026 回測完成")
    print("=" * 72)

    print(
        f"2026 全部可用測試樣本："
        f"{int(baseline_summary.iloc[0]['value']):,}"
    )

    print(
        f"2026 三日大漲 > 5% 基準率："
        f"{baseline_bullish_rate_pct:.2f}%"
    )

    print(
        f"2026 三日大跌 < -5% 基準率："
        f"{baseline_bearish_rate_pct:.2f}%"
    )

    print("\n看漲型態 Top 10：")

    print(
        bullish_results[
            [
                "overall_rank",
                "pattern_id",
                "test_frequency",
                "test_hit_rate_pct",
                "lift_vs_2026_baseline",
                "test_mean_future_return_3d_pct",
                "auto_description"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n看跌型態 Top 10：")

    print(
        bearish_results[
            [
                "overall_rank",
                "pattern_id",
                "test_frequency",
                "test_hit_rate_pct",
                "lift_vs_2026_baseline",
                "test_mean_future_return_3d_pct",
                "auto_description"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\n輸出資料夾：{OUTPUT_DIR.resolve()}")

    print("\n最重要的結果檔案：")
    print("1. top10_bullish_patterns.csv")
    print("2. top10_bearish_patterns.csv")
    print("3. bullish_patterns_train_test_results.csv")
    print("4. bearish_patterns_train_test_results.csv")
    print("5. test_2026_assigned_bullish_patterns.csv")
    print("6. test_2026_assigned_bearish_patterns.csv")


if __name__ == "__main__":
    main()
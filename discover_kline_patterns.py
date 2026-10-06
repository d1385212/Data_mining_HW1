from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. 你可調整的設定
# ============================================================

# 上一支 find_big_move_candidates.py 的輸出資料夾
INPUT_DIR = Path("output_big_move_candidates")

# 上一支程式建立的含 target_3d 資料
INPUT_FILE = INPUT_DIR / "train_with_future_return_3d.csv"

# 本支程式輸出的資料夾
# 使用新名稱，避免覆蓋你前一次有問題的結果
OUTPUT_DIR = Path("output_kline_patterns_v2")

# 每個型態回看幾個交易日
# 10 日可描述「先漲、再回檔、爆量或量縮」。
WINDOW_SIZE = 10

# 上一支程式定義的未來報酬期
FORECAST_HORIZON = 3

# 大漲、大跌定義
BULLISH_THRESHOLD = 0.05
BEARISH_THRESHOLD = -0.05

# 看漲候選與看跌候選各要分成幾種型態
# 先用 8 群，不要過度切碎資料。
N_CLUSTERS_BULLISH = 8
N_CLUSTERS_BEARISH = 8

# 相似度門檻：
# 只有 similarity >= 0.70 才算「這個型態出現一次」。
# 若出現次數都太低，可調為 0.65。
# 若出現次數都太高，可調為 0.75。
MIN_SIMILARITY = 0.70

# 同一檔股票若在很接近的日期重複出現訊號，
# 幾個交易日內只保留相似度最高的一次，避免重複計數。
COOLDOWN_DAYS = 5

# 一個型態至少出現幾次，才有資格被列入有效型態。
MIN_PATTERN_FREQUENCY = 10

# 「大漲超過 5%」或「大跌超過 5%」的最低命中率。
# 第一版設定 35%，因為三日內 ±5% 是很嚴格的標籤。
# 後續可再提高到 40% 或 45% 做穩健性檢查。
MIN_BULLISH_HIT_RATE = 0.35
MIN_BEARISH_HIT_RATE = 0.35

# 最終保留前幾個型態
TOP_N_PATTERNS = 10

# Random Forest 用來學習特徵權重
RF_N_ESTIMATORS = 300
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. 讀取訓練資料
# ============================================================

def load_training_data(file_path: Path) -> pd.DataFrame:
    """
    讀取由 find_big_move_candidates.py 建立的資料。

    必要欄位：
    - date
    - ticker
    - close
    - high
    - low
    - volume
    - target_3d
    - future_return_3d
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到資料檔：{file_path}\n"
            "請先執行 find_big_move_candidates.py，"
            "或檢查 INPUT_DIR 是否正確。"
        )

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    required_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target_3d",
        "future_return_3d"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"輸入資料缺少必要欄位：{missing_columns}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "future_return_3d"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # 前一支程式已把 unknown 排掉，但這裡再做防呆。
    df = df[
        df["target_3d"].isin(
            ["bullish", "bearish", "neutral"]
        )
    ].copy()

    df = df.dropna(
        subset=[
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "future_return_3d"
        ]
    ).copy()

    df = (
        df
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"])
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 2. 建立多日型態特徵
# ============================================================

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    建立描述「最近 10 個交易日型態」的數值特徵。

    包含：
    - 近 3/5/10 日報酬
    - 十日高低點、回撤、反彈
    - 今日與近期成交量相對十日均量
    - 十日內漲跌日數
    - 十日波動
    - 當日與近三日 K 線特徵

    這讓程式不只看單一根 K 線。
    """

    result = df.copy()

    result = (
        result
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    group = result.groupby("ticker", group_keys=False)

    # --------------------------------------------------------
    # 若原始資料已有 10 個 K 線特徵則保留。
    # 若缺少，程式自行重新計算必要特徵。
    # --------------------------------------------------------

    if "upper_shadow_pct" not in result.columns:
        result["upper_shadow_pct"] = (
            (
                result["high"]
                - result[["open", "close"]].max(axis=1)
            )
            / result["close"]
            * 100
        )

    if "lower_shadow_pct" not in result.columns:
        result["lower_shadow_pct"] = (
            (
                result[["open", "close"]].min(axis=1)
                - result["low"]
            )
            / result["close"]
            * 100
        )

    if "body_pct" not in result.columns:
        result["body_pct"] = (
            (result["close"] - result["open"])
            / result["close"]
            * 100
        )

    if "prev_close" not in result.columns:
        result["prev_close"] = group["close"].shift(1)

    if "prev_upper_shadow_pct" not in result.columns:
        result["prev_upper_shadow_pct"] = (
            group["upper_shadow_pct"].shift(1)
        )

    if "prev_lower_shadow_pct" not in result.columns:
        result["prev_lower_shadow_pct"] = (
            group["lower_shadow_pct"].shift(1)
        )

    if "prev_body_pct" not in result.columns:
        result["prev_body_pct"] = group["body_pct"].shift(1)

    if "open_pattern_pct" not in result.columns:
        result["open_pattern_pct"] = (
            (result["open"] - result["prev_close"])
            / result["close"]
            * 100
        )

    if "close_pattern_pct" not in result.columns:
        result["close_pattern_pct"] = (
            (result["close"] - result["prev_close"])
            / result["close"]
            * 100
        )

    if "volume_ma5" not in result.columns:
        result["volume_ma5"] = group["volume"].transform(
            lambda s: s.rolling(
                window=5,
                min_periods=5
            ).mean()
        )

    if "volume_vs_ma5_ratio" not in result.columns:
        result["volume_vs_ma5_ratio"] = (
            (result["volume"] - result["volume_ma5"])
            / result["volume"]
        )

    if "trend_5d_pct" not in result.columns:
        close_2d_ago = group["close"].shift(2)
        close_7d_ago = group["close"].shift(7)

        result["trend_5d_pct"] = (
            (close_2d_ago - close_7d_ago)
            / close_7d_ago
            * 100
        )

    # --------------------------------------------------------
    # 多日趨勢、回撤與量價背景特徵
    # --------------------------------------------------------

    result["daily_return_pct"] = (
        group["close"].pct_change() * 100
    )

    for days in [3, 5, 10]:
        result[f"return_{days}d_pct"] = (
            group["close"].pct_change(days) * 100
        )

    result["high_10d"] = group["high"].transform(
        lambda s: s.rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        ).max()
    )

    result["low_10d"] = group["low"].transform(
        lambda s: s.rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        ).min()
    )

    result["drawdown_from_high_10d_pct"] = (
        result["close"] / result["high_10d"] - 1
    ) * 100

    result["rebound_from_low_10d_pct"] = (
        result["close"] / result["low_10d"] - 1
    ) * 100

    result["volume_ma3"] = group["volume"].transform(
        lambda s: s.rolling(
            window=3,
            min_periods=3
        ).mean()
    )

    result["volume_ma10"] = group["volume"].transform(
        lambda s: s.rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        ).mean()
    )

    result["volume_ratio_today_10d"] = (
        result["volume"] / result["volume_ma10"]
    )

    result["volume_ratio_3d_10d"] = (
        result["volume_ma3"] / result["volume_ma10"]
    )

    result["up_day"] = (
        result["close"] > result["prev_close"]
    ).astype(float)

    result["down_day"] = (
        result["close"] < result["prev_close"]
    ).astype(float)

    result["up_days_10d"] = group["up_day"].transform(
        lambda s: s.rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        ).sum()
    )

    result["down_days_3d"] = group["down_day"].transform(
        lambda s: s.rolling(
            window=3,
            min_periods=3
        ).sum()
    )

    result["volatility_10d"] = group["daily_return_pct"].transform(
        lambda s: s.rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        ).std()
    )

    result["avg_abs_body_3d"] = group["body_pct"].transform(
        lambda s: s.abs().rolling(
            window=3,
            min_periods=3
        ).mean()
    )

    result["avg_upper_shadow_3d"] = group[
        "upper_shadow_pct"
    ].transform(
        lambda s: s.rolling(
            window=3,
            min_periods=3
        ).mean()
    )

    result["avg_lower_shadow_3d"] = group[
        "lower_shadow_pct"
    ].transform(
        lambda s: s.rolling(
            window=3,
            min_periods=3
        ).mean()
    )

    result = result.replace([np.inf, -np.inf], np.nan)

    return result


# ============================================================
# 3. 選擇相似度要使用的特徵
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    這些特徵共同描述一個 10 日的量價／K 線型態。

    前段：當日與前日 K 線。
    中段：短中期報酬與高點回撤。
    後段：成交量、漲跌天數與波動。
    """

    desired_features = [
        "upper_shadow_pct",
        "lower_shadow_pct",
        "body_pct",
        "prev_upper_shadow_pct",
        "prev_lower_shadow_pct",
        "prev_body_pct",
        "open_pattern_pct",
        "close_pattern_pct",
        "volume_vs_ma5_ratio",
        "trend_5d_pct",
        "return_3d_pct",
        "return_5d_pct",
        "return_10d_pct",
        "drawdown_from_high_10d_pct",
        "rebound_from_low_10d_pct",
        "volume_ratio_today_10d",
        "volume_ratio_3d_10d",
        "up_days_10d",
        "down_days_3d",
        "volatility_10d",
        "avg_abs_body_3d",
        "avg_upper_shadow_3d",
        "avg_lower_shadow_3d"
    ]

    return [
        feature
        for feature in desired_features
        if feature in df.columns
    ]


# ============================================================
# 4. 由 Random Forest 學習特徵權重
# ============================================================

def learn_feature_weights(
    df: pd.DataFrame,
    feature_columns: list,
    direction: str
) -> pd.DataFrame:
    """
    看漲權重：
    bullish vs 非 bullish。

    看跌權重：
    bearish vs 非 bearish。

    得到的 normalized_weight 會用於加權歐氏距離。
    """

    valid_df = df.dropna(subset=feature_columns).copy()

    if direction == "bullish":
        y = (valid_df["target_3d"] == "bullish").astype(int)
    elif direction == "bearish":
        y = (valid_df["target_3d"] == "bearish").astype(int)
    else:
        raise ValueError("direction 必須是 bullish 或 bearish。")

    if y.nunique() < 2:
        raise RuntimeError(
            f"{direction} 類別不足，無法建立特徵權重。"
        )

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    X = valid_df[feature_columns]

    model.fit(X, y)

    weights = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "importance": model.feature_importances_
        }
    )

    total_importance = weights["importance"].sum()

    if total_importance == 0:
        weights["normalized_weight"] = (
            1 / len(weights)
        )
    else:
        weights["normalized_weight"] = (
            weights["importance"] / total_importance
        )

    weights = weights.sort_values(
        "normalized_weight",
        ascending=False
    ).reset_index(drop=True)

    return weights


# ============================================================
# 5. 將大漲／大跌候選日分群
# ============================================================

def cluster_candidates(
    candidate_df: pd.DataFrame,
    feature_columns: list,
    requested_clusters: int,
    direction: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler]:
    """
    將真正的大漲或大跌候選樣本做 KMeans 分群。

    每一群表示一種候選 K 線型態。
    群中心（centroid）則是這種型態的平均數值描述。
    """

    candidates = candidate_df.dropna(
        subset=feature_columns
    ).copy()

    if len(candidates) < 20:
        raise RuntimeError(
            f"{direction} 候選資料只有 {len(candidates)} 筆，"
            "不足以可靠分群。"
        )

    # 避免群數過大，每群至少預期約 10 個候選樣本。
    max_reasonable_clusters = max(2, len(candidates) // 10)

    actual_clusters = min(
        requested_clusters,
        max_reasonable_clusters
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        candidates[feature_columns]
    )

    kmeans = KMeans(
        n_clusters=actual_clusters,
        n_init=20,
        random_state=RANDOM_STATE
    )

    candidates["cluster_id"] = kmeans.fit_predict(X_scaled)

    # 各候選點離自己群中心的距離
    distances = np.linalg.norm(
        X_scaled - kmeans.cluster_centers_[candidates["cluster_id"]],
        axis=1
    )

    candidates["distance_to_centroid"] = distances

    candidates["similarity_to_centroid"] = (
        1 / (1 + candidates["distance_to_centroid"])
    )

    cluster_summary = (
        candidates
        .groupby("cluster_id")
        .agg(
            candidate_count=("cluster_id", "count"),
            mean_future_return_3d_pct=(
                "future_return_3d",
                lambda s: s.mean() * 100
            ),
            median_future_return_3d_pct=(
                "future_return_3d",
                lambda s: s.median() * 100
            ),
            mean_similarity_to_centroid=(
                "similarity_to_centroid",
                "mean"
            )
        )
        .reset_index()
        .sort_values("cluster_id")
    )

    cluster_summary.insert(
        0,
        "direction",
        direction
    )

    cluster_summary["n_clusters"] = actual_clusters

    return (
        candidates,
        cluster_summary,
        kmeans.cluster_centers_,
        scaler
    )


# ============================================================
# 6. 計算所有訓練資料與各群中心的相似度
# ============================================================

def build_similarity_table(
    full_df: pd.DataFrame,
    feature_columns: list,
    centroids: np.ndarray,
    weights_df: pd.DataFrame,
    direction: str,
    centroid_scaler: StandardScaler
) -> pd.DataFrame:
    """
    將各分群中心套回完整訓練資料。

    使用加權歐氏距離：
    distance = sqrt(sum(weight * (z - centroid_z)^2))

    similarity = 1 / (1 + distance)

    similarity 越接近 1，代表越像該群中心型態。
    """

    valid_full = full_df.dropna(
        subset=feature_columns
    ).copy()

    # 用「候選型態分群時」的 scaler 轉換完整資料，
    # 才能讓完整資料和 centroid 在同一尺度比較。
    X_full_scaled = centroid_scaler.transform(
        valid_full[feature_columns]
    )

    weight_array = (
        weights_df
        .set_index("feature_name")
        .reindex(feature_columns)["normalized_weight"]
        .fillna(0.0)
        .to_numpy()
    )

    if weight_array.sum() <= 0:
        weight_array = np.ones(
            len(feature_columns)
        ) / len(feature_columns)

    rows = []

    for cluster_id, centroid in enumerate(centroids):
        differences = X_full_scaled - centroid

        weighted_distance = np.sqrt(
            np.sum(
                (differences ** 2) * weight_array,
                axis=1
            )
        )

        similarity = 1 / (1 + weighted_distance)

        temp = valid_full.copy()

        temp["pattern_id"] = (
            f"{direction}_cluster_{cluster_id}"
        )

        temp["direction"] = direction
        temp["source_cluster_id"] = cluster_id
        temp["weighted_distance"] = weighted_distance
        temp["similarity"] = similarity

        rows.append(temp)

    similarity_table = pd.concat(
        rows,
        ignore_index=True
    )

    return similarity_table


# ============================================================
# 7. 同一股票的訊號冷卻期
# ============================================================

def apply_cooldown(
    matches: pd.DataFrame,
    cooldown_days: int
) -> pd.DataFrame:
    """
    同一 ticker 在 cooldown_days 個交易日內，
    若出現多個相似訊號，只保留相似度最高的一筆。

    這避免同一段連續走勢被重複算成很多次 pattern 出現。
    """

    if matches.empty:
        return matches

    result_frames = []

    for ticker, ticker_df in matches.groupby("ticker"):
        ticker_df = ticker_df.sort_values(
            ["date", "similarity"],
            ascending=[True, False]
        ).copy()

        kept_indices = []
        last_kept_position = None

        for position, (index, row) in enumerate(
            ticker_df.iterrows()
        ):
            if last_kept_position is None:
                kept_indices.append(index)
                last_kept_position = position
                continue

            if position - last_kept_position > cooldown_days:
                kept_indices.append(index)
                last_kept_position = position

        kept = ticker_df.loc[kept_indices].copy()
        result_frames.append(kept)

    return pd.concat(
        result_frames,
        ignore_index=True
    )


# ============================================================
# 8. 評估每個型態
# ============================================================

def evaluate_patterns(
    similarity_table: pd.DataFrame,
    direction: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    對每個分群中心型態：

    1. 僅保留 similarity >= MIN_SIMILARITY 的日期。
    2. 對同一股票做冷卻期去重。
    3. 計算：
       - 出現頻率
       - 大漲／大跌命中率
       - 平均未來三日報酬
       - 中位數
       - 平均相似度
       - 綜合 pattern_score
    4. 依最低頻率、最低命中率、方向性報酬篩選。
    """

    all_summaries = []
    all_matches = []

    for pattern_id, pattern_df in similarity_table.groupby(
        "pattern_id"
    ):
        matches = pattern_df[
            pattern_df["similarity"] >= MIN_SIMILARITY
        ].copy()

        if matches.empty:
            continue

        matches = apply_cooldown(
            matches=matches,
            cooldown_days=COOLDOWN_DAYS
        )

        if matches.empty:
            continue

        matches = matches.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        frequency = len(matches)

        if direction == "bullish":
            is_hit = (
                matches["future_return_3d"]
                > BULLISH_THRESHOLD
            )

            hit_rate = is_hit.mean()
            mean_directional_return = (
                matches["future_return_3d"].mean() * 100
            )
            required_hit_rate = MIN_BULLISH_HIT_RATE

        elif direction == "bearish":
            is_hit = (
                matches["future_return_3d"]
                < BEARISH_THRESHOLD
            )

            hit_rate = is_hit.mean()

            # 跌越深，方向性收益越高，故轉正數。
            mean_directional_return = (
                -matches["future_return_3d"].mean() * 100
            )
            required_hit_rate = MIN_BEARISH_HIT_RATE

        else:
            raise ValueError(
                "direction 必須是 bullish 或 bearish。"
            )

        mean_return = matches["future_return_3d"].mean() * 100
        median_return = matches["future_return_3d"].median() * 100
        std_return = matches["future_return_3d"].std() * 100
        mean_similarity = matches["similarity"].mean()
        min_similarity = matches["similarity"].min()
        max_similarity = matches["similarity"].max()

        # Pattern 分數：
        # 方向命中率 × 方向平均報酬 × 出現次數的對數 × 平均相似度
        #
        # 若方向平均報酬錯誤（看漲卻平均跌、看跌卻平均漲），
        # 分數直接視為 0。
        pattern_score = (
            hit_rate
            * max(mean_directional_return, 0)
            * np.log1p(frequency)
            * mean_similarity
        )

        cluster_id = int(
            matches["source_cluster_id"].iloc[0]
        )

        summary = {
            "pattern_id": pattern_id,
            "direction": direction,
            "source_cluster_id": cluster_id,
            "frequency": frequency,
            "hit_rate_5pct": hit_rate,
            "hit_rate_5pct_pct": hit_rate * 100,
            "required_hit_rate_pct": required_hit_rate * 100,
            "mean_future_return_3d_pct": mean_return,
            "median_future_return_3d_pct": median_return,
            "std_future_return_3d_pct": std_return,
            "mean_directional_return_pct": mean_directional_return,
            "mean_similarity": mean_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "pattern_score": pattern_score
        }

        all_summaries.append(summary)

        matches["is_direction_hit"] = is_hit.to_numpy()
        matches["pattern_score"] = pattern_score
        matches["match_rank"] = np.arange(
            1,
            len(matches) + 1
        )

        all_matches.append(matches)

    summary_df = pd.DataFrame(all_summaries)

    if summary_df.empty:
        return (
            summary_df,
            summary_df.copy(),
            pd.DataFrame()
        )

    if direction == "bullish":
        qualified_df = summary_df[
            (summary_df["frequency"] >= MIN_PATTERN_FREQUENCY)
            & (
                summary_df["hit_rate_5pct"]
                >= MIN_BULLISH_HIT_RATE
            )
            & (
                summary_df["mean_future_return_3d_pct"]
                > 0
            )
        ].copy()

    else:
        qualified_df = summary_df[
            (summary_df["frequency"] >= MIN_PATTERN_FREQUENCY)
            & (
                summary_df["hit_rate_5pct"]
                >= MIN_BEARISH_HIT_RATE
            )
            & (
                summary_df["mean_future_return_3d_pct"]
                < 0
            )
        ].copy()

    summary_df = summary_df.sort_values(
        "pattern_score",
        ascending=False
    ).reset_index(drop=True)

    qualified_df = qualified_df.sort_values(
        "pattern_score",
        ascending=False
    ).reset_index(drop=True)

    if all_matches:
        matches_df = pd.concat(
            all_matches,
            ignore_index=True
        )
    else:
        matches_df = pd.DataFrame()

    return summary_df, qualified_df, matches_df


# ============================================================
# 9. 自動產生輔助描述
# ============================================================

def create_auto_descriptions(
    pattern_summary: pd.DataFrame,
    cluster_candidates: pd.DataFrame
) -> pd.DataFrame:
    """
    以每群「真正大漲／大跌候選樣本」的平均特徵，
    產生初步描述。

    它只是一個輔助文字，最後仍要你看 K 線圖再人工命名。
    """

    if pattern_summary.empty:
        return pattern_summary

    profile = (
        cluster_candidates
        .groupby("cluster_id")
        .agg(
            cluster_return_10d_pct=("return_10d_pct", "mean"),
            cluster_drawdown_pct=(
                "drawdown_from_high_10d_pct",
                "mean"
            ),
            cluster_volume_ratio=(
                "volume_ratio_today_10d",
                "mean"
            ),
            cluster_down_days_3d=("down_days_3d", "mean"),
            cluster_upper_shadow=("upper_shadow_pct", "mean"),
            cluster_lower_shadow=("lower_shadow_pct", "mean"),
            cluster_body_pct=("body_pct", "mean")
        )
        .reset_index()
        .rename(columns={"cluster_id": "source_cluster_id"})
    )

    result = pattern_summary.merge(
        profile,
        on="source_cluster_id",
        how="left"
    )

    descriptions = []

    for _, row in result.iterrows():
        parts = []

        r10 = row.get("cluster_return_10d_pct", np.nan)
        dd = row.get("cluster_drawdown_pct", np.nan)
        vol = row.get("cluster_volume_ratio", np.nan)
        down3 = row.get("cluster_down_days_3d", np.nan)
        upper = row.get("cluster_upper_shadow", np.nan)
        lower = row.get("cluster_lower_shadow", np.nan)
        body = row.get("cluster_body_pct", np.nan)

        if pd.notna(r10):
            if r10 >= 15:
                parts.append("近10日急漲")
            elif r10 >= 5:
                parts.append("近10日上漲")
            elif r10 <= -15:
                parts.append("近10日急跌")
            elif r10 <= -5:
                parts.append("近10日走弱")

        if pd.notna(dd) and dd <= -5:
            parts.append("高點後回撤")

        if pd.notna(vol):
            if vol >= 2:
                parts.append("當日爆量")
            elif vol <= 0.7:
                parts.append("當日量縮")

        if pd.notna(down3) and down3 >= 2:
            parts.append("近3日偏弱")

        if pd.notna(upper) and upper >= 2:
            parts.append("平均長上影")

        if pd.notna(lower) and lower >= 2:
            parts.append("平均長下影")

        if pd.notna(body):
            if body >= 1.5:
                parts.append("偏紅K")
            elif body <= -1.5:
                parts.append("偏黑K")

        if not parts:
            parts.append("需搭配K線圖人工判讀")

        descriptions.append("、".join(parts))

    result["auto_description"] = descriptions

    return result


# ============================================================
# 10. 輸出資料
# ============================================================

def save_outputs(
    feature_columns: list,
    bullish_weights: pd.DataFrame,
    bearish_weights: pd.DataFrame,
    bullish_cluster_summary: pd.DataFrame,
    bearish_cluster_summary: pd.DataFrame,
    bullish_candidates_clustered: pd.DataFrame,
    bearish_candidates_clustered: pd.DataFrame,
    bullish_all_scores: pd.DataFrame,
    bearish_all_scores: pd.DataFrame,
    bullish_qualified: pd.DataFrame,
    bearish_qualified: pd.DataFrame,
    bullish_matches: pd.DataFrame,
    bearish_matches: pd.DataFrame
) -> None:
    """
    輸出所有研究中間結果與最終 Top 10。
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 特徵清單
    pd.DataFrame(
        {"feature_name": feature_columns}
    ).to_csv(
        OUTPUT_DIR / "pattern_feature_columns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 特徵權重
    bullish_weights.to_csv(
        OUTPUT_DIR / "bullish_feature_weights.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_weights.to_csv(
        OUTPUT_DIR / "bearish_feature_weights.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 分群摘要
    bullish_cluster_summary.to_csv(
        OUTPUT_DIR / "bullish_cluster_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_cluster_summary.to_csv(
        OUTPUT_DIR / "bearish_cluster_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 含分群 ID 的候選樣本，供你之後畫代表 K 線。
    bullish_clustered_output = bullish_candidates_clustered.copy()
    bearish_clustered_output = bearish_candidates_clustered.copy()

    bullish_clustered_output["date"] = (
        bullish_clustered_output["date"].dt.strftime("%Y-%m-%d")
    )

    bearish_clustered_output["date"] = (
        bearish_clustered_output["date"].dt.strftime("%Y-%m-%d")
    )

    bullish_clustered_output.to_csv(
        OUTPUT_DIR / "bullish_candidates_clustered.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_clustered_output.to_csv(
        OUTPUT_DIR / "bearish_candidates_clustered.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 全部型態評分
    bullish_all_scores.to_csv(
        OUTPUT_DIR / "bullish_all_pattern_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_all_scores.to_csv(
        OUTPUT_DIR / "bearish_all_pattern_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 合格型態排名與 Top 10
    bullish_qualified.to_csv(
        OUTPUT_DIR / "bullish_qualified_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_qualified.to_csv(
        OUTPUT_DIR / "bearish_qualified_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bullish_top = bullish_qualified.head(TOP_N_PATTERNS).copy()
    bearish_top = bearish_qualified.head(TOP_N_PATTERNS).copy()

    bullish_top.insert(
        0,
        "top_rank",
        range(1, len(bullish_top) + 1)
    )

    bearish_top.insert(
        0,
        "top_rank",
        range(1, len(bearish_top) + 1)
    )

    bullish_top.to_csv(
        OUTPUT_DIR / "top10_bullish_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_top.to_csv(
        OUTPUT_DIR / "top10_bearish_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 所有實際被相似度門檻選中的匹配案例
    if not bullish_matches.empty:
        bullish_matches_output = bullish_matches.copy()
        bullish_matches_output["date"] = (
            pd.to_datetime(bullish_matches_output["date"])
            .dt.strftime("%Y-%m-%d")
        )

        bullish_matches_output.to_csv(
            OUTPUT_DIR / "bullish_pattern_matches.csv",
            index=False,
            encoding="utf-8-sig"
        )

    if not bearish_matches.empty:
        bearish_matches_output = bearish_matches.copy()
        bearish_matches_output["date"] = (
            pd.to_datetime(bearish_matches_output["date"])
            .dt.strftime("%Y-%m-%d")
        )

        bearish_matches_output.to_csv(
            OUTPUT_DIR / "bearish_pattern_matches.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # 設定記錄
    settings = pd.DataFrame(
        {
            "setting": [
                "input_file",
                "window_size",
                "forecast_horizon",
                "bullish_threshold",
                "bearish_threshold",
                "n_clusters_bullish",
                "n_clusters_bearish",
                "min_similarity",
                "cooldown_days",
                "min_pattern_frequency",
                "min_bullish_hit_rate",
                "min_bearish_hit_rate",
                "top_n_patterns",
                "random_state"
            ],
            "value": [
                str(INPUT_FILE),
                WINDOW_SIZE,
                FORECAST_HORIZON,
                BULLISH_THRESHOLD,
                BEARISH_THRESHOLD,
                N_CLUSTERS_BULLISH,
                N_CLUSTERS_BEARISH,
                MIN_SIMILARITY,
                COOLDOWN_DAYS,
                MIN_PATTERN_FREQUENCY,
                MIN_BULLISH_HIT_RATE,
                MIN_BEARISH_HIT_RATE,
                TOP_N_PATTERNS,
                RANDOM_STATE
            ]
        }
    )

    settings.to_csv(
        OUTPUT_DIR / "pattern_discovery_settings.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 11. 主程式
# ============================================================

def main() -> None:
    print("=" * 70)
    print("K 線型態發掘：分群、相似度比對、二次篩選")
    print("=" * 70)
    print(f"輸入檔：{INPUT_FILE}")
    print(f"型態視窗：最近 {WINDOW_SIZE} 個交易日")
    print(f"相似度門檻：{MIN_SIMILARITY}")
    print(f"最低出現次數：{MIN_PATTERN_FREQUENCY}")
    print(f"看漲最低命中率：{MIN_BULLISH_HIT_RATE * 100:.1f}%")
    print(f"看跌最低命中率：{MIN_BEARISH_HIT_RATE * 100:.1f}%")

    # 1. 讀取資料
    df = load_training_data(INPUT_FILE)

    print(f"\n讀取資料列數：{len(df):,}")
    print(f"股票數：{df['ticker'].nunique()}")

    # 2. 建立多日背景特徵
    df = add_context_features(df)

    feature_columns = get_feature_columns(df)

    print(f"\n相似度特徵數：{len(feature_columns)}")
    print(feature_columns)

    valid_df = df.dropna(
        subset=feature_columns + [
            "target_3d",
            "future_return_3d"
        ]
    ).copy()

    print(f"可用型態樣本數：{len(valid_df):,}")

    # 3. 學習看漲、看跌的特徵權重
    print("\n建立看漲特徵權重...")
    bullish_weights = learn_feature_weights(
        df=valid_df,
        feature_columns=feature_columns,
        direction="bullish"
    )

    print("\n建立看跌特徵權重...")
    bearish_weights = learn_feature_weights(
        df=valid_df,
        feature_columns=feature_columns,
        direction="bearish"
    )

    print("\n看漲前 8 名特徵權重：")
    print(
        bullish_weights.head(8).to_string(index=False)
    )

    print("\n看跌前 8 名特徵權重：")
    print(
        bearish_weights.head(8).to_string(index=False)
    )

    # 4. 分開抓歷史真正大漲／大跌的候選日
    bullish_candidates = valid_df[
        valid_df["target_3d"] == "bullish"
    ].copy()

    bearish_candidates = valid_df[
        valid_df["target_3d"] == "bearish"
    ].copy()

    print(f"\n大漲候選日數：{len(bullish_candidates):,}")
    print(f"大跌候選日數：{len(bearish_candidates):,}")

    # 5. 看漲候選型態分群
    print("\n分群看漲候選型態...")

    (
        bullish_clustered,
        bullish_cluster_summary,
        bullish_centroids,
        bullish_scaler
    ) = cluster_candidates(
        candidate_df=bullish_candidates,
        feature_columns=feature_columns,
        requested_clusters=N_CLUSTERS_BULLISH,
        direction="bullish"
    )

    # 6. 看跌候選型態分群
    print("分群看跌候選型態...")

    (
        bearish_clustered,
        bearish_cluster_summary,
        bearish_centroids,
        bearish_scaler
    ) = cluster_candidates(
        candidate_df=bearish_candidates,
        feature_columns=feature_columns,
        requested_clusters=N_CLUSTERS_BEARISH,
        direction="bearish"
    )

    # 7. 把各群中心放回所有訓練資料做相似度掃描
    print("\n回測看漲型態在完整訓練資料的出現情形...")

    bullish_similarity_table = build_similarity_table(
        full_df=valid_df,
        feature_columns=feature_columns,
        centroids=bullish_centroids,
        weights_df=bullish_weights,
        direction="bullish",
        centroid_scaler=bullish_scaler
    )

    print("回測看跌型態在完整訓練資料的出現情形...")

    bearish_similarity_table = build_similarity_table(
        full_df=valid_df,
        feature_columns=feature_columns,
        centroids=bearish_centroids,
        weights_df=bearish_weights,
        direction="bearish",
        centroid_scaler=bearish_scaler
    )

    # 8. 二次篩選
    print("\n二次篩選看漲型態...")

    (
        bullish_all_scores,
        bullish_qualified,
        bullish_matches
    ) = evaluate_patterns(
        similarity_table=bullish_similarity_table,
        direction="bullish"
    )

    print("二次篩選看跌型態...")

    (
        bearish_all_scores,
        bearish_qualified,
        bearish_matches
    ) = evaluate_patterns(
        similarity_table=bearish_similarity_table,
        direction="bearish"
    )

    # 9. 產生輔助描述
    bullish_all_scores = create_auto_descriptions(
        pattern_summary=bullish_all_scores,
        cluster_candidates=bullish_clustered
    )

    bearish_all_scores = create_auto_descriptions(
        pattern_summary=bearish_all_scores,
        cluster_candidates=bearish_clustered
    )

    bullish_qualified = create_auto_descriptions(
        pattern_summary=bullish_qualified,
        cluster_candidates=bullish_clustered
    )

    bearish_qualified = create_auto_descriptions(
        pattern_summary=bearish_qualified,
        cluster_candidates=bearish_clustered
    )

    # 10. 輸出
    save_outputs(
        feature_columns=feature_columns,
        bullish_weights=bullish_weights,
        bearish_weights=bearish_weights,
        bullish_cluster_summary=bullish_cluster_summary,
        bearish_cluster_summary=bearish_cluster_summary,
        bullish_candidates_clustered=bullish_clustered,
        bearish_candidates_clustered=bearish_clustered,
        bullish_all_scores=bullish_all_scores,
        bearish_all_scores=bearish_all_scores,
        bullish_qualified=bullish_qualified,
        bearish_qualified=bearish_qualified,
        bullish_matches=bullish_matches,
        bearish_matches=bearish_matches
    )

    # 11. 結果摘要
    print("\n" + "=" * 70)
    print("型態發掘完成")
    print("=" * 70)
    print(f"看漲群數：{len(bullish_cluster_summary)}")
    print(f"看跌群數：{len(bearish_cluster_summary)}")
    print(f"看漲通過二次篩選型態數：{len(bullish_qualified)}")
    print(f"看跌通過二次篩選型態數：{len(bearish_qualified)}")

    if not bullish_qualified.empty:
        print("\n看漲合格型態前 10：")

        print(
            bullish_qualified[
                [
                    "pattern_id",
                    "frequency",
                    "hit_rate_5pct_pct",
                    "mean_future_return_3d_pct",
                    "mean_similarity",
                    "pattern_score",
                    "auto_description"
                ]
            ]
            .head(TOP_N_PATTERNS)
            .to_string(index=False)
        )
    else:
        print(
            "\n目前沒有看漲型態通過門檻。"
            "請先查看 bullish_all_pattern_scores.csv，"
            "再決定是否將 MIN_SIMILARITY 降為 0.65，"
            "或將命中率門檻改為 30%。"
        )

    if not bearish_qualified.empty:
        print("\n看跌合格型態前 10：")

        print(
            bearish_qualified[
                [
                    "pattern_id",
                    "frequency",
                    "hit_rate_5pct_pct",
                    "mean_future_return_3d_pct",
                    "mean_similarity",
                    "pattern_score",
                    "auto_description"
                ]
            ]
            .head(TOP_N_PATTERNS)
            .to_string(index=False)
        )
    else:
        print(
            "\n目前沒有看跌型態通過門檻。"
            "請先查看 bearish_all_pattern_scores.csv，"
            "再決定是否將 MIN_SIMILARITY 降為 0.65，"
            "或將命中率門檻改為 30%。"
        )

    print(f"\n輸出資料夾：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
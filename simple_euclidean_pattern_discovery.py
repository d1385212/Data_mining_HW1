from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. 基本設定
# ============================================================

# 上一步 find_big_move_candidates.py 的輸出資料夾
INPUT_DIR = Path("output_big_move_candidates")

# 已有未來三日報酬與 bullish / bearish / neutral 標籤
INPUT_FILE = INPUT_DIR / "train_with_future_return_3d.csv"

# 新的輸出資料夾，不會覆蓋舊結果
OUTPUT_DIR = Path("output_simple_euclidean_patterns")

# 題目定義：未來 3 個交易日
FORECAST_HORIZON = 3

# 大漲、大跌門檻
BULLISH_THRESHOLD = 0.05
BEARISH_THRESHOLD = -0.05

# 從所有大漲候選日、所有大跌候選日中，
# 各選未來報酬最極端的前幾個作為候選 pattern。
N_SEED_PATTERNS_PER_DIRECTION = 20

# 每個 pattern 只取前幾個「距離最近」的歷史案例，
# 避免 0.70 這種固定相似度門檻過於寬鬆。
TOP_K_SIMILAR_CASES = 30

# 排除 pattern 自己，避免 pattern 和自己距離為 0、
# 相似度為 1，造成不合理高估。
EXCLUDE_SEED_ITSELF = True

# 一個 pattern 至少要有幾個相似歷史案例，才採用。
MIN_FREQUENCY = 10

# 最低命中率。
# 這是探索門檻；題目要求 ±5%，但未必需要命中率 50%。
MIN_BULLISH_HIT_RATE = 0.20
MIN_BEARISH_HIT_RATE = 0.20

# 最終輸出前幾名
TOP_N_PATTERNS = 10

# 同一檔股票同一段連續走勢，避免重複採樣：
# 同 ticker 在此交易日距離內，只保留最相似的一筆。
COOLDOWN_TRADING_DAYS = 5


# ============================================================
# 1. 你原本的 10 個特徵
# ============================================================

FEATURE_COLUMNS = [
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_pct",
    "prev_upper_shadow_pct",
    "prev_lower_shadow_pct",
    "prev_body_pct",
    "open_pattern_pct",
    "close_pattern_pct",
    "volume_vs_ma5_ratio",
    "trend_5d_pct"
]


# ============================================================
# 2. 讀取資料
# ============================================================

def load_training_data(file_path: Path) -> pd.DataFrame:
    """
    讀取上一支程式建立的訓練資料與三日後標籤。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到輸入檔案：{file_path}\n"
            "請確認你已經跑過 find_big_move_candidates.py。"
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
            f"輸入檔缺少欄位：{missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = (
        ["close", "future_return_3d"]
        + FEATURE_COLUMNS
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

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


# ============================================================
# 3. 選擇候選 pattern
# ============================================================

def select_seed_patterns(
    df: pd.DataFrame,
    direction: str,
    n_patterns: int
) -> pd.DataFrame:
    """
    看漲：
    從 future_return_3d 最大的大漲候選日中選前 n 個。

    看跌：
    從 future_return_3d 最小的大跌候選日中選前 n 個。

    每個 seed 是一個具體的歷史 K 線型態。
    """

    if direction == "bullish":
        candidates = df[
            df["target_3d"] == "bullish"
        ].copy()

        candidates = candidates.sort_values(
            "future_return_3d",
            ascending=False
        )

    elif direction == "bearish":
        candidates = df[
            df["target_3d"] == "bearish"
        ].copy()

        candidates = candidates.sort_values(
            "future_return_3d",
            ascending=True
        )

    else:
        raise ValueError(
            "direction 必須是 bullish 或 bearish。"
        )

    seeds = candidates.head(n_patterns).copy()

    seeds = seeds.reset_index(drop=True)

    seeds["seed_rank"] = np.arange(
        1,
        len(seeds) + 1
    )

    seeds["pattern_id"] = [
        f"{direction}_seed_{rank:02d}"
        for rank in seeds["seed_rank"]
    ]

    return seeds


# ============================================================
# 4. 套用冷卻期，避免連續重複型態
# ============================================================

def apply_cooldown(
    matches: pd.DataFrame,
    cooldown_days: int
) -> pd.DataFrame:
    """
    同一檔股票內，
    在 cooldown_days 個交易日內只保留相似度最高的一次。

    例如同一檔股票連續三天都很像 seed pattern，
    不應被算成三個完全獨立的訊號。
    """

    if matches.empty:
        return matches

    kept_frames = []

    for ticker, ticker_df in matches.groupby("ticker"):
        ticker_df = ticker_df.sort_values(
            ["date", "similarity"],
            ascending=[True, False]
        ).copy()

        kept_rows = []
        last_kept_position = None

        for position, (_, row) in enumerate(
            ticker_df.iterrows()
        ):
            if last_kept_position is None:
                kept_rows.append(row)
                last_kept_position = position

            elif position - last_kept_position > cooldown_days:
                kept_rows.append(row)
                last_kept_position = position

        if kept_rows:
            kept_frames.append(
                pd.DataFrame(kept_rows)
            )

    if not kept_frames:
        return pd.DataFrame(columns=matches.columns)

    return pd.concat(
        kept_frames,
        ignore_index=True
    )


# ============================================================
# 5. 歐氏距離相似度比對
# ============================================================

def evaluate_one_seed_pattern(
    full_df: pd.DataFrame,
    full_scaled_features: np.ndarray,
    seed_row: pd.Series,
    scaler: StandardScaler,
    direction: str
) -> tuple[dict, pd.DataFrame]:
    """
    對一個 seed pattern：

    1. 計算它與所有訓練資料的歐氏距離。
    2. 轉換成 similarity = 1 / (1 + distance)。
    3. 取最相似的前 TOP_K_SIMILAR_CASES 筆。
    4. 排除 seed 本身。
    5. 做同股票冷卻期去重。
    6. 計算出現頻率、命中率與平均三日報酬。
    """

    seed_vector = seed_row[
        FEATURE_COLUMNS
    ].to_frame().T

    seed_scaled = scaler.transform(
        seed_vector
    )[0]

    # 標準歐氏距離
    distances = np.sqrt(
        np.sum(
            (full_scaled_features - seed_scaled) ** 2,
            axis=1
        )
    )

    similarity = 1 / (1 + distances)

    matches = full_df.copy()

    matches["euclidean_distance"] = distances
    matches["similarity"] = similarity

    matches["pattern_id"] = seed_row["pattern_id"]
    matches["direction"] = direction
    matches["seed_rank"] = seed_row["seed_rank"]
    matches["seed_date"] = seed_row["date"]
    matches["seed_ticker"] = seed_row["ticker"]
    matches["seed_future_return_3d"] = (
        seed_row["future_return_3d"]
    )

    matches = matches.sort_values(
        "euclidean_distance",
        ascending=True
    ).copy()

    # 不把 seed 自己當作匹配案例。
    if EXCLUDE_SEED_ITSELF:
        matches = matches[
            ~(
                (matches["date"] == seed_row["date"])
                & (
                    matches["ticker"]
                    == seed_row["ticker"]
                )
            )
        ].copy()

    # 先只取前 K 個最相似案例。
    matches = matches.head(
        TOP_K_SIMILAR_CASES
    ).copy()

    # 再做同股票的冷卻期去重。
    matches = apply_cooldown(
        matches=matches,
        cooldown_days=COOLDOWN_TRADING_DAYS
    )

    matches = matches.sort_values(
        "euclidean_distance",
        ascending=True
    ).reset_index(drop=True)

    matches["match_rank"] = np.arange(
        1,
        len(matches) + 1
    )

    frequency = len(matches)

    if frequency == 0:
        summary = {
            "pattern_id": seed_row["pattern_id"],
            "direction": direction,
            "seed_rank": seed_row["seed_rank"],
            "seed_date": seed_row["date"],
            "seed_ticker": seed_row["ticker"],
            "seed_future_return_3d_pct": (
                seed_row["future_return_3d"] * 100
            ),
            "frequency": 0,
            "hit_rate_5pct": np.nan,
            "hit_rate_5pct_pct": np.nan,
            "mean_future_return_3d_pct": np.nan,
            "median_future_return_3d_pct": np.nan,
            "std_future_return_3d_pct": np.nan,
            "mean_distance": np.nan,
            "mean_similarity": np.nan,
            "min_similarity": np.nan,
            "pattern_score": np.nan
        }

        return summary, matches

    if direction == "bullish":
        is_hit = (
            matches["future_return_3d"]
            > BULLISH_THRESHOLD
        )

        hit_rate = is_hit.mean()

        mean_directional_return = (
            matches["future_return_3d"].mean() * 100
        )

    else:
        is_hit = (
            matches["future_return_3d"]
            < BEARISH_THRESHOLD
        )

        hit_rate = is_hit.mean()

        mean_directional_return = (
            -matches["future_return_3d"].mean() * 100
        )

    matches["is_direction_hit"] = is_hit.to_numpy()

    mean_return = (
        matches["future_return_3d"].mean() * 100
    )

    median_return = (
        matches["future_return_3d"].median() * 100
    )

    std_return = (
        matches["future_return_3d"].std() * 100
    )

    mean_distance = matches["euclidean_distance"].mean()
    mean_similarity = matches["similarity"].mean()
    min_similarity = matches["similarity"].min()

    # 排名分數：
    # 命中率 × 方向平均報酬 × log(1 + 出現次數) × 平均相似度
    pattern_score = (
        hit_rate
        * max(mean_directional_return, 0)
        * np.log1p(frequency)
        * mean_similarity
    )

    summary = {
        "pattern_id": seed_row["pattern_id"],
        "direction": direction,
        "seed_rank": seed_row["seed_rank"],
        "seed_date": seed_row["date"],
        "seed_ticker": seed_row["ticker"],
        "seed_future_return_3d_pct": (
            seed_row["future_return_3d"] * 100
        ),
        "frequency": frequency,
        "hit_rate_5pct": hit_rate,
        "hit_rate_5pct_pct": hit_rate * 100,
        "mean_future_return_3d_pct": mean_return,
        "median_future_return_3d_pct": median_return,
        "std_future_return_3d_pct": std_return,
        "mean_distance": mean_distance,
        "mean_similarity": mean_similarity,
        "min_similarity": min_similarity,
        "mean_directional_return_pct": mean_directional_return,
        "pattern_score": pattern_score
    }

    return summary, matches


# ============================================================
# 6. 為所有 seed pattern 做相似度比對
# ============================================================

def evaluate_seed_patterns(
    df: pd.DataFrame,
    seeds: pd.DataFrame,
    direction: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對看漲或看跌的所有 seed pattern 逐一進行比對。
    """

    scaler = StandardScaler()

    full_scaled_features = scaler.fit_transform(
        df[FEATURE_COLUMNS]
    )

    summaries = []
    all_matches = []

    for _, seed_row in seeds.iterrows():
        summary, matches = evaluate_one_seed_pattern(
            full_df=df,
            full_scaled_features=full_scaled_features,
            seed_row=seed_row,
            scaler=scaler,
            direction=direction
        )

        summaries.append(summary)

        if not matches.empty:
            all_matches.append(matches)

    summary_df = pd.DataFrame(summaries)

    if all_matches:
        matches_df = pd.concat(
            all_matches,
            ignore_index=True
        )
    else:
        matches_df = pd.DataFrame()

    return summary_df, matches_df


# ============================================================
# 7. 建立人類可讀描述
# ============================================================

def make_auto_description(seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    根據 seed pattern 的特徵產生簡單描述，
    讓你快速挑出值得畫圖和分析的型態。
    """

    result = seed_df.copy()

    descriptions = []

    for _, row in result.iterrows():
        parts = []

        trend = row.get("trend_5d_pct", np.nan)
        volume_ratio = row.get("volume_vs_ma5_ratio", np.nan)
        upper_shadow = row.get("upper_shadow_pct", np.nan)
        lower_shadow = row.get("lower_shadow_pct", np.nan)
        body = row.get("body_pct", np.nan)
        close_pattern = row.get("close_pattern_pct", np.nan)

        if pd.notna(trend):
            if trend >= 8:
                parts.append("前五日明顯上漲")
            elif trend <= -8:
                parts.append("前五日明顯下跌")

        if pd.notna(volume_ratio):
            if volume_ratio >= 0.5:
                parts.append("相對五日均量明顯放大")
            elif volume_ratio <= -0.3:
                parts.append("相對五日均量縮")

        if pd.notna(upper_shadow) and upper_shadow >= 2:
            parts.append("長上影")

        if pd.notna(lower_shadow) and lower_shadow >= 2:
            parts.append("長下影")

        if pd.notna(body):
            if body >= 2:
                parts.append("強紅K")
            elif body <= -2:
                parts.append("強黑K")

        if pd.notna(close_pattern):
            if close_pattern >= 2:
                parts.append("強勢收漲")
            elif close_pattern <= -2:
                parts.append("明顯收跌")

        if not parts:
            parts.append("一般K線結構，需看圖判讀")

        descriptions.append("、".join(parts))

    result["auto_description"] = descriptions

    return result


# ============================================================
# 8. 輸出結果
# ============================================================

def save_outputs(
    df: pd.DataFrame,
    bullish_seeds: pd.DataFrame,
    bearish_seeds: pd.DataFrame,
    bullish_scores: pd.DataFrame,
    bearish_scores: pd.DataFrame,
    bullish_matches: pd.DataFrame,
    bearish_matches: pd.DataFrame
) -> None:
    """
    輸出候選 pattern、完整評分、Top 10 與匹配案例。
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 記錄使用特徵
    pd.DataFrame(
        {"feature_name": FEATURE_COLUMNS}
    ).to_csv(
        OUTPUT_DIR / "feature_columns_used.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Seed pattern 加上易讀描述
    bullish_seeds_output = make_auto_description(
        bullish_seeds
    )

    bearish_seeds_output = make_auto_description(
        bearish_seeds
    )

    for output_df in [
        bullish_seeds_output,
        bearish_seeds_output
    ]:
        output_df["date"] = output_df["date"].dt.strftime(
            "%Y-%m-%d"
        )

    bullish_seeds_output.to_csv(
        OUTPUT_DIR / "bullish_seed_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_seeds_output.to_csv(
        OUTPUT_DIR / "bearish_seed_patterns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 合併 seed 的簡短描述到評分表
    bullish_description = bullish_seeds_output[
        ["pattern_id", "auto_description"]
    ]

    bearish_description = bearish_seeds_output[
        ["pattern_id", "auto_description"]
    ]

    bullish_scores = bullish_scores.merge(
        bullish_description,
        on="pattern_id",
        how="left"
    )

    bearish_scores = bearish_scores.merge(
        bearish_description,
        on="pattern_id",
        how="left"
    )

    # 排序前先篩選合格型態
    bullish_scores = bullish_scores.sort_values(
        "pattern_score",
        ascending=False
    ).reset_index(drop=True)

    bearish_scores = bearish_scores.sort_values(
        "pattern_score",
        ascending=False
    ).reset_index(drop=True)

    bullish_qualified = bullish_scores[
        (bullish_scores["frequency"] >= MIN_FREQUENCY)
        & (
            bullish_scores["hit_rate_5pct"]
            >= MIN_BULLISH_HIT_RATE
        )
        & (
            bullish_scores["mean_future_return_3d_pct"]
            > 0
        )
    ].copy()

    bearish_qualified = bearish_scores[
        (bearish_scores["frequency"] >= MIN_FREQUENCY)
        & (
            bearish_scores["hit_rate_5pct"]
            >= MIN_BEARISH_HIT_RATE
        )
        & (
            bearish_scores["mean_future_return_3d_pct"]
            < 0
        )
    ].copy()

    bullish_scores.to_csv(
        OUTPUT_DIR / "bullish_all_pattern_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_scores.to_csv(
        OUTPUT_DIR / "bearish_all_pattern_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

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

    bullish_top10 = bullish_qualified.head(
        TOP_N_PATTERNS
    ).copy()

    bearish_top10 = bearish_qualified.head(
        TOP_N_PATTERNS
    ).copy()

    bullish_top10.insert(
        0,
        "top_rank",
        range(1, len(bullish_top10) + 1)
    )

    bearish_top10.insert(
        0,
        "top_rank",
        range(1, len(bearish_top10) + 1)
    )

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

    # 輸出相似案例
    if not bullish_matches.empty:
        bullish_matches_output = bullish_matches.copy()

        bullish_matches_output["date"] = (
            pd.to_datetime(
                bullish_matches_output["date"]
            ).dt.strftime("%Y-%m-%d")
        )

        bullish_matches_output["seed_date"] = (
            pd.to_datetime(
                bullish_matches_output["seed_date"]
            ).dt.strftime("%Y-%m-%d")
        )

        bullish_matches_output.to_csv(
            OUTPUT_DIR / "bullish_pattern_matches.csv",
            index=False,
            encoding="utf-8-sig"
        )

    if not bearish_matches.empty:
        bearish_matches_output = bearish_matches.copy()

        bearish_matches_output["date"] = (
            pd.to_datetime(
                bearish_matches_output["date"]
            ).dt.strftime("%Y-%m-%d")
        )

        bearish_matches_output["seed_date"] = (
            pd.to_datetime(
                bearish_matches_output["seed_date"]
            ).dt.strftime("%Y-%m-%d")
        )

        bearish_matches_output.to_csv(
            OUTPUT_DIR / "bearish_pattern_matches.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # 保存設定
    settings = pd.DataFrame(
        {
            "setting": [
                "input_file",
                "forecast_horizon",
                "bullish_threshold",
                "bearish_threshold",
                "number_of_seed_patterns_per_direction",
                "top_k_similar_cases",
                "cooldown_trading_days",
                "min_frequency",
                "min_bullish_hit_rate",
                "min_bearish_hit_rate",
                "top_n_patterns"
            ],
            "value": [
                str(INPUT_FILE),
                FORECAST_HORIZON,
                BULLISH_THRESHOLD,
                BEARISH_THRESHOLD,
                N_SEED_PATTERNS_PER_DIRECTION,
                TOP_K_SIMILAR_CASES,
                COOLDOWN_TRADING_DAYS,
                MIN_FREQUENCY,
                MIN_BULLISH_HIT_RATE,
                MIN_BEARISH_HIT_RATE,
                TOP_N_PATTERNS
            ]
        }
    )

    settings.to_csv(
        OUTPUT_DIR / "settings_used.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 9. 主程式
# ============================================================

def main() -> None:
    print("=" * 70)
    print("簡化版 K 線型態發掘：標準化歐氏距離")
    print("=" * 70)
    print(f"輸入檔：{INPUT_FILE}")
    print(f"每個方向 seed pattern 數：{N_SEED_PATTERNS_PER_DIRECTION}")
    print(f"每個 seed 取最相似案例數：{TOP_K_SIMILAR_CASES}")
    print(f"冷卻期：{COOLDOWN_TRADING_DAYS} 個交易日")
    print(f"最低頻率：{MIN_FREQUENCY}")

    # 讀取資料
    df = load_training_data(INPUT_FILE)

    print(f"\n訓練資料筆數：{len(df):,}")
    print(f"股票數：{df['ticker'].nunique()}")

    # 選取看漲與看跌 seed pattern
    bullish_seeds = select_seed_patterns(
        df=df,
        direction="bullish",
        n_patterns=N_SEED_PATTERNS_PER_DIRECTION
    )

    bearish_seeds = select_seed_patterns(
        df=df,
        direction="bearish",
        n_patterns=N_SEED_PATTERNS_PER_DIRECTION
    )

    print(f"看漲 seed pattern 數：{len(bullish_seeds)}")
    print(f"看跌 seed pattern 數：{len(bearish_seeds)}")

    # 分別計算歐氏距離與二次篩選
    print("\n開始比對看漲 K 線型態...")
    bullish_scores, bullish_matches = evaluate_seed_patterns(
        df=df,
        seeds=bullish_seeds,
        direction="bullish"
    )

    print("開始比對看跌 K 線型態...")
    bearish_scores, bearish_matches = evaluate_seed_patterns(
        df=df,
        seeds=bearish_seeds,
        direction="bearish"
    )

    # 輸出
    save_outputs(
        df=df,
        bullish_seeds=bullish_seeds,
        bearish_seeds=bearish_seeds,
        bullish_scores=bullish_scores,
        bearish_scores=bearish_scores,
        bullish_matches=bullish_matches,
        bearish_matches=bearish_matches
    )

    # 顯示摘要
    bullish_qualified = bullish_scores[
        (bullish_scores["frequency"] >= MIN_FREQUENCY)
        & (
            bullish_scores["hit_rate_5pct"]
            >= MIN_BULLISH_HIT_RATE
        )
        & (
            bullish_scores["mean_future_return_3d_pct"]
            > 0
        )
    ].copy()

    bearish_qualified = bearish_scores[
        (bearish_scores["frequency"] >= MIN_FREQUENCY)
        & (
            bearish_scores["hit_rate_5pct"]
            >= MIN_BEARISH_HIT_RATE
        )
        & (
            bearish_scores["mean_future_return_3d_pct"]
            < 0
        )
    ].copy()

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)
    print(f"看漲合格型態數：{len(bullish_qualified)}")
    print(f"看跌合格型態數：{len(bearish_qualified)}")
    print(f"輸出資料夾：{OUTPUT_DIR.resolve()}")

    print("\n請優先檢查：")
    print("1. top10_bullish_patterns.csv")
    print("2. top10_bearish_patterns.csv")
    print("3. bullish_pattern_matches.csv")
    print("4. bearish_pattern_matches.csv")


if __name__ == "__main__":
    main()
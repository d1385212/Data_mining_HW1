from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. 基本設定
# ============================================================

# 2018~2025 訓練資料所在資料夾
FEATURE_DIR = Path("output_tw_stock_features_v3_ex_right_or_split")

# 2026 測試資料
TEST_FILE = FEATURE_DIR / "test_2026.csv"

# 2018~2025 原始訓練資料
# 用於 fit StandardScaler，避免 2026 資料洩漏
TRAIN_FILE = FEATURE_DIR / "train_2018_2025.csv"

# 簡化歐氏距離程式的輸出資料夾
PATTERN_DIR = Path("output_simple_euclidean_patterns")

# 訓練期挑選出的 pattern 結果
BULLISH_TOP_FILE = PATTERN_DIR / "top10_bullish_patterns.csv"
BEARISH_TOP_FILE = PATTERN_DIR / "top10_bearish_patterns.csv"

# 每個 Top pattern 的 seed 資料來源
BULLISH_SEED_FILE = PATTERN_DIR / "bullish_seed_patterns.csv"
BEARISH_SEED_FILE = PATTERN_DIR / "bearish_seed_patterns.csv"

# 本程式輸出資料夾
OUTPUT_DIR = Path("output_2026_pattern_test")

# 題目定義：預測未來 3 個交易日
FORECAST_HORIZON = 3

# 題目定義
BULLISH_THRESHOLD = 0.05
BEARISH_THRESHOLD = -0.05

# 必須與 simple_euclidean_pattern_discovery.py 一致：
# 每個 pattern 在訓練期比對時，取前 30 個相似案例
TOP_K_SIMILAR_CASES = 30

# 2026 中，每個 pattern 也只取最接近的前 K 筆，
# 避免某個 pattern 對大量普通交易日都發出訊號。
TOP_K_TEST_SIGNALS_PER_PATTERN = 30

# 只保留與訓練期 pattern 至少這麼相似的測試訊號。
# 訓練期若使用「前 30 個相似案例」，會自動計算每個 pattern
# 的訓練期最弱入選相似度，並用它作為主要門檻。
# 這個固定下限是額外防呆。
MIN_TEST_SIMILARITY_FLOOR = 0.20

# 同一股票在接近日期重複出現同型態時，
# 只保留最相似的那一筆。
COOLDOWN_TRADING_DAYS = 5


# ============================================================
# 1. 與前一支程式完全相同的 10 個 K 線特徵
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
# 2. 讀取並清理資料
# ============================================================

def load_feature_data(
    file_path: Path,
    data_name: str
) -> pd.DataFrame:
    """
    讀取訓練或測試資料。

    需要：
    - date
    - ticker
    - close
    - 10 個 K 線特徵
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 {data_name}：{file_path}"
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
            f"{data_name} 缺少欄位：{missing_columns}"
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


def add_test_future_return(
    test_df: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """
    對每檔股票計算 2026 的未來 3 個交易日報酬。

    future_return_3d：
    (Close_t+3 / Close_t) - 1

    最後 3 個交易日因沒有完整未來資料，會是 NaN。
    """

    result = test_df.copy()

    result = (
        result
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    group = result.groupby("ticker", group_keys=False)

    result["future_close_3d"] = group["close"].shift(-horizon)

    result["future_return_3d"] = (
        result["future_close_3d"] / result["close"] - 1
    )

    result["future_return_3d_pct"] = (
        result["future_return_3d"] * 100
    )

    return result


# ============================================================
# 3. 讀取 Top pattern 與 seed K 線
# ============================================================

def load_pattern_files(
    top_file: Path,
    seed_file: Path,
    direction: str
) -> pd.DataFrame:
    """
    讀取實際通過篩選的 Top pattern，
    再從 seed patterns 檔取得該 pattern 的 10 項特徵。

    即使檔案名稱叫 top10，
    內部只有 5 個、4 個 pattern 也完全可執行。
    """

    if not top_file.exists():
        raise FileNotFoundError(
            f"找不到 {direction} Top pattern 檔案：{top_file}"
        )

    if not seed_file.exists():
        raise FileNotFoundError(
            f"找不到 {direction} seed pattern 檔案：{seed_file}"
        )

    top_patterns = pd.read_csv(
        top_file,
        encoding="utf-8-sig"
    )

    seed_patterns = pd.read_csv(
        seed_file,
        encoding="utf-8-sig"
    )

    if top_patterns.empty:
        print(f"警告：{direction} Top pattern 檔案為空。")
        return pd.DataFrame()

    required_top_columns = [
        "pattern_id",
        "seed_date",
        "seed_ticker"
    ]

    missing_top_columns = [
        column
        for column in required_top_columns
        if column not in top_patterns.columns
    ]

    if missing_top_columns:
        raise ValueError(
            f"{direction} Top pattern 缺少欄位："
            f"{missing_top_columns}"
        )

    required_seed_columns = [
        "pattern_id",
        "date",
        "ticker"
    ] + FEATURE_COLUMNS

    missing_seed_columns = [
        column
        for column in required_seed_columns
        if column not in seed_patterns.columns
    ]

    if missing_seed_columns:
        raise ValueError(
            f"{direction} seed pattern 缺少欄位："
            f"{missing_seed_columns}"
        )

    seed_patterns["date"] = pd.to_datetime(
        seed_patterns["date"],
        errors="coerce"
    )

    # 只保留訓練期最後選中的 pattern
    patterns = top_patterns.merge(
        seed_patterns,
        on="pattern_id",
        how="left",
        suffixes=("_top", "_seed")
    )

    # 合併後可能有 date_seed、ticker_seed，
    # 統一整理為 seed_date、seed_ticker。
    if "date" in patterns.columns:
        patterns["seed_date_from_seed_file"] = patterns["date"]

    if "ticker" in patterns.columns:
        patterns["seed_ticker_from_seed_file"] = patterns["ticker"]

    # 10 特徵轉數值
    for column in FEATURE_COLUMNS:
        patterns[column] = pd.to_numeric(
            patterns[column],
            errors="coerce"
        )

    patterns = patterns.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    patterns["direction"] = direction

    return patterns


# ============================================================
# 4. 訓練期取得每個 pattern 的相似度門檻
# ============================================================

def calculate_training_similarity_thresholds(
    train_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    對每個 seed pattern，在 2018~2025 訓練資料重新算距離。

    取訓練期距離最近的 TOP_K_SIMILAR_CASES 筆（排除 seed 自己），
    其中最低的 similarity，作為此 pattern 在 2026 的動態門檻。

    為什麼這樣做：
    - 不同 pattern 的自然分散程度不同。
    - 不適合所有 pattern 共用單一固定 0.70 門檻。
    - 2026 只有達到「至少與訓練期入選案例同等相似」的訊號才保留。
    """

    train_scaled = scaler.transform(
        train_df[FEATURE_COLUMNS]
    )

    rows = []

    for _, pattern in patterns_df.iterrows():
        seed_vector = pattern[
            FEATURE_COLUMNS
        ].to_frame().T

        seed_scaled = scaler.transform(seed_vector)[0]

        distances = np.sqrt(
            np.sum(
                (train_scaled - seed_scaled) ** 2,
                axis=1
            )
        )

        similarities = 1 / (1 + distances)

        temp = train_df[
            ["date", "ticker"]
        ].copy()

        temp["distance"] = distances
        temp["similarity"] = similarities

        # 排除 seed 自己
        seed_date = pd.to_datetime(
            pattern["seed_date"]
        )

        seed_ticker = str(
            pattern["seed_ticker"]
        )

        temp = temp[
            ~(
                (temp["date"] == seed_date)
                & (temp["ticker"] == seed_ticker)
            )
        ].copy()

        temp = temp.sort_values(
            "distance",
            ascending=True
        ).head(TOP_K_SIMILAR_CASES)

        if temp.empty:
            train_similarity_floor = np.nan
            train_mean_similarity = np.nan
            train_mean_distance = np.nan
        else:
            train_similarity_floor = temp["similarity"].min()
            train_mean_similarity = temp["similarity"].mean()
            train_mean_distance = temp["distance"].mean()

        rows.append(
            {
                "pattern_id": pattern["pattern_id"],
                "training_similarity_floor": train_similarity_floor,
                "training_mean_similarity": train_mean_similarity,
                "training_mean_distance": train_mean_distance
            }
        )

    thresholds = pd.DataFrame(rows)

    return thresholds


# ============================================================
# 5. 同股票冷卻期去重
# ============================================================

def apply_cooldown(
    matches: pd.DataFrame,
    cooldown_days: int
) -> pd.DataFrame:
    """
    同一檔股票在接近日期的訊號，只保留相似度最高者。

    例如：
    同一檔股票連續三天都很像 bearish pattern，
    不將其視為三個獨立訊號。
    """

    if matches.empty:
        return matches

    kept_frames = []

    for ticker, ticker_df in matches.groupby("ticker"):
        ticker_df = ticker_df.sort_values(
            ["date", "similarity"],
            ascending=[True, False]
        ).copy()

        kept_indices = []
        last_kept_position = None

        for position, (index, _) in enumerate(
            ticker_df.iterrows()
        ):
            if last_kept_position is None:
                kept_indices.append(index)
                last_kept_position = position

            elif position - last_kept_position > cooldown_days:
                kept_indices.append(index)
                last_kept_position = position

        kept_frames.append(
            ticker_df.loc[kept_indices].copy()
        )

    return pd.concat(
        kept_frames,
        ignore_index=True
    )


# ============================================================
# 6. 用 Top patterns 掃描 2026
# ============================================================

def test_patterns_on_2026(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
    direction: str,
    scaler: StandardScaler
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對每個 Top pattern：

    1. 使用訓練資料 fitted scaler 標準化。
    2. 算 2026 每日 K 線與 seed 的歐氏距離。
    3. 用訓練期相似度門檻篩選。
    4. 每個 pattern 最多保留 30 個最相似測試訊號。
    5. 同一檔股票做冷卻期去重。
    6. 統計 2026 命中率與平均未來三日報酬。
    """

    if patterns_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    thresholds = calculate_training_similarity_thresholds(
        train_df=train_df,
        patterns_df=patterns_df,
        scaler=scaler
    )

    patterns = patterns_df.merge(
        thresholds,
        on="pattern_id",
        how="left"
    )

    test_scaled = scaler.transform(
        test_df[FEATURE_COLUMNS]
    )

    all_signals = []
    summary_rows = []

    for _, pattern in patterns.iterrows():
        seed_vector = pattern[
            FEATURE_COLUMNS
        ].to_frame().T

        seed_scaled = scaler.transform(seed_vector)[0]

        distances = np.sqrt(
            np.sum(
                (test_scaled - seed_scaled) ** 2,
                axis=1
            )
        )

        similarities = 1 / (1 + distances)

        signals = test_df.copy()

        signals["pattern_id"] = pattern["pattern_id"]
        signals["direction"] = direction
        signals["seed_date"] = pattern["seed_date"]
        signals["seed_ticker"] = pattern["seed_ticker"]
        signals["euclidean_distance"] = distances
        signals["similarity"] = similarities

        # 使用訓練期該 pattern 第 30 名相似案例的 similarity
        # 作為 2026 的動態門檻。
        dynamic_threshold = pattern[
            "training_similarity_floor"
        ]

        if pd.isna(dynamic_threshold):
            dynamic_threshold = MIN_TEST_SIMILARITY_FLOOR

        # 加上固定下限，避免門檻過低
        effective_threshold = max(
            dynamic_threshold,
            MIN_TEST_SIMILARITY_FLOOR
        )

        signals["training_similarity_threshold"] = (
            effective_threshold
        )

        signals = signals[
            signals["similarity"] >= effective_threshold
        ].copy()

        # 只保留有未來三日結果的訊號
        signals = signals.dropna(
            subset=["future_return_3d"]
        ).copy()

        # 取最相似前 K 筆
        signals = signals.sort_values(
            "euclidean_distance",
            ascending=True
        ).head(TOP_K_TEST_SIGNALS_PER_PATTERN)

        # 同一股票冷卻期去重
        signals = apply_cooldown(
            matches=signals,
            cooldown_days=COOLDOWN_TRADING_DAYS
        )

        signals = signals.sort_values(
            "euclidean_distance",
            ascending=True
        ).reset_index(drop=True)

        signals["signal_rank"] = np.arange(
            1,
            len(signals) + 1
        )

        frequency = len(signals)

        if frequency == 0:
            summary_rows.append(
                {
                    "pattern_id": pattern["pattern_id"],
                    "direction": direction,
                    "seed_date": pattern["seed_date"],
                    "seed_ticker": pattern["seed_ticker"],
                    "training_frequency": pattern.get(
                        "frequency",
                        np.nan
                    ),
                    "training_hit_rate_5pct_pct": pattern.get(
                        "hit_rate_5pct_pct",
                        np.nan
                    ),
                    "training_mean_future_return_3d_pct": pattern.get(
                        "mean_future_return_3d_pct",
                        np.nan
                    ),
                    "training_similarity_threshold": (
                        effective_threshold
                    ),
                    "test_signal_count": 0,
                    "test_hit_count": 0,
                    "test_hit_rate_5pct_pct": np.nan,
                    "test_mean_future_return_3d_pct": np.nan,
                    "test_median_future_return_3d_pct": np.nan,
                    "test_mean_similarity": np.nan,
                    "test_min_similarity": np.nan
                }
            )

            continue

        if direction == "bullish":
            is_hit = (
                signals["future_return_3d"]
                > BULLISH_THRESHOLD
            )
        else:
            is_hit = (
                signals["future_return_3d"]
                < BEARISH_THRESHOLD
            )

        signals["is_direction_hit"] = is_hit.to_numpy()

        hit_count = int(is_hit.sum())
        hit_rate = is_hit.mean()

        summary_rows.append(
            {
                "pattern_id": pattern["pattern_id"],
                "direction": direction,
                "seed_date": pattern["seed_date"],
                "seed_ticker": pattern["seed_ticker"],
                "training_frequency": pattern.get(
                    "frequency",
                    np.nan
                ),
                "training_hit_rate_5pct_pct": pattern.get(
                    "hit_rate_5pct_pct",
                    np.nan
                ),
                "training_mean_future_return_3d_pct": pattern.get(
                    "mean_future_return_3d_pct",
                    np.nan
                ),
                "training_similarity_threshold": (
                    effective_threshold
                ),
                "test_signal_count": frequency,
                "test_hit_count": hit_count,
                "test_hit_rate_5pct_pct": hit_rate * 100,
                "test_mean_future_return_3d_pct": (
                    signals["future_return_3d"].mean() * 100
                ),
                "test_median_future_return_3d_pct": (
                    signals["future_return_3d"].median() * 100
                ),
                "test_std_future_return_3d_pct": (
                    signals["future_return_3d"].std() * 100
                ),
                "test_mean_similarity": (
                    signals["similarity"].mean()
                ),
                "test_min_similarity": (
                    signals["similarity"].min()
                )
            }
        )

        all_signals.append(signals)

    summary_df = pd.DataFrame(summary_rows)

    if all_signals:
        signals_df = pd.concat(
            all_signals,
            ignore_index=True
        )
    else:
        signals_df = pd.DataFrame()

    return summary_df, signals_df


# ============================================================
# 7. 輸出結果
# ============================================================

def save_outputs(
    test_with_future_return: pd.DataFrame,
    bullish_summary: pd.DataFrame,
    bearish_summary: pd.DataFrame,
    bullish_signals: pd.DataFrame,
    bearish_signals: pd.DataFrame
) -> None:
    """
    輸出：

    - test_2026_with_future_return.csv
    - bullish_2026_pattern_summary.csv
    - bearish_2026_pattern_summary.csv
    - bullish_2026_signals.csv
    - bearish_2026_signals.csv
    - overall_2026_test_summary.csv
    - settings_used.csv
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2026 加上未來三日報酬
    test_output = test_with_future_return.copy()
    test_output["date"] = test_output["date"].dt.strftime(
        "%Y-%m-%d"
    )

    test_output.to_csv(
        OUTPUT_DIR / "test_2026_with_future_return.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 個別 pattern 的 2026 統計
    bullish_summary.to_csv(
        OUTPUT_DIR / "bullish_2026_pattern_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bearish_summary.to_csv(
        OUTPUT_DIR / "bearish_2026_pattern_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 個別訊號資料
    if not bullish_signals.empty:
        bullish_signals_output = bullish_signals.copy()

        bullish_signals_output["date"] = (
            pd.to_datetime(
                bullish_signals_output["date"]
            ).dt.strftime("%Y-%m-%d")
        )

        bullish_signals_output["seed_date"] = (
            pd.to_datetime(
                bullish_signals_output["seed_date"]
            ).dt.strftime("%Y-%m-%d")
        )

        bullish_signals_output.to_csv(
            OUTPUT_DIR / "bullish_2026_signals.csv",
            index=False,
            encoding="utf-8-sig"
        )

    if not bearish_signals.empty:
        bearish_signals_output = bearish_signals.copy()

        bearish_signals_output["date"] = (
            pd.to_datetime(
                bearish_signals_output["date"]
            ).dt.strftime("%Y-%m-%d")
        )

        bearish_signals_output["seed_date"] = (
            pd.to_datetime(
                bearish_signals_output["seed_date"]
            ).dt.strftime("%Y-%m-%d")
        )

        bearish_signals_output.to_csv(
            OUTPUT_DIR / "bearish_2026_signals.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # 看漲與看跌合併，方便報告比較
    comparison = pd.concat(
        [bullish_summary, bearish_summary],
        ignore_index=True
    )

    comparison.to_csv(
        OUTPUT_DIR / "all_patterns_2026_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 全部測試期訊號的總體摘要
    overall_rows = []

    for direction, signals in [
        ("bullish", bullish_signals),
        ("bearish", bearish_signals)
    ]:
        if signals.empty:
            overall_rows.append(
                {
                    "direction": direction,
                    "signal_count": 0,
                    "hit_count": 0,
                    "hit_rate_5pct_pct": np.nan,
                    "mean_future_return_3d_pct": np.nan,
                    "median_future_return_3d_pct": np.nan
                }
            )
            continue

        hit_count = int(
            signals["is_direction_hit"].sum()
        )

        overall_rows.append(
            {
                "direction": direction,
                "signal_count": len(signals),
                "hit_count": hit_count,
                "hit_rate_5pct_pct": (
                    hit_count / len(signals) * 100
                ),
                "mean_future_return_3d_pct": (
                    signals["future_return_3d"].mean() * 100
                ),
                "median_future_return_3d_pct": (
                    signals["future_return_3d"].median() * 100
                )
            }
        )

    overall_summary = pd.DataFrame(overall_rows)

    overall_summary.to_csv(
        OUTPUT_DIR / "overall_2026_test_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 設定紀錄
    settings = pd.DataFrame(
        {
            "setting": [
                "train_file",
                "test_file",
                "bullish_top_file",
                "bearish_top_file",
                "forecast_horizon",
                "bullish_threshold",
                "bearish_threshold",
                "top_k_train_similar_cases",
                "top_k_test_signals_per_pattern",
                "min_test_similarity_floor",
                "cooldown_trading_days"
            ],
            "value": [
                str(TRAIN_FILE),
                str(TEST_FILE),
                str(BULLISH_TOP_FILE),
                str(BEARISH_TOP_FILE),
                FORECAST_HORIZON,
                BULLISH_THRESHOLD,
                BEARISH_THRESHOLD,
                TOP_K_SIMILAR_CASES,
                TOP_K_TEST_SIGNALS_PER_PATTERN,
                MIN_TEST_SIMILARITY_FLOOR,
                COOLDOWN_TRADING_DAYS
            ]
        }
    )

    settings.to_csv(
        OUTPUT_DIR / "settings_used.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 8. 主程式
# ============================================================

def main() -> None:
    print("=" * 70)
    print("2026 樣本外測試：Top K 線型態歐氏距離比對")
    print("=" * 70)

    # 讀取訓練與測試資料
    train_df = load_feature_data(
        file_path=TRAIN_FILE,
        data_name="2018~2025 訓練資料"
    )

    test_df = load_feature_data(
        file_path=TEST_FILE,
        data_name="2026 測試資料"
    )

    # 為 2026 每檔股票建立三日後報酬
    test_df = add_test_future_return(
        test_df=test_df,
        horizon=FORECAST_HORIZON
    )

    print(f"訓練資料列數：{len(train_df):,}")
    print(f"2026 測試資料列數：{len(test_df):,}")
    print(f"2026 股票數：{test_df['ticker'].nunique()}")

    # 只在訓練資料 fit，再 transform 訓練與測試資料
    scaler = StandardScaler()

    scaler.fit(
        train_df[FEATURE_COLUMNS]
    )

    # 讀取實際存在的看漲、看跌 Top pattern
    bullish_patterns = load_pattern_files(
        top_file=BULLISH_TOP_FILE,
        seed_file=BULLISH_SEED_FILE,
        direction="bullish"
    )

    bearish_patterns = load_pattern_files(
        top_file=BEARISH_TOP_FILE,
        seed_file=BEARISH_SEED_FILE,
        direction="bearish"
    )

    print(f"讀取看漲 pattern 數：{len(bullish_patterns)}")
    print(f"讀取看跌 pattern 數：{len(bearish_patterns)}")

    # 測試看漲 pattern
    print("\n開始掃描 2026 看漲型態...")

    bullish_summary, bullish_signals = test_patterns_on_2026(
        train_df=train_df,
        test_df=test_df,
        patterns_df=bullish_patterns,
        direction="bullish",
        scaler=scaler
    )

    # 測試看跌 pattern
    print("開始掃描 2026 看跌型態...")

    bearish_summary, bearish_signals = test_patterns_on_2026(
        train_df=train_df,
        test_df=test_df,
        patterns_df=bearish_patterns,
        direction="bearish",
        scaler=scaler
    )

    # 儲存
    save_outputs(
        test_with_future_return=test_df,
        bullish_summary=bullish_summary,
        bearish_summary=bearish_summary,
        bullish_signals=bullish_signals,
        bearish_signals=bearish_signals
    )

    # 顯示最終摘要
    print("\n" + "=" * 70)
    print("2026 樣本外測試完成")
    print("=" * 70)

    if not bullish_summary.empty:
        print("\n看漲型態 2026 摘要：")

        display_cols = [
            "pattern_id",
            "test_signal_count",
            "test_hit_rate_5pct_pct",
            "test_mean_future_return_3d_pct",
            "test_mean_similarity"
        ]

        print(
            bullish_summary[
                [
                    column
                    for column in display_cols
                    if column in bullish_summary.columns
                ]
            ].to_string(index=False)
        )

    if not bearish_summary.empty:
        print("\n看跌型態 2026 摘要：")

        display_cols = [
            "pattern_id",
            "test_signal_count",
            "test_hit_rate_5pct_pct",
            "test_mean_future_return_3d_pct",
            "test_mean_similarity"
        ]

        print(
            bearish_summary[
                [
                    column
                    for column in display_cols
                    if column in bearish_summary.columns
                ]
            ].to_string(index=False)
        )

    print(f"\n輸出資料夾：{OUTPUT_DIR.resolve()}")

    print("\n最重要的結果檔案：")
    print("1. overall_2026_test_summary.csv")
    print("2. bullish_2026_pattern_summary.csv")
    print("3. bearish_2026_pattern_summary.csv")
    print("4. bullish_2026_signals.csv")
    print("5. bearish_2026_signals.csv")


if __name__ == "__main__":
    main()
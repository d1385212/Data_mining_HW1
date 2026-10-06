from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. 基本設定
# ============================================================

# find_big_move_candidates.py 的輸出：
# 已含 train 期 future_return_3d 與 target_3d。
CANDIDATE_DIR = Path("output_big_move_candidates")

TRAIN_FILE = (
    CANDIDATE_DIR / "train_with_future_return_3d.csv"
)

# build_tw_features_v3.py 的輸出：
# 2026 尚未有 future_return_3d，本程式會自行計算。
FEATURE_DIR = Path(
    "output_tw_stock_features_v3_ex_right_or_split"
)

TEST_FILE = FEATURE_DIR / "test_2026.csv"

# 本程式輸出資料夾
OUTPUT_DIR = Path("output_rule_based_pattern_test")

# 題目規定：預測 3 個交易日後
FORECAST_HORIZON = 3

# 題目規定的大漲／大跌門檻
BULLISH_THRESHOLD = 0.05
BEARISH_THRESHOLD = -0.05

# ------------------------------------------------------------
# 規則門檻：可在此調整
# ------------------------------------------------------------

# 大紅 K／大黑 K：
# 實體線占當日收盤價的比例至少 2%
BIG_BODY_PCT = 2.0

# 爆量：
# volume_vs_ma5_ratio = (當日量 - 五日均量) / 當日量
#
# >= 0.50 代表：
# 五日均量 <= 當日量的 50%，
# 即當日量至少約為五日均量的 2 倍。
VOLUME_SURGE_RATIO = 0.50

# 量縮：
VOLUME_CONTRACTION_RATIO = -0.30

# 長上影、長下影：占收盤價至少 2%
LONG_SHADOW_PCT = 2.0

# 高檔／先上漲：
# 前五日趨勢至少上漲 3%
UPTREND_5D_PCT = 3.0

# 上升趨勢量縮回檔：
# 前五日趨勢至少 +3%，近期收跌或當日黑 K
PULLBACK_BODY_PCT = -0.5

# 最低出現次數：
# 少於此數的型態仍會輸出，但結果可能不穩定。
MIN_FREQUENCY_FOR_RELIABLE_RESULT = 10


# ============================================================
# 1. 使用的特徵欄位
# ============================================================

REQUIRED_FEATURE_COLUMNS = [
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_pct",
    "open_pattern_pct",
    "close_pattern_pct",
    "volume_vs_ma5_ratio",
    "trend_5d_pct"
]


# ============================================================
# 2. 讀取資料
# ============================================================

def load_data(
    file_path: Path,
    is_training_data: bool
) -> pd.DataFrame:
    """
    讀取訓練或測試資料並檢查必要欄位。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到檔案：{file_path}"
        )

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    required_columns = [
        "date",
        "ticker",
        "close"
    ] + REQUIRED_FEATURE_COLUMNS

    if is_training_data:
        required_columns += [
            "future_return_3d",
            "target_3d"
        ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"檔案缺少欄位：{missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = [
        "close"
    ] + REQUIRED_FEATURE_COLUMNS

    if "future_return_3d" in df.columns:
        numeric_columns.append(
            "future_return_3d"
        )

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
        ] + REQUIRED_FEATURE_COLUMNS
    ).copy()

    df = (
        df
        .sort_values(["ticker", "date"])
        .drop_duplicates(
            subset=["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    return df


def add_future_return(
    df: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """
    以每檔股票的收盤價計算 t+3 日報酬。

    公式：
    future_return_3d = Close(t+3) / Close(t) - 1
    """

    result = df.copy()

    result = (
        result
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    group = result.groupby(
        "ticker",
        group_keys=False
    )

    result["future_close_3d"] = (
        group["close"].shift(-horizon)
    )

    result["future_return_3d"] = (
        result["future_close_3d"]
        / result["close"]
        - 1
    )

    result["future_return_3d_pct"] = (
        result["future_return_3d"] * 100
    )

    result["target_3d"] = "unknown"

    has_future = result[
        "future_return_3d"
    ].notna()

    result.loc[
        has_future,
        "target_3d"
    ] = "neutral"

    result.loc[
        result["future_return_3d"]
        > BULLISH_THRESHOLD,
        "target_3d"
    ] = "bullish"

    result.loc[
        result["future_return_3d"]
        < BEARISH_THRESHOLD,
        "target_3d"
    ] = "bearish"

    return result


# ============================================================
# 3. 建立規則型態
# ============================================================

def add_rule_based_patterns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    建立你想研究的可解釋 K 線型態。

    每個 pattern 欄位為 True / False。
    """

    result = df.copy()

    # --------------------------------------------------------
    # 1. 爆量大紅 K
    #
    # 實體至少 +2%
    # 且成交量至少約為五日均量 2 倍
    # --------------------------------------------------------
    result["pattern_volume_big_red"] = (
        (result["body_pct"] >= BIG_BODY_PCT)
        & (
            result["volume_vs_ma5_ratio"]
            >= VOLUME_SURGE_RATIO
        )
        & (result["close_pattern_pct"] > 0)
    )

    # --------------------------------------------------------
    # 2. 爆量大黑／大綠 K
    #
    # 實體至少 -2%
    # 且成交量至少約為五日均量 2 倍
    # --------------------------------------------------------
    result["pattern_volume_big_black"] = (
        (result["body_pct"] <= -BIG_BODY_PCT)
        & (
            result["volume_vs_ma5_ratio"]
            >= VOLUME_SURGE_RATIO
        )
        & (result["close_pattern_pct"] < 0)
    )

    # --------------------------------------------------------
    # 3. 高檔爆量長上影
    #
    # 前五日先上漲至少 3%
    # 當日爆量
    # 上影線至少 2%
    # --------------------------------------------------------
    result["pattern_high_volume_long_upper"] = (
        (result["trend_5d_pct"] >= UPTREND_5D_PCT)
        & (
            result["volume_vs_ma5_ratio"]
            >= VOLUME_SURGE_RATIO
        )
        & (
            result["upper_shadow_pct"]
            >= LONG_SHADOW_PCT
        )
    )

    # --------------------------------------------------------
    # 4. 高檔爆量長上影 + 黑 K
    #
    # 這是較嚴格版的「爆量滯漲／出貨疑慮」。
    # --------------------------------------------------------
    result["pattern_high_volume_upper_black"] = (
        result["pattern_high_volume_long_upper"]
        & (result["body_pct"] <= 0)
    )

    # --------------------------------------------------------
    # 5. 上升趨勢中的量縮回檔 + 長下影
    #
    # 前五日上漲
    # 當日量縮
    # 小黑／黑 K 或回檔
    # 長下影表示下方可能有承接
    # --------------------------------------------------------
    result["pattern_uptrend_pullback_lower_shadow"] = (
        (result["trend_5d_pct"] >= UPTREND_5D_PCT)
        & (
            result["volume_vs_ma5_ratio"]
            <= VOLUME_CONTRACTION_RATIO
        )
        & (result["body_pct"] <= PULLBACK_BODY_PCT)
        & (
            result["lower_shadow_pct"]
            >= LONG_SHADOW_PCT
        )
    )

    # --------------------------------------------------------
    # 6. 長下影止跌紅 K
    #
    # 前五日偏弱
    # 當日出現長下影
    # 收紅或接近收紅
    # --------------------------------------------------------
    result["pattern_long_lower_reversal"] = (
        (result["trend_5d_pct"] <= -UPTREND_5D_PCT)
        & (
            result["lower_shadow_pct"]
            >= LONG_SHADOW_PCT
        )
        & (result["body_pct"] >= 0)
    )

    return result


# ============================================================
# 4. 計算單一規則的結果
# ============================================================

def evaluate_one_pattern(
    df: pd.DataFrame,
    pattern_column: str,
    dataset_name: str,
    baseline_bullish_rate: float,
    baseline_bearish_rate: float
) -> tuple[dict, pd.DataFrame]:
    """
    計算單一規則型態：

    - 出現次數
    - 未來三日大漲 > 5% 次數與機率
    - 未來三日大跌 < -5% 次數與機率
    - Bull Lift / Bear Lift
    - 平均、中位數三日報酬
    """

    valid_df = df.dropna(
        subset=["future_return_3d"]
    ).copy()

    matches = valid_df[
        valid_df[pattern_column]
    ].copy()

    frequency = len(matches)

    if frequency == 0:
        summary = {
            "dataset": dataset_name,
            "pattern_column": pattern_column,
            "frequency": 0,
            "bullish_hit_count": 0,
            "bearish_hit_count": 0,
            "bullish_hit_rate_pct": np.nan,
            "bearish_hit_rate_pct": np.nan,
            "bullish_lift": np.nan,
            "bearish_lift": np.nan,
            "mean_future_return_3d_pct": np.nan,
            "median_future_return_3d_pct": np.nan,
            "std_future_return_3d_pct": np.nan,
            "reliable_frequency": False
        }

        return summary, matches

    bullish_hit = (
        matches["future_return_3d"]
        > BULLISH_THRESHOLD
    )

    bearish_hit = (
        matches["future_return_3d"]
        < BEARISH_THRESHOLD
    )

    bullish_rate = bullish_hit.mean()
    bearish_rate = bearish_hit.mean()

    summary = {
        "dataset": dataset_name,
        "pattern_column": pattern_column,
        "frequency": frequency,
        "bullish_hit_count": int(bullish_hit.sum()),
        "bearish_hit_count": int(bearish_hit.sum()),
        "bullish_hit_rate_pct": bullish_rate * 100,
        "bearish_hit_rate_pct": bearish_rate * 100,
        "baseline_bullish_rate_pct": (
            baseline_bullish_rate * 100
        ),
        "baseline_bearish_rate_pct": (
            baseline_bearish_rate * 100
        ),
        "bullish_lift": (
            bullish_rate / baseline_bullish_rate
            if baseline_bullish_rate > 0
            else np.nan
        ),
        "bearish_lift": (
            bearish_rate / baseline_bearish_rate
            if baseline_bearish_rate > 0
            else np.nan
        ),
        "mean_future_return_3d_pct": (
            matches["future_return_3d"].mean() * 100
        ),
        "median_future_return_3d_pct": (
            matches["future_return_3d"].median() * 100
        ),
        "std_future_return_3d_pct": (
            matches["future_return_3d"].std() * 100
        ),
        "reliable_frequency": (
            frequency >= MIN_FREQUENCY_FOR_RELIABLE_RESULT
        )
    }

    matches["is_bullish_hit"] = bullish_hit.to_numpy()
    matches["is_bearish_hit"] = bearish_hit.to_numpy()

    return summary, matches


# ============================================================
# 5. 規則名稱與預期方向
# ============================================================

PATTERN_INFO = {
    "pattern_volume_big_red": {
        "pattern_name": "爆量大紅K",
        "expected_direction": "bullish",
        "description": (
            "實體紅K至少 2%，"
            "當日量至少約為五日均量 2 倍"
        )
    },
    "pattern_volume_big_black": {
        "pattern_name": "爆量大黑K",
        "expected_direction": "bearish",
        "description": (
            "實體黑K至少 -2%，"
            "當日量至少約為五日均量 2 倍"
        )
    },
    "pattern_high_volume_long_upper": {
        "pattern_name": "高檔爆量長上影",
        "expected_direction": "bearish",
        "description": (
            "前五日上漲至少 3%，"
            "當日爆量且上影線至少 2%"
        )
    },
    "pattern_high_volume_upper_black": {
        "pattern_name": "高檔爆量長上影黑K",
        "expected_direction": "bearish",
        "description": (
            "前五日上漲、爆量、長上影，"
            "且當日收黑或平盤"
        )
    },
    "pattern_uptrend_pullback_lower_shadow": {
        "pattern_name": "上升趨勢量縮回檔長下影",
        "expected_direction": "bullish",
        "description": (
            "前五日上漲、量縮回檔、"
            "黑K且有長下影"
        )
    },
    "pattern_long_lower_reversal": {
        "pattern_name": "下跌後長下影止跌紅K",
        "expected_direction": "bullish",
        "description": (
            "前五日下跌，"
            "長下影且當日收紅"
        )
    }
}


# ============================================================
# 6. 針對一個資料集跑完整規則測試
# ============================================================

def evaluate_all_patterns(
    df: pd.DataFrame,
    dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對某一期間（訓練期或測試期）：
    - 先算全體基準率
    - 再測所有規則型態
    """

    valid_df = df.dropna(
        subset=["future_return_3d"]
    ).copy()

    baseline_bullish_rate = (
        valid_df["future_return_3d"]
        > BULLISH_THRESHOLD
    ).mean()

    baseline_bearish_rate = (
        valid_df["future_return_3d"]
        < BEARISH_THRESHOLD
    ).mean()

    summaries = []
    all_matches = []

    for pattern_column, info in PATTERN_INFO.items():
        summary, matches = evaluate_one_pattern(
            df=df,
            pattern_column=pattern_column,
            dataset_name=dataset_name,
            baseline_bullish_rate=baseline_bullish_rate,
            baseline_bearish_rate=baseline_bearish_rate
        )

        summary["pattern_name"] = info["pattern_name"]
        summary["expected_direction"] = (
            info["expected_direction"]
        )
        summary["rule_description"] = (
            info["description"]
        )

        summaries.append(summary)

        if not matches.empty:
            matches["pattern_name"] = (
                info["pattern_name"]
            )
            matches["expected_direction"] = (
                info["expected_direction"]
            )
            all_matches.append(matches)

    summary_df = pd.DataFrame(summaries)

    # 依「符合預期方向」的 Lift 排序
    summary_df["expected_lift"] = np.where(
        summary_df["expected_direction"] == "bullish",
        summary_df["bullish_lift"],
        summary_df["bearish_lift"]
    )

    summary_df["expected_hit_rate_pct"] = np.where(
        summary_df["expected_direction"] == "bullish",
        summary_df["bullish_hit_rate_pct"],
        summary_df["bearish_hit_rate_pct"]
    )

    summary_df["expected_direction_return_pct"] = np.where(
        summary_df["expected_direction"] == "bullish",
        summary_df["mean_future_return_3d_pct"],
        -summary_df["mean_future_return_3d_pct"]
    )

    summary_df["direction_is_consistent"] = np.where(
        summary_df["expected_direction"] == "bullish",
        summary_df["mean_future_return_3d_pct"] > 0,
        summary_df["mean_future_return_3d_pct"] < 0
    )

    summary_df = summary_df.sort_values(
        [
            "direction_is_consistent",
            "expected_lift",
            "frequency"
        ],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    if all_matches:
        matches_df = pd.concat(
            all_matches,
            ignore_index=True
        )
    else:
        matches_df = pd.DataFrame()

    return summary_df, matches_df


# ============================================================
# 7. 產生訓練期與測試期對照表
# ============================================================

def create_train_test_comparison(
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    合併 2018~2025 訓練期與 2026 測試期，
    方便寫報告。
    """

    train_cols = [
        "pattern_column",
        "pattern_name",
        "expected_direction",
        "rule_description",
        "frequency",
        "expected_hit_rate_pct",
        "expected_lift",
        "mean_future_return_3d_pct",
        "direction_is_consistent"
    ]

    test_cols = [
        "pattern_column",
        "frequency",
        "expected_hit_rate_pct",
        "expected_lift",
        "mean_future_return_3d_pct",
        "direction_is_consistent"
    ]

    train_part = train_summary[
        train_cols
    ].copy()

    test_part = test_summary[
        test_cols
    ].copy()

    train_part = train_part.rename(
        columns={
            "frequency": "train_frequency",
            "expected_hit_rate_pct": (
                "train_expected_hit_rate_pct"
            ),
            "expected_lift": "train_expected_lift",
            "mean_future_return_3d_pct": (
                "train_mean_future_return_3d_pct"
            ),
            "direction_is_consistent": (
                "train_direction_is_consistent"
            )
        }
    )

    test_part = test_part.rename(
        columns={
            "frequency": "test_frequency",
            "expected_hit_rate_pct": (
                "test_expected_hit_rate_pct"
            ),
            "expected_lift": "test_expected_lift",
            "mean_future_return_3d_pct": (
                "test_mean_future_return_3d_pct"
            ),
            "direction_is_consistent": (
                "test_direction_is_consistent"
            )
        }
    )

    comparison = train_part.merge(
        test_part,
        on="pattern_column",
        how="left"
    )

    comparison = comparison.sort_values(
        "test_expected_lift",
        ascending=False
    ).reset_index(drop=True)

    return comparison


# ============================================================
# 8. 儲存 CSV
# ============================================================

def save_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    train_matches: pd.DataFrame,
    test_matches: pd.DataFrame
) -> None:
    """
    儲存所有結果。
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 設定紀錄
    settings = pd.DataFrame(
        {
            "setting": [
                "forecast_horizon",
                "bullish_threshold",
                "bearish_threshold",
                "big_body_pct",
                "volume_surge_ratio",
                "volume_contraction_ratio",
                "long_shadow_pct",
                "uptrend_5d_pct",
                "minimum_reliable_frequency"
            ],
            "value": [
                FORECAST_HORIZON,
                BULLISH_THRESHOLD,
                BEARISH_THRESHOLD,
                BIG_BODY_PCT,
                VOLUME_SURGE_RATIO,
                VOLUME_CONTRACTION_RATIO,
                LONG_SHADOW_PCT,
                UPTREND_5D_PCT,
                MIN_FREQUENCY_FOR_RELIABLE_RESULT
            ]
        }
    )

    settings.to_csv(
        OUTPUT_DIR / "rule_settings.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 2026 含未來三日報酬資料
    test_output = test_df.copy()
    test_output["date"] = (
        test_output["date"]
        .dt.strftime("%Y-%m-%d")
    )

    test_output.to_csv(
        OUTPUT_DIR / "test_2026_with_future_return.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 訓練、測試與比較表
    train_summary.to_csv(
        OUTPUT_DIR / "rule_pattern_train_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_summary.to_csv(
        OUTPUT_DIR / "rule_pattern_2026_test_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    comparison.to_csv(
        OUTPUT_DIR / "rule_pattern_train_test_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 個別匹配事件，供你看圖或查股票
    if not train_matches.empty:
        train_matches_output = train_matches.copy()
        train_matches_output["date"] = (
            pd.to_datetime(
                train_matches_output["date"]
            )
            .dt.strftime("%Y-%m-%d")
        )

        train_matches_output.to_csv(
            OUTPUT_DIR / "rule_pattern_train_matches.csv",
            index=False,
            encoding="utf-8-sig"
        )

    if not test_matches.empty:
        test_matches_output = test_matches.copy()
        test_matches_output["date"] = (
            pd.to_datetime(
                test_matches_output["date"]
            )
            .dt.strftime("%Y-%m-%d")
        )

        test_matches_output.to_csv(
            OUTPUT_DIR / "rule_pattern_2026_matches.csv",
            index=False,
            encoding="utf-8-sig"
        )


# ============================================================
# 9. 主程式
# ============================================================

def main() -> None:
    print("=" * 72)
    print("規則型 K 線型態檢定：爆量紅K、爆量黑K、長上影")
    print("=" * 72)

    # --------------------------------------------------------
    # 讀取訓練期
    # --------------------------------------------------------
    train_df = load_data(
        file_path=TRAIN_FILE,
        is_training_data=True
    )

    # 訓練檔本來已有 future_return_3d，
    # 這裡補齊 percentage 欄位即可。
    train_df["future_return_3d_pct"] = (
        train_df["future_return_3d"] * 100
    )

    # --------------------------------------------------------
    # 讀取測試期並自行計算 future return
    # --------------------------------------------------------
    test_df = load_data(
        file_path=TEST_FILE,
        is_training_data=False
    )

    test_df = add_future_return(
        df=test_df,
        horizon=FORECAST_HORIZON
    )

    # --------------------------------------------------------
    # 將相同規則加到訓練與測試資料
    # --------------------------------------------------------
    train_df = add_rule_based_patterns(train_df)
    test_df = add_rule_based_patterns(test_df)

    print(f"訓練期資料列數：{len(train_df):,}")
    print(f"2026 測試資料列數：{len(test_df):,}")

    # --------------------------------------------------------
    # 分別評估訓練期與 2026
    # --------------------------------------------------------
    train_summary, train_matches = evaluate_all_patterns(
        df=train_df,
        dataset_name="train_2018_2025"
    )

    test_summary, test_matches = evaluate_all_patterns(
        df=test_df,
        dataset_name="test_2026"
    )

    comparison = create_train_test_comparison(
        train_summary=train_summary,
        test_summary=test_summary
    )

    # --------------------------------------------------------
    # 輸出
    # --------------------------------------------------------
    save_outputs(
        train_df=train_df,
        test_df=test_df,
        train_summary=train_summary,
        test_summary=test_summary,
        comparison=comparison,
        train_matches=train_matches,
        test_matches=test_matches
    )

    # --------------------------------------------------------
    # 終端機顯示最重要的 2026 結果
    # --------------------------------------------------------
    display_columns = [
        "pattern_name",
        "expected_direction",
        "frequency",
        "expected_hit_rate_pct",
        "expected_lift",
        "mean_future_return_3d_pct",
        "direction_is_consistent"
    ]

    print("\n" + "=" * 72)
    print("2026 規則型態測試結果")
    print("=" * 72)

    print(
        test_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n判讀原則：")
    print("- expected_lift > 1：型態出現時，比一般交易日更容易達成預期事件")
    print("- direction_is_consistent = True：平均三日報酬方向與預期一致")
    print("- frequency >= 10：樣本數較有基本參考價值")
    print(f"\n輸出資料夾：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. 基本設定
# ============================================================

# 這個資料夾必須和 build_tw_features_v3.py 的 OUTPUT_DIR 一致。
INPUT_DIR = Path("output_tw_stock_features_v3_ex_right_or_split")

# 本程式輸出的資料夾。
# 和原始訓練資料分開，避免覆蓋。
OUTPUT_DIR = Path("output_big_move_candidates")

# 訓練資料檔案。
TRAIN_FILE = INPUT_DIR / "train_2018_2025.csv"

# 預測未來幾個「交易日」。
FORECAST_HORIZON = 3

# 題目定義：
# 3 天後收盤價比當日收盤價上漲超過 5% = bullish。
# 3 天後收盤價比當日收盤價下跌超過 5% = bearish。
BULLISH_THRESHOLD = 0.05
BEARISH_THRESHOLD = -0.05

# 是否要求未來 3 個交易日不能落在除權息／除權或分割排除窗口。
# 你的 train_2018_2025.csv 本身已排除事件窗口，
# 但原始特徵若是由事件日前後資料計算，仍可能有影響。
# 第一版設定 True，採比較保守的樣本篩選。
EXCLUDE_FUTURE_EVENT_WINDOW = True


# ============================================================
# 1. 讀取與檢查資料
# ============================================================

def load_training_data(file_path: Path) -> pd.DataFrame:
    """
    讀取 train_2018_2025.csv，並檢查必要欄位。

    必要欄位：
    - date
    - ticker
    - close

    若檔案仍保留 is_corporate_action_window，
    後面會用它檢查未來 3 日是否碰到事件窗口。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到訓練資料：{file_path}\n"
            "請確認 INPUT_DIR 是否和前一支資料建立程式的 OUTPUT_DIR 相同。"
        )

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    required_columns = [
        "date",
        "ticker",
        "close"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "訓練資料缺少必要欄位："
            f"{missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["ticker"] = df["ticker"].astype(str).str.strip()

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "ticker", "close"]
    ).copy()

    df = (
        df
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"])
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 2. 建立未來 3 日報酬與標籤
# ============================================================

def create_future_return_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    bullish_threshold: float = 0.05,
    bearish_threshold: float = -0.05
) -> pd.DataFrame:
    """
    對每檔股票分別建立：

    future_close_3d：
    第 t + 3 個交易日的收盤價。

    future_return_3d：
    (Close_t+3 - Close_t) / Close_t

    target_3d：
    bullish：未來 3 日報酬 > +5%
    bearish：未來 3 日報酬 < -5%
    neutral：其餘情況
    unknown：資料尾端沒有 t+3 收盤價
    """

    result = df.copy()

    result = (
        result
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    group = result.groupby("ticker", group_keys=False)

    # 取得同一檔股票的 3 個交易日後收盤價。
    result["future_close_3d"] = group["close"].shift(-horizon)

    # 未來 3 日報酬。
    result["future_return_3d"] = (
        result["future_close_3d"] / result["close"] - 1
    )

    # 百分比格式，方便 Excel 和報告閱讀。
    result["future_return_3d_pct"] = (
        result["future_return_3d"] * 100
    )

    # 預設未知：通常是各股票最後 3 個交易日。
    result["target_3d"] = "unknown"

    # 有完整未來資料時，先設成 neutral。
    has_future_data = result["future_return_3d"].notna()

    result.loc[
        has_future_data,
        "target_3d"
    ] = "neutral"

    # 看漲：未來 3 日上漲超過 5%。
    result.loc[
        result["future_return_3d"] > bullish_threshold,
        "target_3d"
    ] = "bullish"

    # 看跌：未來 3 日下跌超過 5%。
    result.loc[
        result["future_return_3d"] < bearish_threshold,
        "target_3d"
    ] = "bearish"

    # Bool 欄位，之後程式使用更方便。
    result["is_bullish_candidate"] = (
        result["target_3d"] == "bullish"
    )

    result["is_bearish_candidate"] = (
        result["target_3d"] == "bearish"
    )

    return result


# ============================================================
# 3. 保守檢查：未來 3 天是否碰到事件窗口
# ============================================================

def add_future_event_window_flags(
    df: pd.DataFrame,
    horizon: int = 3
) -> pd.DataFrame:
    """
    檢查今天之後的 1、2、3 個交易日內，
    是否有任何一天落入 is_corporate_action_window。

    若沒有 is_corporate_action_window 欄位，
    則假設訓練資料已經清理過，全部設為 False。

    注意：
    train_2018_2025.csv 已排除了事件窗口日期，
    因此此步多數情況下都會是 False；
    它主要是保留完整研究紀錄與防呆。
    """

    result = df.copy()

    if "is_corporate_action_window" not in result.columns:
        result["future_has_event_window"] = False
        return result

    result["is_corporate_action_window"] = (
        result["is_corporate_action_window"]
        .fillna(False)
        .astype(bool)
    )

    group = result.groupby("ticker", group_keys=False)

    future_event_columns = []

    for day in range(1, horizon + 1):
        column_name = f"event_window_t_plus_{day}"

        result[column_name] = (
            group["is_corporate_action_window"]
            .shift(-day)
            .fillna(False)
            .astype(bool)
        )

        future_event_columns.append(column_name)

    result["future_has_event_window"] = (
        result[future_event_columns]
        .any(axis=1)
    )

    return result


# ============================================================
# 4. 產生可用標籤資料與候選樣本
# ============================================================

def create_candidate_datasets(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    回傳：

    labeled_data：
    有完整 future_close_3d 的所有資料，
    包含 bullish / bearish / neutral。

    bullish_candidates：
    未來 3 日上漲超過 5% 的候選日。

    bearish_candidates：
    未來 3 日下跌超過 5% 的候選日。
    """

    labeled_data = df[
        df["target_3d"] != "unknown"
    ].copy()

    if EXCLUDE_FUTURE_EVENT_WINDOW:
        labeled_data = labeled_data[
            ~labeled_data["future_has_event_window"]
        ].copy()

    bullish_candidates = labeled_data[
        labeled_data["target_3d"] == "bullish"
    ].copy()

    bearish_candidates = labeled_data[
        labeled_data["target_3d"] == "bearish"
    ].copy()

    return labeled_data, bullish_candidates, bearish_candidates


# ============================================================
# 5. 統計與輸出
# ============================================================

def save_outputs(
    all_labeled_data: pd.DataFrame,
    bullish_candidates: pd.DataFrame,
    bearish_candidates: pd.DataFrame
) -> None:
    """
    輸出：

    1. train_with_future_return_3d.csv
       所有有完整未來三日資料的訓練樣本，
       內含 future_close_3d / future_return_3d / target_3d。

    2. bullish_candidates_3d_gt_5pct.csv
       未來三日上漲超過 5% 的候選日。

    3. bearish_candidates_3d_lt_minus_5pct.csv
       未來三日下跌超過 5% 的候選日。

    4. candidate_summary_by_ticker.csv
       每檔股票的大漲、大跌、中性樣本數與比例。

    5. overall_candidate_summary.csv
       所有股票合併後的統計摘要。

    6. settings_used.csv
       本次標籤設定紀錄。
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 日期輸出成 YYYY-MM-DD，方便 Excel 閱讀。
    labeled_output = all_labeled_data.copy()
    bullish_output = bullish_candidates.copy()
    bearish_output = bearish_candidates.copy()

    for output_df in [
        labeled_output,
        bullish_output,
        bearish_output
    ]:
        output_df["date"] = output_df["date"].dt.strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # 1. 全部具標籤資料
    # --------------------------------------------------------
    labeled_output.to_csv(
        OUTPUT_DIR / "train_with_future_return_3d.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 2. 看漲候選日
    # --------------------------------------------------------
    bullish_output.to_csv(
        OUTPUT_DIR / "bullish_candidates_3d_gt_5pct.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 3. 看跌候選日
    # --------------------------------------------------------
    bearish_output.to_csv(
        OUTPUT_DIR / "bearish_candidates_3d_lt_minus_5pct.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 4. 每檔股票統計
    # --------------------------------------------------------
    ticker_summary = (
        all_labeled_data
        .groupby(
            ["ticker", "stock_id", "name", "rank"],
            dropna=False
        )
        .agg(
            total_labeled_days=("target_3d", "count"),
            bullish_count=("is_bullish_candidate", "sum"),
            bearish_count=("is_bearish_candidate", "sum"),
            neutral_count=(
                "target_3d",
                lambda series: (series == "neutral").sum()
            ),
            avg_future_return_3d_pct=(
                "future_return_3d_pct",
                "mean"
            ),
            median_future_return_3d_pct=(
                "future_return_3d_pct",
                "median"
            ),
            max_future_return_3d_pct=(
                "future_return_3d_pct",
                "max"
            ),
            min_future_return_3d_pct=(
                "future_return_3d_pct",
                "min"
            )
        )
        .reset_index()
        .sort_values("rank")
    )

    ticker_summary["bullish_rate_pct"] = (
        ticker_summary["bullish_count"]
        / ticker_summary["total_labeled_days"]
        * 100
    )

    ticker_summary["bearish_rate_pct"] = (
        ticker_summary["bearish_count"]
        / ticker_summary["total_labeled_days"]
        * 100
    )

    ticker_summary.to_csv(
        OUTPUT_DIR / "candidate_summary_by_ticker.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 5. 全體摘要
    # --------------------------------------------------------
    total_count = len(all_labeled_data)
    bullish_count = len(bullish_candidates)
    bearish_count = len(bearish_candidates)
    neutral_count = int(
        (all_labeled_data["target_3d"] == "neutral").sum()
    )

    overall_summary = pd.DataFrame(
        {
            "metric": [
                "total_labeled_days",
                "bullish_count",
                "bearish_count",
                "neutral_count",
                "bullish_rate_pct",
                "bearish_rate_pct",
                "neutral_rate_pct",
                "mean_future_return_3d_pct",
                "median_future_return_3d_pct"
            ],
            "value": [
                total_count,
                bullish_count,
                bearish_count,
                neutral_count,
                bullish_count / total_count * 100 if total_count else np.nan,
                bearish_count / total_count * 100 if total_count else np.nan,
                neutral_count / total_count * 100 if total_count else np.nan,
                all_labeled_data["future_return_3d_pct"].mean(),
                all_labeled_data["future_return_3d_pct"].median()
            ]
        }
    )

    overall_summary.to_csv(
        OUTPUT_DIR / "overall_candidate_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 6. 本次設定紀錄
    # --------------------------------------------------------
    settings = pd.DataFrame(
        {
            "setting": [
                "input_train_file",
                "forecast_horizon_trading_days",
                "bullish_threshold",
                "bearish_threshold",
                "exclude_future_event_window"
            ],
            "value": [
                str(TRAIN_FILE),
                FORECAST_HORIZON,
                BULLISH_THRESHOLD,
                BEARISH_THRESHOLD,
                EXCLUDE_FUTURE_EVENT_WINDOW
            ]
        }
    )

    settings.to_csv(
        OUTPUT_DIR / "settings_used.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 終端機印出摘要
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("大漲／大跌候選樣本建立完成")
    print("=" * 60)
    print(f"有完整未來 {FORECAST_HORIZON} 日資料的樣本數：{total_count:,}")
    print(
        f"看漲候選（未來 {FORECAST_HORIZON} 日 > +5%）："
        f"{bullish_count:,} 筆 "
        f"({bullish_count / total_count * 100:.2f}%)"
        if total_count else "看漲候選：0 筆"
    )
    print(
        f"看跌候選（未來 {FORECAST_HORIZON} 日 < -5%）："
        f"{bearish_count:,} 筆 "
        f"({bearish_count / total_count * 100:.2f}%)"
        if total_count else "看跌候選：0 筆"
    )
    print(
        f"中性樣本（介於 -5% 與 +5%）："
        f"{neutral_count:,} 筆 "
        f"({neutral_count / total_count * 100:.2f}%)"
        if total_count else "中性樣本：0 筆"
    )
    print(f"輸出資料夾：{OUTPUT_DIR.resolve()}")


# ============================================================
# 6. 主程式
# ============================================================

def main() -> None:
    print("=" * 60)
    print("開始建立未來 3 日大漲／大跌候選樣本")
    print("=" * 60)
    print(f"訓練資料來源：{TRAIN_FILE}")
    print(f"預測期間：未來 {FORECAST_HORIZON} 個交易日")
    print(f"看漲條件：未來報酬 > {BULLISH_THRESHOLD * 100:.1f}%")
    print(f"看跌條件：未來報酬 < {BEARISH_THRESHOLD * 100:.1f}%")

    # 1. 讀取已完成特徵工程的訓練資料
    train_data = load_training_data(TRAIN_FILE)

    print(f"\n讀取成功：{len(train_data):,} 筆訓練資料")
    print(f"股票數：{train_data['ticker'].nunique()}")

    # 2. 計算同一檔股票的三日後收盤價與未來三日報酬
    labeled_data = create_future_return_labels(
        df=train_data,
        horizon=FORECAST_HORIZON,
        bullish_threshold=BULLISH_THRESHOLD,
        bearish_threshold=BEARISH_THRESHOLD
    )

    # 3. 檢查未來三個交易日是否碰到事件窗口
    labeled_data = add_future_event_window_flags(
        df=labeled_data,
        horizon=FORECAST_HORIZON
    )

    # 4. 建立大漲、大跌候選資料
    all_labeled_data, bullish_candidates, bearish_candidates = (
        create_candidate_datasets(labeled_data)
    )

    # 5. 輸出 CSV 與統計資料
    save_outputs(
        all_labeled_data=all_labeled_data,
        bullish_candidates=bullish_candidates,
        bearish_candidates=bearish_candidates
    )


if __name__ == "__main__":
    main()
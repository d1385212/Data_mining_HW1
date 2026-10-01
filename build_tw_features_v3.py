from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 0. 基本設定
# ============================================================

START_DATE = "2018-01-01"

# yfinance 的 end 日期不包含當天。
# 要抓完整 2026 年，所以用 2027-01-01。
END_DATE = "2027-01-01"

TRAIN_START = "2018-01-01"
TRAIN_END = "2025-12-31"

TEST_START = "2026-01-01"
TEST_END = "2026-12-31"

# 使用新輸出資料夾，避免覆蓋舊版 CSV。
OUTPUT_DIR = Path("output_tw_stock_features_v3_ex_right_or_split")

MARKET_SUFFIX = ".TW"

# 每次下載 10 檔，比較不容易遇到 Yahoo 或 yfinance 限制。
BATCH_SIZE = 10

# False 可降低 yfinance SQLite cache 被鎖定的機率。
YFINANCE_THREADS = False

# True：排除事件日前 1 個交易日、事件當日、事件後 1 個交易日。
# False：只排除事件當日。
EXCLUDE_EVENT_WINDOW = True

# 1 = 事件前 1 日、事件日、事件後 1 日。
EVENT_WINDOW_TRADING_DAYS = 1


# ============================================================
# 1. 股票代碼處理
# ============================================================

def normalize_tw_ticker(stock_id: str, suffix: str = ".TW") -> str:
    """
    將上市股票代碼轉為 Yahoo Finance 格式。

    例：
    2330 -> 2330.TW
    1101 -> 1101.TW
    """
    stock_id = str(stock_id).strip().upper()

    if stock_id.endswith(".TW") or stock_id.endswith(".TWO"):
        return stock_id

    return f"{stock_id}{suffix}"


# ============================================================
# 2. 市值排名第 51～100 名股票池
# ============================================================

def get_rank_51_100_universe() -> pd.DataFrame:
    """
    目前使用的市值排名第 51～100 名上市股票池。

    注意：
    市值排行會隨市場價格變動。本清單是目前研究使用的快照。
    """

    rank_51_100 = [
        (51, "1326", "台化"),
        (52, "3481", "群創"),
        (53, "2379", "瑞昱"),
        (54, "4904", "遠傳"),
        (55, "6770", "力積電"),
        (56, "3661", "世芯-KY"),
        (57, "2449", "京元電子"),
        (58, "3034", "聯詠"),
        (59, "2615", "萬海"),
        (60, "2313", "華通"),
        (61, "2801", "彰銀"),
        (62, "2002", "中鋼"),
        (63, "2207", "和泰車"),
        (64, "1590", "亞德客-KY"),
        (65, "3044", "健鼎"),
        (66, "1519", "華城"),
        (67, "2337", "旺宏"),
        (68, "3036", "文曄"),
        (69, "4938", "和碩"),
        (70, "2376", "技嘉"),
        (71, "2356", "英業達"),
        (72, "2618", "長榮航"),
        (73, "6515", "穎崴"),
        (74, "2912", "統一超"),
        (75, "5876", "上海商銀"),
        (76, "6239", "力成"),
        (77, "2609", "陽明"),
        (78, "5871", "中租-KY"),
        (79, "2404", "漢唐"),
        (80, "2409", "友達"),
        (81, "6213", "聯茂"),
        (82, "3533", "嘉澤"),
        (83, "1101", "台泥"),
        (84, "2324", "仁寶"),
        (85, "6139", "亞翔"),
        (86, "1802", "台玻"),
        (87, "2834", "臺企銀"),
        (88, "1605", "華新"),
        (89, "1504", "東元"),
        (90, "3702", "大聯大"),
        (91, "7750", "新代"),
        (92, "6415", "矽力*-KY"),
        (93, "6805", "富世達"),
        (94, "1402", "遠東新"),
        (95, "6531", "愛普*"),
        (96, "6789", "采鈺"),
        (97, "6919", "康霈*"),
        (98, "3532", "台勝科"),
        (99, "2347", "聯強"),
        (100, "2492", "華新科"),
    ]

    universe = pd.DataFrame(
        rank_51_100,
        columns=["rank", "stock_id", "name"]
    )

    universe["market_cap_date"] = "current_snapshot"

    universe["ticker"] = universe["stock_id"].apply(
        lambda x: normalize_tw_ticker(x, suffix=MARKET_SUFFIX)
    )

    return universe


# ============================================================
# 3. 下載日 K、成交量、股利與除權／分割資料
# ============================================================

def download_price_data(
    tickers: list,
    start_date: str,
    end_date: str,
    batch_size: int = 10
) -> tuple[pd.DataFrame, list]:
    """
    從 Yahoo Finance 下載：

    - Open
    - High
    - Low
    - Close
    - Adj Close
    - Volume
    - Dividends
    - Stock Splits

    注意：
    Stock Splits 在台股環境可能代表股票股利除權或拆併股，
    後續以 ex_right_or_split 統稱。
    """

    all_frames = []
    failed_tickers = []

    tickers = list(dict.fromkeys(tickers))
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for start_index in range(0, len(tickers), batch_size):
        batch = tickers[start_index:start_index + batch_size]
        batch_no = start_index // batch_size + 1

        print(
            f"下載第 {batch_no}/{total_batches} 批："
            f"{len(batch)} 檔，{batch[0]} ~ {batch[-1]}"
        )

        try:
            raw = yf.download(
                tickers=batch,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                actions=True,
                group_by="column",
                threads=YFINANCE_THREADS,
                progress=False,
                timeout=30
            )
        except Exception as error:
            print(f"此批下載失敗：{error}")
            failed_tickers.extend(batch)
            sleep(2)
            continue

        if raw is None or raw.empty:
            print("此批沒有取得資料。")
            failed_tickers.extend(batch)
            sleep(2)
            continue

        # 一次只下載一檔時的欄位格式
        if len(batch) == 1:
            ticker = batch[0]
            temp = raw.copy().reset_index()

            temp.columns = [
                str(column).lower().replace(" ", "_")
                for column in temp.columns
            ]

            if "close" not in temp.columns:
                failed_tickers.append(ticker)
                continue

            if temp["close"].dropna().empty:
                failed_tickers.append(ticker)
                continue

            temp["ticker"] = ticker

            for column in ["adj_close", "dividends", "stock_splits"]:
                if column not in temp.columns:
                    temp[column] = 0.0

            temp = temp[
                [
                    "date",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "dividends",
                    "stock_splits"
                ]
            ]

            all_frames.append(temp)

        # 一次下載多檔時的 MultiIndex 格式
        else:
            if not isinstance(raw.columns, pd.MultiIndex):
                print("Yahoo 回傳欄位格式異常，本批略過。")
                failed_tickers.extend(batch)
                sleep(2)
                continue

            available_tickers = raw.columns.get_level_values(-1).unique()

            for ticker in batch:
                if ticker not in available_tickers:
                    failed_tickers.append(ticker)
                    continue

                try:
                    temp = raw.xs(
                        ticker,
                        axis=1,
                        level=-1
                    ).copy()
                except KeyError:
                    failed_tickers.append(ticker)
                    continue

                temp = temp.dropna(how="all")

                if temp.empty:
                    failed_tickers.append(ticker)
                    continue

                if "Close" not in temp.columns:
                    failed_tickers.append(ticker)
                    continue

                if temp["Close"].dropna().empty:
                    failed_tickers.append(ticker)
                    continue

                temp = temp.reset_index()

                temp.columns = [
                    str(column).lower().replace(" ", "_")
                    for column in temp.columns
                ]

                temp["ticker"] = ticker

                for column in ["adj_close", "dividends", "stock_splits"]:
                    if column not in temp.columns:
                        temp[column] = 0.0

                temp = temp[
                    [
                        "date",
                        "ticker",
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj_close",
                        "volume",
                        "dividends",
                        "stock_splits"
                    ]
                ]

                all_frames.append(temp)

        sleep(1)

    if not all_frames:
        raise RuntimeError(
            "完全沒有下載到資料。請檢查網路、Yahoo Finance 狀態或股票代碼。"
        )

    prices = pd.concat(all_frames, ignore_index=True)

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce"
    )

    # 若 date 含時區，移除時區避免後續比較日期時出錯。
    try:
        prices["date"] = prices["date"].dt.tz_localize(None)
    except TypeError:
        pass

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits"
    ]

    for column in numeric_columns:
        prices[column] = pd.to_numeric(
            prices[column],
            errors="coerce"
        )

    prices["dividends"] = prices["dividends"].fillna(0.0)
    prices["stock_splits"] = prices["stock_splits"].fillna(0.0)

    prices = (
        prices
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"])
        .reset_index(drop=True)
    )

    return prices, sorted(set(failed_tickers))


# ============================================================
# 4. 標記除息、除權或分割事件
# ============================================================

def add_corporate_action_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    建立事件欄位：

    is_ex_dividend：
    Yahoo Finance Dividends > 0，代表除息／現金股利事件。

    is_ex_right_or_split：
    Yahoo Finance Stock Splits 不等於 0 且不等於 1。
    對台股可能代表：
    - 股票股利造成的除權
    - 真正股票分割
    - 反分割

    所以統一命名為 ex_right_or_split。

    is_corporate_action_day：
    除息或除權／分割事件日。

    is_corporate_action_window：
    若設定 EVENT_WINDOW_TRADING_DAYS = 1，
    則事件日前一交易日、事件日、後一交易日全部標記 True。
    """

    result = df.copy()

    result["dividends"] = pd.to_numeric(
        result["dividends"],
        errors="coerce"
    ).fillna(0.0)

    result["stock_splits"] = pd.to_numeric(
        result["stock_splits"],
        errors="coerce"
    ).fillna(0.0)

    # 除息／現金股利
    result["is_ex_dividend"] = result["dividends"] > 0

    # 除權／股票股利／股票分割／反分割
    result["is_ex_right_or_split"] = (
        (result["stock_splits"] != 0)
        & (result["stock_splits"] != 1)
    )

    # 任一公司行動事件日
    result["is_corporate_action_day"] = (
        result["is_ex_dividend"]
        | result["is_ex_right_or_split"]
    )

    # 事件類型文字
    result["corporate_action_type"] = ""

    result.loc[
        result["is_ex_dividend"],
        "corporate_action_type"
    ] = "ex_dividend"

    result.loc[
        result["is_ex_right_or_split"],
        "corporate_action_type"
    ] = np.where(
        result.loc[
            result["is_ex_right_or_split"],
            "corporate_action_type"
        ] == "",
        "ex_right_or_split",
        (
            result.loc[
                result["is_ex_right_or_split"],
                "corporate_action_type"
            ]
            + "|ex_right_or_split"
        )
    )

    result = (
        result
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # 先把真正事件日設定為要排除。
    result["is_corporate_action_window"] = (
        result["is_corporate_action_day"].astype(bool)
    )

    # 若設定排除事件前後交易日，建立排除窗口。
    if EXCLUDE_EVENT_WINDOW and EVENT_WINDOW_TRADING_DAYS > 0:
        for shift_days in range(1, EVENT_WINDOW_TRADING_DAYS + 1):
            # 事件後 shift_days 個交易日
            event_before = (
                result.groupby("ticker")["is_corporate_action_day"]
                .shift(shift_days)
                .fillna(False)
                .astype(bool)
            )

            # 事件前 shift_days 個交易日
            event_after = (
                result.groupby("ticker")["is_corporate_action_day"]
                .shift(-shift_days)
                .fillna(False)
                .astype(bool)
            )

            result["is_corporate_action_window"] = (
                result["is_corporate_action_window"]
                | event_before
                | event_after
            )

    return result


# ============================================================
# 5. 建立 10 個 K 線特徵
# ============================================================

def add_kline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特徵 1：上影線
    (High - max(Open, Close)) / Close * 100

    特徵 2：下影線
    (min(Open, Close) - Low) / Close * 100

    特徵 3：實體線
    (Close - Open) / Close * 100

    特徵 4、5、6：前一日上影線、下影線、實體線

    特徵 7：開盤型態
    (Open_t - Close_t-1) / Close_t * 100

    特徵 8：收盤型態
    (Close_t - Close_t-1) / Close_t * 100

    特徵 9：成交量與五日均量差
    (Volume_t - MA5(Volume)_t) / Volume_t

    特徵 10：前五日趨勢
    (Close_t-2 - Close_t-7) / Close_t-7 * 100
    """

    result = df.copy()

    result = (
        result
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    group = result.groupby("ticker", group_keys=False)

    # 特徵 1：上影線
    result["upper_shadow_pct"] = (
        (
            result["high"]
            - result[["open", "close"]].max(axis=1)
        )
        / result["close"]
        * 100
    )

    # 特徵 2：下影線
    result["lower_shadow_pct"] = (
        (
            result[["open", "close"]].min(axis=1)
            - result["low"]
        )
        / result["close"]
        * 100
    )

    # 特徵 3：實體線
    result["body_pct"] = (
        (result["close"] - result["open"])
        / result["close"]
        * 100
    )

    # 特徵 4、5、6：前一日 K 線特徵
    result["prev_upper_shadow_pct"] = (
        group["upper_shadow_pct"].shift(1)
    )

    result["prev_lower_shadow_pct"] = (
        group["lower_shadow_pct"].shift(1)
    )

    result["prev_body_pct"] = (
        group["body_pct"].shift(1)
    )

    # 前一日收盤價
    result["prev_close"] = group["close"].shift(1)

    # 特徵 7：開盤型態
    result["open_pattern_pct"] = (
        (result["open"] - result["prev_close"])
        / result["close"]
        * 100
    )

    # 特徵 8：收盤型態
    result["close_pattern_pct"] = (
        (result["close"] - result["prev_close"])
        / result["close"]
        * 100
    )

    # 特徵 9：成交量相對 5 日均量
    result["volume_ma5"] = group["volume"].transform(
        lambda series: series.rolling(
            window=5,
            min_periods=5
        ).mean()
    )

    result["volume_vs_ma5_ratio"] = (
        (result["volume"] - result["volume_ma5"])
        / result["volume"]
    )

    # 特徵 10：前五日趨勢
    close_2_days_ago = group["close"].shift(2)
    close_7_days_ago = group["close"].shift(7)

    result["trend_5d_pct"] = (
        (close_2_days_ago - close_7_days_ago)
        / close_7_days_ago
        * 100
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return result


# ============================================================
# 6. 合併股票池資訊
# ============================================================

def merge_universe_info(
    df: pd.DataFrame,
    universe: pd.DataFrame
) -> pd.DataFrame:
    """
    將市值排名、股票代碼與公司名稱加回每日資料。
    """

    result = df.merge(
        universe[
            [
                "ticker",
                "rank",
                "stock_id",
                "name",
                "market_cap_date"
            ]
        ],
        on="ticker",
        how="left"
    )

    first_columns = [
        "date",
        "ticker",
        "stock_id",
        "name",
        "rank",
        "market_cap_date"
    ]

    remaining_columns = [
        column
        for column in result.columns
        if column not in first_columns
    ]

    return result[first_columns + remaining_columns]


# ============================================================
# 7. 匯出 CSV
# ============================================================

def export_datasets(
    df: pd.DataFrame,
    universe: pd.DataFrame,
    failed_tickers: list
) -> None:
    """
    輸出檔案：

    universe_used.csv
    all_daily_ohlcv_and_features.csv
    corporate_action_days.csv
    excluded_event_window_days.csv
    train_2018_2025.csv
    test_2026.csv
    feature_columns.csv
    data_summary_by_ticker.csv
    failed_tickers.csv
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = [
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

    # 儲存此次使用的股票池
    universe.to_csv(
        OUTPUT_DIR / "universe_used.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 所有資料：含事件日，不刪除，供檢查使用
    all_output = df.copy()
    all_output["date"] = all_output["date"].dt.strftime("%Y-%m-%d")

    all_output.to_csv(
        OUTPUT_DIR / "all_daily_ohlcv_and_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 真正的除息、除權／分割事件日
    corporate_actions = df[
        df["is_corporate_action_day"]
    ].copy()

    corporate_action_columns = [
        "date",
        "ticker",
        "stock_id",
        "name",
        "rank",
        "dividends",
        "stock_splits",
        "is_ex_dividend",
        "is_ex_right_or_split",
        "corporate_action_type"
    ]

    corporate_actions = corporate_actions[
        [
            column
            for column in corporate_action_columns
            if column in corporate_actions.columns
        ]
    ]

    corporate_actions["date"] = corporate_actions["date"].dt.strftime(
        "%Y-%m-%d"
    )

    corporate_actions.to_csv(
        OUTPUT_DIR / "corporate_action_days.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 事件日與其前後交易日的排除清單
    excluded_days = df[
        df["is_corporate_action_window"]
    ].copy()

    excluded_columns = [
        "date",
        "ticker",
        "stock_id",
        "name",
        "rank",
        "is_corporate_action_day",
        "is_corporate_action_window",
        "corporate_action_type",
        "dividends",
        "stock_splits"
    ]

    excluded_days = excluded_days[
        [
            column
            for column in excluded_columns
            if column in excluded_days.columns
        ]
    ]

    excluded_days["date"] = excluded_days["date"].dt.strftime("%Y-%m-%d")

    excluded_days.to_csv(
        OUTPUT_DIR / "excluded_event_window_days.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 模型資料：
    # 1. 10 個特徵都完整
    # 2. 不是除權息／除權或分割的事件窗口
    model_data = df.dropna(subset=feature_columns).copy()

    model_data = model_data[
        ~model_data["is_corporate_action_window"]
    ].copy()

    # 2018～2025 訓練資料
    train_data = model_data[
        (model_data["date"] >= TRAIN_START)
        & (model_data["date"] <= TRAIN_END)
    ].copy()

    # 2026 測試資料
    test_data = model_data[
        (model_data["date"] >= TEST_START)
        & (model_data["date"] <= TEST_END)
    ].copy()

    train_output = train_data.copy()
    test_output = test_data.copy()

    train_output["date"] = train_output["date"].dt.strftime("%Y-%m-%d")
    test_output["date"] = test_output["date"].dt.strftime("%Y-%m-%d")

    train_output.to_csv(
        OUTPUT_DIR / "train_2018_2025.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_output.to_csv(
        OUTPUT_DIR / "test_2026.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 特徵公式說明
    feature_definitions = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "definition": [
                "(High - max(Open, Close)) / Close * 100",
                "(min(Open, Close) - Low) / Close * 100",
                "(Close - Open) / Close * 100",
                "前一交易日 upper_shadow_pct",
                "前一交易日 lower_shadow_pct",
                "前一交易日 body_pct",
                "(Open_t - Close_t-1) / Close_t * 100",
                "(Close_t - Close_t-1) / Close_t * 100",
                "(Volume_t - MA5(Volume)_t) / Volume_t",
                "(Close_t-2 - Close_t-7) / Close_t-7 * 100"
            ]
        }
    )

    feature_definitions.to_csv(
        OUTPUT_DIR / "feature_columns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 每檔股票的資料與事件統計
    summary = (
        df
        .groupby(
            ["ticker", "stock_id", "name", "rank"],
            dropna=False
        )
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            raw_rows=("date", "count"),
            corporate_action_days=(
                "is_corporate_action_day",
                "sum"
            ),
            excluded_window_days=(
                "is_corporate_action_window",
                "sum"
            )
        )
        .reset_index()
        .sort_values("rank")
    )

    summary["first_date"] = summary["first_date"].dt.strftime("%Y-%m-%d")
    summary["last_date"] = summary["last_date"].dt.strftime("%Y-%m-%d")

    summary.to_csv(
        OUTPUT_DIR / "data_summary_by_ticker.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 下載失敗股票
    pd.DataFrame(
        {"failed_ticker": failed_tickers}
    ).to_csv(
        OUTPUT_DIR / "failed_tickers.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("資料輸出完成")
    print("=" * 60)
    print(f"完整 OHLCV + 特徵資料列數：{len(df):,}")
    print(f"除權／除息／除權或分割事件日數：{len(corporate_actions):,}")
    print(f"排除事件窗口後資料列數：{len(model_data):,}")
    print(f"訓練集 2018～2025 列數：{len(train_data):,}")
    print(f"測試集 2026 列數：{len(test_data):,}")
    print(f"成功取得股票數：{df['ticker'].nunique()}")
    print(f"失敗或無資料股票數：{len(failed_tickers)}")
    print(f"輸出資料夾：{OUTPUT_DIR.resolve()}")


# ============================================================
# 8. 主程式
# ============================================================

def main() -> None:
    universe = get_rank_51_100_universe()

    print("\n========== 股票池 ==========")
    print("模式：目前市值排名第 51～100 名")
    print(f"股票數：{len(universe)}")
    print(universe.head())

    prices, failed_tickers = download_price_data(
        tickers=universe["ticker"].tolist(),
        start_date=START_DATE,
        end_date=END_DATE,
        batch_size=BATCH_SIZE
    )

    print("\n========== 原始資料 ==========")
    print(f"成功下載股票數：{prices['ticker'].nunique()}")
    print(f"原始日資料列數：{len(prices):,}")

    # 標記除息、除權／分割事件與其前後排除窗口。
    prices = add_corporate_action_flags(prices)

    # 建立你的 10 個 K 線特徵。
    features = add_kline_features(prices)

    # 合併排名、代碼與公司名稱。
    features = merge_universe_info(
        df=features,
        universe=universe
    )

    # 輸出全部 CSV。
    export_datasets(
        df=features,
        universe=universe,
        failed_tickers=failed_tickers
    )


if __name__ == "__main__":
    main()
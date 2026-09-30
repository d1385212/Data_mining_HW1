from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 0. 基本設定
# ============================================================

START_DATE = "2018-01-01"
END_DATE = "2027-01-01"        # yfinance 的 end 為「不包含」，故設 2027-01-01
TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-12-31"

OUTPUT_DIR = Path("output_tw_stock_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# "rank_51_100": 使用市值排名第 51~100 名的 50 檔
# "all_listed": 使用 twse_listed_tickers.csv 中的全部上市股票
UNIVERSE_MODE = "rank_51_100"

# True：若你尚未準備完整 50 檔名單，可先用示範名單驗證流程
# False：正式研究時應改成 False，並提供自己核對好的 CSV
USE_DEMO_TICKERS = True

# Yahoo Finance 台灣上市股票通常使用 .TW
MARKET_SUFFIX = ".TW"

# 每批下載幾檔；若網路或 Yahoo 限流不穩，可調降至 10~20
BATCH_SIZE = 20

# True：使用 auto_adjust=False 的 OHLC 建構 K 線
AUTO_ADJUST = False


# ============================================================
# 1. 股票池：市值第 51~100 名或全部上市股票
# ============================================================

def normalize_tw_ticker(stock_id: str, suffix: str = ".TW") -> str:
    """
    將股票代碼統一轉為 Yahoo Finance 台灣上市格式，例如：
    2330 -> 2330.TW
    2330.TW -> 2330.TW
    """
    stock_id = str(stock_id).strip().upper()

    if stock_id.endswith(".TW") or stock_id.endswith(".TWO"):
        return stock_id

    return f"{stock_id}{suffix}"


def load_tickers_from_csv(csv_path: str, suffix: str = ".TW") -> pd.DataFrame:
    """
    CSV 至少需要有以下任一欄：
    stock_id / code / ticker

    建議欄位：
    rank, stock_id, name, market_cap_date, market_cap
    """
    df = pd.read_csv(csv_path, dtype=str)

    lower_to_original = {c.lower(): c for c in df.columns}

    if "stock_id" in lower_to_original:
        code_col = lower_to_original["stock_id"]
    elif "code" in lower_to_original:
        code_col = lower_to_original["code"]
    elif "ticker" in lower_to_original:
        code_col = lower_to_original["ticker"]
    else:
        raise ValueError(
            f"{csv_path} 找不到股票代碼欄位。"
            "請使用 stock_id、code 或 ticker 作為欄位名稱。"
        )

    result = df.copy()
    result["stock_id"] = result[code_col].astype(str).str.strip()
    result["ticker"] = result["stock_id"].apply(
        lambda x: normalize_tw_ticker(x, suffix=suffix)
    )

    result = result.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return result


def get_demo_tickers_51_100() -> pd.DataFrame:
    """
    上市市值第 51 ~ 100 名股票池。

    資料來源：
    臺灣期貨交易所公布之「臺灣證券交易所發行量加權股價指數
    成分股暨市值比重」排行。

    注意：
    - 此名單是依查詢當下頁面所列排行建立。
    - 市值排名會隨股價、股本變動。
    - 若做嚴謹歷史回測，應固定一個基準日，
      例如 2025-12-31 的第 51~100 名，避免前視偏誤。
    """

    rank_51_100 = [
        # rank, stock_id, name, market_weight_pct
        (51, "1326", "台化", 0.2662),
        (52, "3481", "群創", 0.2649),
        (53, "2379", "瑞昱", 0.2472),
        (54, "4904", "遠傳", 0.2361),
        (55, "6770", "力積電", 0.2264),
        (56, "3661", "世芯-KY", 0.2261),
        (57, "2449", "京元電子", 0.2194),
        (58, "3034", "聯詠", 0.2185),
        (59, "2615", "萬海", 0.2128),
        (60, "2313", "華通", 0.2039),
        (61, "2801", "彰銀", 0.2017),
        (62, "2002", "中鋼", 0.1983),
        (63, "2207", "和泰車", 0.1901),
        (64, "1590", "亞德客-KY", 0.1895),
        (65, "3044", "健鼎", 0.1736),
        (66, "1519", "華城", 0.1685),
        (67, "2337", "旺宏", 0.1660),
        (68, "3036", "文曄", 0.1654),
        (69, "4938", "和碩", 0.1625),
        (70, "2376", "技嘉", 0.1595),
        (71, "2356", "英業達", 0.1591),
        (72, "2618", "長榮航", 0.1508),
        (73, "6515", "穎崴", 0.1506),
        (74, "2912", "統一超", 0.1497),
        (75, "5876", "上海商銀", 0.1401),
        (76, "6239", "力成", 0.1399),
        (77, "2609", "陽明", 0.1382),
        (78, "5871", "中租-KY", 0.1358),
        (79, "2404", "漢唐", 0.1344),
        (80, "2409", "友達", 0.1338),
        (81, "6213", "聯茂", 0.1325),
        (82, "3533", "嘉澤", 0.1265),
        (83, "1101", "台泥", 0.1221),
        (84, "2324", "仁寶", 0.1183),
        (85, "6139", "亞翔", 0.1183),
        (86, "1802", "台玻", 0.1176),
        (87, "2834", "臺企銀", 0.1172),
        (88, "1605", "華新", 0.1155),
        (89, "1504", "東元", 0.1146),
        (90, "3702", "大聯大", 0.1115),
        (91, "7750", "新代", 0.1085),
        (92, "6415", "矽力*-KY", 0.1068),
        (93, "6805", "富世達", 0.1051),
        (94, "1402", "遠東新", 0.1009),
        (95, "6531", "愛普*", 0.1004),
        (96, "6789", "采鈺", 0.0990),
        (97, "6919", "康霈*", 0.0982),
        (98, "3532", "台勝科", 0.0978),
        (99, "2347", "聯強", 0.0965),
        (100, "2492", "華新科", 0.0952),
    ]

    df = pd.DataFrame(
        rank_51_100,
        columns=["rank", "stock_id", "name", "market_weight_pct"]
    )

    # 作為此次名單的資料版本紀錄；非歷史實際市值日期
    df["market_cap_date"] = "current_page_snapshot"
    df["ticker"] = df["stock_id"].apply(
        lambda x: normalize_tw_ticker(x, suffix=MARKET_SUFFIX)
    )

    return df


def get_universe() -> pd.DataFrame:
    """
    依 UNIVERSE_MODE 取得股票池。
    """
    if UNIVERSE_MODE == "rank_51_100":
        if USE_DEMO_TICKERS:
            print("警告：目前使用示範 50 檔名單，非保證官方市值第 51~100 名。")
            print("正式回測前，請建立 twse_rank_51_100.csv 並將 USE_DEMO_TICKERS 改為 False。")
            universe = get_demo_tickers_51_100()
        else:
            universe = load_tickers_from_csv(
                "twse_rank_51_100.csv",
                suffix=MARKET_SUFFIX
            )

            if len(universe) != 50:
                raise ValueError(
                    f"twse_rank_51_100.csv 目前共有 {len(universe)} 檔，"
                    "預期應為市值第 51~100 名的 50 檔。"
                )

    elif UNIVERSE_MODE == "all_listed":
        universe = load_tickers_from_csv(
            "twse_listed_tickers.csv",
            suffix=MARKET_SUFFIX
        )

    else:
        raise ValueError(
            "UNIVERSE_MODE 只能是 'rank_51_100' 或 'all_listed'。"
        )

    universe = universe.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    universe.to_csv(
        OUTPUT_DIR / "universe_used.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return universe


# ============================================================
# 2. 下載 Yahoo Finance 日線 OHLCV
# ============================================================

def download_price_data(tickers, start_date, end_date, batch_size=20):
    """
    以 batch 下載多檔日資料，並標準化成長表格式：

    date, ticker, open, high, low, close, adj_close, volume
    """
    all_frames = []
    failed_tickers = []

    tickers = list(dict.fromkeys(tickers))

    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start: batch_start + batch_size]

        print(
            f"下載第 {batch_start // batch_size + 1} 批："
            f"{len(batch)} 檔，{batch[0]} ~ {batch[-1]}"
        )

        try:
            raw = yf.download(
                tickers=batch,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=AUTO_ADJUST,
                actions=False,
                group_by="column",
                threads=True,
                progress=False,
                timeout=30
            )
        except Exception as e:
            print(f"批次下載失敗：{e}")
            failed_tickers.extend(batch)
            continue

        if raw is None or raw.empty:
            print("此批沒有下載到資料。")
            failed_tickers.extend(batch)
            continue

        # 單檔與多檔下載時，yfinance 的欄位結構不同，分開處理
        if len(batch) == 1:
            ticker = batch[0]
            temp = raw.copy().reset_index()
            temp.columns = [str(c).lower().replace(" ", "_") for c in temp.columns]

            rename_map = {
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "adj_close": "adj_close",
                "volume": "volume",
            }
            temp = temp.rename(columns=rename_map)
            temp["ticker"] = ticker

            keep_cols = [
                c for c in
                ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
                if c in temp.columns
            ]

            if "close" not in temp.columns:
                failed_tickers.append(ticker)
            else:
                all_frames.append(temp[keep_cols])

        else:
            # 多檔時預期為 MultiIndex 欄位：Price × Ticker
            if not isinstance(raw.columns, pd.MultiIndex):
                print("欄位結構異常，略過此批。")
                failed_tickers.extend(batch)
                continue

            available_tickers = raw.columns.get_level_values(-1).unique()

            for ticker in batch:
                if ticker not in available_tickers:
                    failed_tickers.append(ticker)
                    continue

                try:
                    temp = raw.xs(ticker, axis=1, level=-1).copy()
                except KeyError:
                    failed_tickers.append(ticker)
                    continue

                if temp.empty or "Close" not in temp.columns:
                    failed_tickers.append(ticker)
                    continue

                temp = temp.dropna(how="all")

                if temp.empty or temp["Close"].dropna().empty:
                    failed_tickers.append(ticker)
                    continue

                temp = temp.reset_index()
                temp.columns = [str(c).lower().replace(" ", "_") for c in temp.columns]
                temp["ticker"] = ticker

                keep_cols = [
                    c for c in
                    ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
                    if c in temp.columns
                ]

                all_frames.append(temp[keep_cols])

        sleep(1)

    if not all_frames:
        raise RuntimeError(
            "完全沒有取得資料。請檢查網路、yfinance 版本、Ticker 格式或 Yahoo Finance 狀態。"
        )

    prices = pd.concat(all_frames, ignore_index=True)

    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str)

    numeric_cols = [
        c for c in ["open", "high", "low", "close", "adj_close", "volume"]
        if c in prices.columns
    ]
    prices[numeric_cols] = prices[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    prices = (
        prices
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"])
        .reset_index(drop=True)
    )

    failed_tickers = sorted(set(failed_tickers))
    return prices, failed_tickers


# ============================================================
# 3. 建立你的 10 項特徵
# ============================================================

def add_kline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特徵定義：

    1. upper_shadow_pct
       = (H - max(O, C)) / C * 100

    2. lower_shadow_pct
       = (min(O, C) - L) / C * 100

    3. body_pct
       = (C - O) / C * 100

    4. prev_upper_shadow_pct
       = 前一日 upper_shadow_pct

    5. prev_lower_shadow_pct
       = 前一日 lower_shadow_pct

    6. prev_body_pct
       = 前一日 body_pct

    7. open_pattern_pct
       = (O_t - C_(t-1)) / C_t * 100

    8. close_pattern_pct
       = (C_t - C_(t-1)) / C_t * 100

    9. volume_vs_ma5_ratio
       = (V_t - MA5(V)_t) / V_t
       其中 MA5 使用「含當日」的 5 日平均成交量。

    10. trend_5d_pct
       = (C_(t-2) - C_(t-7)) / C_(t-7) * 100

    注意：
    - trend_5d_pct 依照你原始敘述「2日前收盤價 vs 7天前收盤價」直接實作。
    - 這裡的 2 日、7 日是「交易日列數位移」，不是日曆天。
    """
    df = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    g = df.groupby("ticker", group_keys=False)

    # 當日 K 線型態：3 個特徵
    df["upper_shadow_pct"] = (
        (df["high"] - df[["open", "close"]].max(axis=1))
        / df["close"]
        * 100
    )

    df["lower_shadow_pct"] = (
        (df[["open", "close"]].min(axis=1) - df["low"])
        / df["close"]
        * 100
    )

    df["body_pct"] = (
        (df["close"] - df["open"])
        / df["close"]
        * 100
    )

    # 前一日 K 線型態：3 個特徵
    df["prev_upper_shadow_pct"] = g["upper_shadow_pct"].shift(1)
    df["prev_lower_shadow_pct"] = g["lower_shadow_pct"].shift(1)
    df["prev_body_pct"] = g["body_pct"].shift(1)

    # 前一日收盤價
    df["prev_close"] = g["close"].shift(1)

    # 開盤型態、收盤型態：2 個特徵
    df["open_pattern_pct"] = (
        (df["open"] - df["prev_close"])
        / df["close"]
        * 100
    )

    df["close_pattern_pct"] = (
        (df["close"] - df["prev_close"])
        / df["close"]
        * 100
    )

    # 成交量與 5 日均量差異比率：1 個特徵
    # rolling(5) 預設包含當日，符合你的寫法：(V - 5MV) / V
    df["volume_ma5"] = (
        g["volume"]
        .transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    )

    df["volume_vs_ma5_ratio"] = (
        (df["volume"] - df["volume_ma5"])
        / df["volume"]
    )

    # 前五日趨勢：1 個特徵
    close_2d_ago = g["close"].shift(2)
    close_7d_ago = g["close"].shift(7)

    df["trend_5d_pct"] = (
        (close_2d_ago - close_7d_ago)
        / close_7d_ago
        * 100
    )

    # 防呆：理論上影線不應為負；遇到不完整資料時改為 NaN
    df.loc[df["upper_shadow_pct"] < -1e-10, "upper_shadow_pct"] = np.nan
    df.loc[df["lower_shadow_pct"] < -1e-10, "lower_shadow_pct"] = np.nan

    # 浮點數精度處理
    feature_cols = [
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
    ]

    df[feature_cols] = df[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# 4. 加入股票資訊、切分訓練與測試資料、輸出 CSV
# ============================================================

def merge_universe_info(features: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """
    將 stock_id、名稱、市值排名等資訊合併回每日資料。
    """
    keep_cols = [
        c for c in
        ["ticker", "stock_id", "name", "rank", "market_cap_date", "market_cap"]
        if c in universe.columns
    ]

    universe_info = universe[keep_cols].drop_duplicates(subset=["ticker"])

    result = features.merge(
        universe_info,
        on="ticker",
        how="left"
    )

    front_cols = [
        c for c in
        ["date", "ticker", "stock_id", "name", "rank", "market_cap_date", "market_cap"]
        if c in result.columns
    ]

    other_cols = [c for c in result.columns if c not in front_cols]
    result = result[front_cols + other_cols]

    return result


def export_datasets(df: pd.DataFrame, failed_tickers: list):
    """
    匯出：
    1. all_daily_ohlcv_and_features.csv
    2. train_2018_2025.csv
    3. test_2026.csv
    4. feature_columns.csv
    5. failed_tickers.csv
    6. data_summary_by_ticker.csv
    """
    feature_cols = [
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
    ]

    # 這份是保留原始 OHLCV + 衍生特徵的完整資料
    df.to_csv(
        OUTPUT_DIR / "all_daily_ohlcv_and_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 對模型可用的資料：10 個特徵皆完整才保留
    model_ready = df.dropna(subset=feature_cols).copy()

    train_df = model_ready[
        (model_ready["date"] >= START_DATE) &
        (model_ready["date"] <= TRAIN_END)
    ].copy()

    test_df = model_ready[
        (model_ready["date"] >= TEST_START) &
        (model_ready["date"] <= TEST_END)
    ].copy()

    train_df.to_csv(
        OUTPUT_DIR / "train_2018_2025.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_df.to_csv(
        OUTPUT_DIR / "test_2026.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({
        "feature_name": feature_cols,
        "definition": [
            "(High - max(Open, Close)) / Close * 100",
            "(min(Open, Close) - Low) / Close * 100",
            "(Close - Open) / Close * 100",
            "前一交易日 upper_shadow_pct",
            "前一交易日 lower_shadow_pct",
            "前一交易日 body_pct",
            "(Open_t - Close_t-1) / Close_t * 100",
            "(Close_t - Close_t-1) / Close_t * 100",
            "(Volume_t - Volume_5日均量_t) / Volume_t",
            "(Close_t-2 - Close_t-7) / Close_t-7 * 100",
        ]
    }).to_csv(
        OUTPUT_DIR / "feature_columns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({
        "failed_ticker": failed_tickers
    }).to_csv(
        OUTPUT_DIR / "failed_tickers.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = (
        df.groupby(["ticker", "stock_id"], dropna=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            raw_rows=("date", "count"),
            model_ready_rows=("trend_5d_pct", "count"),
            first_close=("close", "first"),
            last_close=("close", "last"),
        )
        .reset_index()
        .sort_values("ticker")
    )

    summary.to_csv(
        OUTPUT_DIR / "data_summary_by_ticker.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 資料輸出完成 ==========")
    print(f"完整 OHLCV + 特徵資料列數：{len(df):,}")
    print(f"訓練集 2018~2025 列數：{len(train_df):,}")
    print(f"測試集 2026 列數：{len(test_df):,}")
    print(f"成功取得股票數：{df['ticker'].nunique()}")
    print(f"失敗或無資料股票數：{len(failed_tickers)}")
    print(f"輸出資料夾：{OUTPUT_DIR.resolve()}")


# ============================================================
# 5. 主程式
# ============================================================

def main():
    universe = get_universe()

    print("\n========== 股票池 ==========")
    print(f"模式：{UNIVERSE_MODE}")
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

    features = add_kline_features(prices)
    features = merge_universe_info(features, universe)

    export_datasets(features, failed_tickers)


if __name__ == "__main__":
    main()
from pathlib import Path
import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle


# ============================================================
# 0. 基本設定
# ============================================================

RESULT_DIR = Path("output_kmeans_weighted_backtest")

BULLISH_TOP_FILE = RESULT_DIR / "top10_bullish_patterns.csv"
BEARISH_TOP_FILE = RESULT_DIR / "top10_bearish_patterns.csv"

BULLISH_CLUSTERED_FILE = (
    RESULT_DIR / "train_bullish_candidates_clustered.csv"
)

BEARISH_CLUSTERED_FILE = (
    RESULT_DIR / "train_bearish_candidates_clustered.csv"
)

CANDIDATE_DIR = Path("output_big_move_candidates")

TRAIN_FILE = (
    CANDIDATE_DIR / "train_with_future_return_3d.csv"
)

DISPLAY_WINDOW_DAYS = 10
PATTERNS_PER_PAGE = 5

WINDOW_WIDTH = 1680
WINDOW_HEIGHT = 1020

UP_COLOR = "#d62728"
DOWN_COLOR = "#2ca02c"

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\msjhl.ttc",
    r"C:\Windows\Fonts\mingliu.ttc",
    r"C:\Windows\Fonts\kaiu.ttf"
]


# ============================================================
# 1. Matplotlib 中文字型
# ============================================================

def configure_chinese_font() -> str:
    """
    設定 Matplotlib 中文字型，避免中文名稱顯示為方塊。
    """

    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            font_prop = font_manager.FontProperties(
                fname=font_path
            )

            font_name = font_prop.get_name()

            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False

            return font_name

    plt.rcParams["font.family"] = [
        "Microsoft JhengHei",
        "Microsoft JhengHei UI",
        "Noto Sans CJK TC",
        "SimHei"
    ]

    plt.rcParams["axes.unicode_minus"] = False

    return "fallback"


CHINESE_FONT_NAME = configure_chinese_font()


# ============================================================
# 2. 讀取與整理資料
# ============================================================

def read_csv_or_raise(
    file_path: Path,
    description: str
) -> pd.DataFrame:
    """
    讀 CSV；若檔案不存在則回報。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 {description}：\n{file_path}\n\n"
            "請確認已執行 kmeans_weighted_pattern_backtest.py。"
        )

    return pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )


def prepare_top_patterns(
    df: pd.DataFrame,
    direction: str
) -> pd.DataFrame:
    """
    按 overall_rank 排序，最多保留前十名。
    """

    result = df.copy()

    if result.empty:
        return result

    required_columns = [
        "pattern_id",
        "cluster_id"
    ]

    for column in required_columns:
        if column not in result.columns:
            raise ValueError(
                f"{direction} Top pattern 缺少欄位：{column}"
            )

    result["cluster_id"] = pd.to_numeric(
        result["cluster_id"],
        errors="coerce"
    )

    result = result.dropna(
        subset=["cluster_id"]
    ).copy()

    result["cluster_id"] = result["cluster_id"].astype(int)

    if "overall_rank" in result.columns:
        result["overall_rank"] = pd.to_numeric(
            result["overall_rank"],
            errors="coerce"
        )

        result = result.sort_values(
            "overall_rank",
            ascending=True
        )

    elif "pattern_score" in result.columns:
        result = result.sort_values(
            "pattern_score",
            ascending=False
        )

        result["overall_rank"] = np.arange(
            1,
            len(result) + 1
        )

    else:
        result["overall_rank"] = np.arange(
            1,
            len(result) + 1
        )

    return result.head(10).reset_index(drop=True)


def prepare_clustered_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    整理訓練期候選樣本分群資料。
    """

    result = df.copy()

    required_columns = [
        "date",
        "ticker",
        "cluster_id"
    ]

    for column in required_columns:
        if column not in result.columns:
            raise ValueError(
                f"分群資料缺少欄位：{column}"
            )

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce"
    )

    result["ticker"] = (
        result["ticker"]
        .astype(str)
        .str.strip()
    )

    result["cluster_id"] = pd.to_numeric(
        result["cluster_id"],
        errors="coerce"
    )

    if "distance_to_centroid" in result.columns:
        result["distance_to_centroid"] = pd.to_numeric(
            result["distance_to_centroid"],
            errors="coerce"
        )

    result = result.dropna(
        subset=required_columns
    ).copy()

    result["cluster_id"] = result["cluster_id"].astype(int)

    return result


def prepare_train_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    整理用來繪製 K 線的訓練期 OHLCV 資料。
    """

    result = df.copy()

    required_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in required_columns:
        if column not in result.columns:
            raise ValueError(
                f"訓練資料缺少欄位：{column}"
            )

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce"
    )

    result["ticker"] = (
        result["ticker"]
        .astype(str)
        .str.strip()
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    result = result.dropna(
        subset=required_columns
    ).copy()

    return (
        result
        .sort_values(["ticker", "date"])
        .drop_duplicates(
            subset=["ticker", "date"]
        )
        .reset_index(drop=True)
    )


def load_all_data() -> tuple:
    """
    載入 GUI 所需資料。
    """

    bullish_top = prepare_top_patterns(
        read_csv_or_raise(
            BULLISH_TOP_FILE,
            "看漲 Top 10 結果"
        ),
        direction="bullish"
    )

    bearish_top = prepare_top_patterns(
        read_csv_or_raise(
            BEARISH_TOP_FILE,
            "看跌 Top 10 結果"
        ),
        direction="bearish"
    )

    bullish_clustered = prepare_clustered_data(
        read_csv_or_raise(
            BULLISH_CLUSTERED_FILE,
            "看漲訓練期分群資料"
        )
    )

    bearish_clustered = prepare_clustered_data(
        read_csv_or_raise(
            BEARISH_CLUSTERED_FILE,
            "看跌訓練期分群資料"
        )
    )

    train_df = prepare_train_data(
        read_csv_or_raise(
            TRAIN_FILE,
            "訓練期 OHLCV 資料"
        )
    )

    return (
        bullish_top,
        bearish_top,
        bullish_clustered,
        bearish_clustered,
        train_df
    )


# ============================================================
# 3. 找群組代表案例與 OHLCV 視窗
# ============================================================

def get_representative_case(
    pattern_row: pd.Series,
    clustered_df: pd.DataFrame
) -> pd.Series:
    """
    從該 cluster 中找到距離 KMeans 中心最近的訓練案例。
    """

    cluster_id = int(pattern_row["cluster_id"])

    cases = clustered_df[
        clustered_df["cluster_id"] == cluster_id
    ].copy()

    if cases.empty:
        raise ValueError(
            f"找不到 cluster_id={cluster_id} 的代表案例。"
        )

    if "distance_to_centroid" in cases.columns:
        cases = cases.sort_values(
            "distance_to_centroid",
            ascending=True,
            na_position="last"
        )
    else:
        cases = cases.sort_values("date")

    return cases.iloc[0]


def get_ohlcv_window(
    train_df: pd.DataFrame,
    ticker: str,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    取得訊號日以前、包含訊號日在內的最近 10 個交易日。
    """

    stock_df = train_df[
        train_df["ticker"] == ticker
    ].copy()

    stock_df = stock_df[
        stock_df["date"] <= end_date
    ].sort_values("date")

    window = stock_df.tail(
        DISPLAY_WINDOW_DAYS
    ).copy()

    if len(window) < 2:
        raise ValueError(
            f"{ticker} 在 {end_date.strftime('%Y-%m-%d')} 前資料不足。"
        )

    return window


# ============================================================
# 4. 工具函數
# ============================================================

def format_number(
    value,
    digits: int = 2,
    suffix: str = ""
) -> str:
    """
    數字格式化；NaN 顯示 N/A。
    """

    if pd.isna(value):
        return "N/A"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if digits == 0:
        return f"{number:,.0f}{suffix}"

    return f"{number:.{digits}f}{suffix}"


def get_text_value(
    row: pd.Series,
    column: str,
    digits: int = 2,
    suffix: str = ""
) -> str:
    """
    安全取得 CSV 欄位並格式化。
    """

    if column not in row.index:
        return "N/A"

    return format_number(
        row.get(column),
        digits=digits,
        suffix=suffix
    )


# ============================================================
# 5. 左側文字：明確區分「訊號日前」與「訊號日」
# ============================================================

def get_pattern_text(
    pattern_row: pd.Series,
    representative: pd.Series,
    direction: str
) -> str:
    """
    生成左側 Tkinter 文字。

    關鍵設計：
    - 訓練期群組平均特徵，與代表案例當日特徵分開呈現。
    - 明確寫「訊號日前五日」與「訊號日當天」。
    - 避免把群組平均誤認為右圖每一根 K 線都具備的特徵。
    """

    rank = int(
        pattern_row.get("overall_rank", 0)
    )

    direction_text = (
        "看漲型態"
        if direction == "bullish"
        else "看跌／風險候選型態"
    )

    ticker = str(
        representative.get("ticker", "N/A")
    )

    name = str(
        representative.get("name", "")
    )

    date = pd.to_datetime(
        representative.get("date")
    ).strftime("%Y-%m-%d")

    pattern_id = str(
        pattern_row.get("pattern_id", "N/A")
    )

    # --------------------------------------------------------
    # 代表案例的實際特徵：
    # 這些才是右側圖「最後一根訊號日 K 線」的數值。
    # --------------------------------------------------------
    representative_trend_5d = get_text_value(
        representative,
        "trend_5d_pct",
        2,
        "%"
    )

    representative_body = get_text_value(
        representative,
        "body_pct",
        2,
        "%"
    )

    representative_upper = get_text_value(
        representative,
        "upper_shadow_pct",
        2,
        "%"
    )

    representative_lower = get_text_value(
        representative,
        "lower_shadow_pct",
        2,
        "%"
    )

    representative_volume = get_text_value(
        representative,
        "volume_vs_ma5_ratio",
        3
    )

    # --------------------------------------------------------
    # 群組平均特徵：
    # 這是整個 KMeans 群的平均，不等於右圖單一案例。
    # --------------------------------------------------------
    average_trend_5d = get_text_value(
        pattern_row,
        "avg_trend_5d_pct",
        2,
        "%"
    )

    average_body = get_text_value(
        pattern_row,
        "avg_body_pct",
        2,
        "%"
    )

    average_upper = get_text_value(
        pattern_row,
        "avg_upper_shadow_pct",
        2,
        "%"
    )

    average_lower = get_text_value(
        pattern_row,
        "avg_lower_shadow_pct",
        2,
        "%"
    )

    average_volume = get_text_value(
        pattern_row,
        "avg_volume_vs_ma5_ratio",
        3
    )

    test_frequency = get_text_value(
        pattern_row,
        "test_frequency",
        0
    )

    test_hit_rate = get_text_value(
        pattern_row,
        "test_hit_rate_pct",
        2,
        "%"
    )

    lift = get_text_value(
        pattern_row,
        "lift_vs_2026_baseline",
        2
    )

    mean_return = get_text_value(
        pattern_row,
        "test_mean_future_return_3d_pct",
        2,
        "%"
    )

    train_count = get_text_value(
        pattern_row,
        "train_candidate_count",
        0
    )

    auto_description = str(
        pattern_row.get(
            "auto_description",
            "無自動描述"
        )
    )

    return (
        f"第 {rank} 名｜{direction_text}\n"
        f"Pattern：{pattern_id}\n\n"
        f"【右圖代表案例】\n"
        f"股票：{ticker} {name}\n"
        f"訊號日：{date}\n\n"
        f"【此代表案例：訊號日前五日】\n"
        f"五日趨勢：{representative_trend_5d}\n\n"
        f"【此代表案例：訊號日當天】\n"
        f"實體線：{representative_body}\n"
        f"上影線：{representative_upper}\n"
        f"下影線：{representative_lower}\n"
        f"量能相對五日均量差：{representative_volume}\n\n"
        f"【訓練期群組平均，不等於右圖每一日】\n"
        f"訊號日前五日平均趨勢：{average_trend_5d}\n"
        f"訊號日平均實體線：{average_body}\n"
        f"訊號日平均上影線：{average_upper}\n"
        f"訊號日平均下影線：{average_lower}\n"
        f"訊號日平均量能差：{average_volume}\n"
        f"群組摘要：{auto_description}\n\n"
        f"【2026 樣本外回測】\n"
        f"出現次數：{test_frequency}\n"
        f"三日後方向命中率：{test_hit_rate}\n"
        f"Lift：{lift}\n"
        f"平均未來三日報酬：{mean_return}\n\n"
        f"【訓練期群內大漲／大跌樣本數】\n"
        f"{train_count}"
    )


# ============================================================
# 6. 右側 K 線和成交量
# ============================================================

def draw_candlestick_chart(
    ax_price,
    ax_volume,
    ohlcv: pd.DataFrame,
    direction: str
) -> None:
    """
    右側只畫技術圖：
    - K 線
    - 成交量
    不顯示多餘統計文字。
    """

    ax_price.clear()
    ax_volume.clear()

    x_values = np.arange(len(ohlcv))

    for x, (_, row) in zip(
        x_values,
        ohlcv.iterrows()
    ):
        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]

        color = (
            UP_COLOR
            if close_price >= open_price
            else DOWN_COLOR
        )

        # 上下影線
        ax_price.vlines(
            x=x,
            ymin=low_price,
            ymax=high_price,
            color=color,
            linewidth=1.35,
            zorder=1
        )

        # 實體
        lower = min(open_price, close_price)
        body_height = abs(close_price - open_price)

        if body_height == 0:
            body_height = max(
                high_price - low_price,
                0.01
            ) * 0.03

        candle = Rectangle(
            xy=(x - 0.30, lower),
            width=0.60,
            height=body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.90,
            zorder=2
        )

        ax_price.add_patch(candle)

        # 成交量
        ax_volume.bar(
            x=x,
            height=row["volume"],
            width=0.60,
            color=color,
            alpha=0.76
        )

    ax_price.set_ylabel(
        "股價",
        fontsize=9
    )

    ax_price.tick_params(
        axis="y",
        labelsize=8
    )

    ax_price.grid(
        alpha=0.22,
        linestyle="--"
    )

    ax_price.set_xlim(
        -0.7,
        len(ohlcv) - 0.2
    )

    # 最後一天就是訊號日
    signal_x = len(ohlcv) - 1

    ax_price.axvline(
        signal_x,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        alpha=0.90
    )

    ax_price.text(
        signal_x + 0.08,
        ax_price.get_ylim()[1],
        "訊號日",
        color="#1f77b4",
        fontsize=8,
        verticalalignment="top"
    )

    ax_volume.set_ylabel(
        "成交量",
        fontsize=9
    )

    ax_volume.tick_params(
        axis="y",
        labelsize=8
    )

    ax_volume.grid(
        alpha=0.22,
        linestyle="--"
    )

    ax_volume.set_xlim(
        -0.7,
        len(ohlcv) - 0.2
    )

    tick_positions = sorted(
        set(
            [
                0,
                len(ohlcv) // 4,
                len(ohlcv) // 2,
                (len(ohlcv) * 3) // 4,
                len(ohlcv) - 1
            ]
        )
    )

    tick_labels = [
        ohlcv.iloc[position]["date"].strftime(
            "%m/%d"
        )
        for position in tick_positions
    ]

    ax_volume.set_xticks(tick_positions)

    ax_volume.set_xticklabels(
        tick_labels,
        fontsize=8
    )

    if direction == "bullish":
        ax_price.set_facecolor("#fffafa")
    else:
        ax_price.set_facecolor("#fafffa")


# ============================================================
# 7. GUI 中的一列 Pattern
# ============================================================

class PatternRow:
    """
    每個 pattern 一列：

    左側：Tkinter 純文字，明確區分訊號日前五日／訊號日。
    右側：Matplotlib 的 K 線與成交量。
    """

    def __init__(
        self,
        parent,
        pattern_row: pd.Series,
        clustered_df: pd.DataFrame,
        train_df: pd.DataFrame,
        direction: str
    ) -> None:
        self.parent = parent

        row_frame = ttk.Frame(
            parent,
            padding=(6, 6)
        )

        row_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=4,
            pady=3
        )

        row_frame.columnconfigure(0, weight=35)
        row_frame.columnconfigure(1, weight=65)
        row_frame.rowconfigure(0, weight=1)

        # ----------------------------------------------------
        # 左側文字區
        # ----------------------------------------------------
        if direction == "bullish":
            background_color = "#fff4f4"
            border_color = "#d6604d"
        else:
            background_color = "#f3fff3"
            border_color = "#5aae61"

        info_outer = tk.Frame(
            row_frame,
            bg=border_color,
            padx=2,
            pady=2
        )

        info_outer.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 8)
        )

        info_inner = tk.Frame(
            info_outer,
            bg=background_color,
            padx=13,
            pady=10
        )

        info_inner.pack(
            fill=tk.BOTH,
            expand=True
        )

        try:
            representative = get_representative_case(
                pattern_row=pattern_row,
                clustered_df=clustered_df
            )

            info_text = get_pattern_text(
                pattern_row=pattern_row,
                representative=representative,
                direction=direction
            )

        except Exception as error:
            representative = None

            info_text = (
                "無法讀取代表案例\n\n"
                f"{pattern_row.get('pattern_id', '')}\n\n"
                f"{str(error)}"
            )

        info_label = tk.Label(
            info_inner,
            text=info_text,
            justify=tk.LEFT,
            anchor="nw",
            bg=background_color,
            fg="#202020",
            font=(
                "Microsoft JhengHei",
                9
            ),
            wraplength=420
        )

        info_label.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # 右側 K 線技術圖
        # ----------------------------------------------------
        chart_frame = ttk.Frame(
            row_frame
        )

        chart_frame.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.figure, (
            self.ax_price,
            self.ax_volume
        ) = plt.subplots(
            2,
            1,
            figsize=(8.0, 3.4),
            gridspec_kw={
                "height_ratios": [3, 1]
            }
        )

        self.figure.tight_layout(pad=1.0)

        if representative is not None:
            try:
                ticker = str(
                    representative["ticker"]
                )

                representative_date = pd.to_datetime(
                    representative["date"]
                )

                ohlcv_window = get_ohlcv_window(
                    train_df=train_df,
                    ticker=ticker,
                    end_date=representative_date
                )

                draw_candlestick_chart(
                    ax_price=self.ax_price,
                    ax_volume=self.ax_volume,
                    ohlcv=ohlcv_window,
                    direction=direction
                )

            except Exception as error:
                self.ax_price.clear()
                self.ax_volume.clear()

                self.ax_price.text(
                    0.5,
                    0.5,
                    f"技術圖繪製失敗\n{str(error)[:100]}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=self.ax_price.transAxes,
                    fontsize=10
                )

                self.ax_price.axis("off")
                self.ax_volume.axis("off")

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=chart_frame
        )

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )


# ============================================================
# 8. GUI 主類別
# ============================================================

class PatternViewerApp:
    """
    每頁顯示五個 pattern。

    下拉選單可切換：
    - 看漲第 1～5 名
    - 看漲第 6～10 名
    - 看跌／風險第 1～5 名
    - 看跌／風險第 6～10 名
    """

    PAGE_OPTIONS = [
        "看漲排名第 1～5 名",
        "看漲排名第 6～10 名",
        "看跌／風險排名第 1～5 名",
        "看跌／風險排名第 6～10 名"
    ]

    def __init__(
        self,
        root: tk.Tk,
        bullish_top: pd.DataFrame,
        bearish_top: pd.DataFrame,
        bullish_clustered: pd.DataFrame,
        bearish_clustered: pd.DataFrame,
        train_df: pd.DataFrame
    ) -> None:
        self.root = root

        self.bullish_top = bullish_top
        self.bearish_top = bearish_top

        self.bullish_clustered = bullish_clustered
        self.bearish_clustered = bearish_clustered

        self.train_df = train_df
        self.chart_rows = []

        self.root.title(
            "K 線型態 GUI：訊號日前五日與訊號日特徵分開顯示"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.create_widgets()
        self.render_page()

    def create_widgets(self) -> None:
        """
        建立上方控制區和下方可捲動內容區。
        """

        control_frame = ttk.Frame(
            self.root,
            padding=8
        )

        control_frame.pack(
            side=tk.TOP,
            fill=tk.X
        )

        ttk.Label(
            control_frame,
            text="選擇頁面：",
            font=(
                "Microsoft JhengHei",
                11
            )
        ).pack(
            side=tk.LEFT,
            padx=(5, 4)
        )

        self.page_var = tk.StringVar(
            value=self.PAGE_OPTIONS[0]
        )

        page_combo = ttk.Combobox(
            control_frame,
            textvariable=self.page_var,
            values=self.PAGE_OPTIONS,
            state="readonly",
            width=28,
            font=(
                "Microsoft JhengHei",
                10
            )
        )

        page_combo.pack(
            side=tk.LEFT,
            padx=5
        )

        page_combo.bind(
            "<<ComboboxSelected>>",
            self.on_page_changed
        )

        ttk.Button(
            control_frame,
            text="重新顯示",
            command=self.render_page
        ).pack(
            side=tk.LEFT,
            padx=10
        )

        self.status_label = ttk.Label(
            control_frame,
            text="",
            font=(
                "Microsoft JhengHei",
                10
            )
        )

        self.status_label.pack(
            side=tk.LEFT,
            padx=15
        )

        # 可捲動區
        main_container = ttk.Frame(
            self.root
        )

        main_container.pack(
            fill=tk.BOTH,
            expand=True
        )

        self.scroll_canvas = tk.Canvas(
            main_container,
            highlightthickness=0
        )

        scrollbar = ttk.Scrollbar(
            main_container,
            orient=tk.VERTICAL,
            command=self.scroll_canvas.yview
        )

        self.scrollable_frame = ttk.Frame(
            self.scroll_canvas
        )

        self.scrollable_frame.bind(
            "<Configure>",
            lambda event: self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            )
        )

        self.canvas_window = self.scroll_canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw"
        )

        self.scroll_canvas.bind(
            "<Configure>",
            self.on_canvas_configure
        )

        self.scroll_canvas.configure(
            yscrollcommand=scrollbar.set
        )

        self.scroll_canvas.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )

        self.scroll_canvas.bind_all(
            "<MouseWheel>",
            self.on_mousewheel
        )

    def on_canvas_configure(self, event) -> None:
        """
        讓可捲動內容寬度隨 GUI 視窗寬度調整。
        """

        self.scroll_canvas.itemconfig(
            self.canvas_window,
            width=event.width
        )

    def on_mousewheel(self, event) -> None:
        """
        Windows 滑鼠滾輪。
        """

        self.scroll_canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )

    def get_page_settings(self) -> tuple:
        """
        依選單取得目前要顯示的資料。
        """

        selected = self.page_var.get()

        if selected == "看漲排名第 1～5 名":
            return (
                self.bullish_top,
                self.bullish_clustered,
                "bullish",
                0,
                5
            )

        if selected == "看漲排名第 6～10 名":
            return (
                self.bullish_top,
                self.bullish_clustered,
                "bullish",
                5,
                10
            )

        if selected == "看跌／風險排名第 1～5 名":
            return (
                self.bearish_top,
                self.bearish_clustered,
                "bearish",
                0,
                5
            )

        return (
            self.bearish_top,
            self.bearish_clustered,
            "bearish",
            5,
            10
        )

    def on_page_changed(self, event=None) -> None:
        """
        切換頁面時重新顯示。
        """

        self.render_page()

    def clear_old_rows(self) -> None:
        """
        清除舊視圖與釋放 figure。
        """

        for pattern_row in self.chart_rows:
            try:
                plt.close(pattern_row.figure)
            except Exception:
                pass

        self.chart_rows = []

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def render_page(self) -> None:
        """
        建立目前五個 pattern 的 GUI 列。
        """

        (
            top_patterns,
            clustered_df,
            direction,
            start_index,
            end_index
        ) = self.get_page_settings()

        if top_patterns.empty:
            messagebox.showwarning(
                "沒有資料",
                "此方向沒有 Top pattern 可顯示。"
            )
            return

        if start_index >= len(top_patterns):
            messagebox.showwarning(
                "資料不足",
                "此頁沒有足夠的 pattern 可顯示。"
            )
            return

        self.clear_old_rows()

        page_patterns = top_patterns.iloc[
            start_index:end_index
        ].copy()

        direction_text = (
            "看漲"
            if direction == "bullish"
            else "看跌／風險"
        )

        title_label = ttk.Label(
            self.scrollable_frame,
            text=(
                f"{direction_text} K 線型態："
                f"第 {start_index + 1}～"
                f"{min(end_index, len(top_patterns))} 名"
            ),
            font=(
                "Microsoft JhengHei",
                16,
                "bold"
            )
        )

        title_label.pack(
            fill=tk.X,
            padx=10,
            pady=(8, 2)
        )

        for _, pattern_row in page_patterns.iterrows():
            new_row = PatternRow(
                parent=self.scrollable_frame,
                pattern_row=pattern_row,
                clustered_df=clustered_df,
                train_df=self.train_df,
                direction=direction
            )

            self.chart_rows.append(new_row)

        self.status_label.configure(
            text=(
                "左側已清楚區分：代表案例的訊號日前五日、"
                "訊號日當天，以及訓練期群組平均特徵。"
            )
        )

        self.root.after(
            100,
            lambda: self.scroll_canvas.yview_moveto(0)
        )


# ============================================================
# 9. 主程式
# ============================================================

def main() -> None:
    """
    讀取資料並啟動 GUI。
    """

    try:
        (
            bullish_top,
            bearish_top,
            bullish_clustered,
            bearish_clustered,
            train_df
        ) = load_all_data()

    except Exception as error:
        print("\n無法啟動 GUI。")
        print(str(error))
        return

    if bullish_top.empty and bearish_top.empty:
        print(
            "看漲與看跌 Top pattern 都沒有資料，無法顯示。"
        )
        return

    root = tk.Tk()

    PatternViewerApp(
        root=root,
        bullish_top=bullish_top,
        bearish_top=bearish_top,
        bullish_clustered=bullish_clustered,
        bearish_clustered=bearish_clustered,
        train_df=train_df
    )

    root.mainloop()


if __name__ == "__main__":
    main()
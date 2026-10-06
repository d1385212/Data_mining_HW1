from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "output_2026_pattern_test/test_2026_with_future_return.csv"
)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

df["future_return_3d"] = pd.to_numeric(
    df["future_return_3d"],
    errors="coerce"
)

df = df.dropna(
    subset=["future_return_3d"]
).copy()

bullish_rate = (
    df["future_return_3d"] > 0.05
).mean() * 100

bearish_rate = (
    df["future_return_3d"] < -0.05
).mean() * 100

print("2026 全部可用交易日基準率")
print(f"樣本數：{len(df):,}")
print(f"三日後大漲 > 5%：{bullish_rate:.2f}%")
print(f"三日後大跌 < -5%：{bearish_rate:.2f}%")
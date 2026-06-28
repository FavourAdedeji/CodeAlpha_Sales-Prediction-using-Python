"""
01_data_cleaning.py
--------------------
Loads the raw advertising/sales dataset, cleans it, validates it, and
engineers features (spend share per channel, total spend, interaction
terms) that are used by the modeling and impact-analysis scripts.

Run from the project root:
    python src/01_data_cleaning.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw() -> pd.DataFrame:
    """The raw CSV ships with an unnamed leading index column (1..200)."""
    df = pd.read_csv(RAW_DIR / "advertising.csv", index_col=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    # Drop exact duplicate rows, if any
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    # Coerce to numeric and drop rows that fail (defensive — none expected)
    for col in ["TV", "Radio", "Newspaper", "Sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_missing = df.isna().sum().sum()
    df = df.dropna().reset_index(drop=True)

    # Sales and spend cannot logically be negative
    for col in ["TV", "Radio", "Newspaper", "Sales"]:
        df = df[df[col] >= 0]

    print(f"Rows before cleaning: {n_before}")
    print(f"Duplicate rows removed: {n_dupes}")
    print(f"Rows with missing/invalid values removed: {n_missing}")
    print(f"Rows after cleaning: {len(df)}")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used in modeling and channel-impact analysis."""
    df = df.copy()
    df["total_spend"] = df["TV"] + df["Radio"] + df["Newspaper"]

    # Spend share per channel (the closest available proxy for "platform mix",
    # since the raw data has no explicit platform/segment label — see README
    # for the data-availability note on this).
    for col in ["TV", "Radio", "Newspaper"]:
        df[f"{col.lower()}_share"] = np.where(
            df["total_spend"] > 0, df[col] / df["total_spend"], 0.0
        )

    # Simple two-way interaction terms (captures cross-channel synergy,
    # e.g. TV + Radio campaigns reinforcing each other)
    df["tv_radio_interaction"] = df["TV"] * df["Radio"]
    df["tv_newspaper_interaction"] = df["TV"] * df["Newspaper"]
    df["radio_newspaper_interaction"] = df["Radio"] * df["Newspaper"]

    # Sales per unit of total spend — a simple efficiency/ROI proxy
    df["sales_per_spend"] = np.where(
        df["total_spend"] > 0, df["Sales"] / df["total_spend"], np.nan
    )

    return df


def main():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)

    out_path = PROCESSED_DIR / "advertising_clean.csv"
    df.to_csv(out_path, index=False)

    print("\nFinal feature set:", list(df.columns))
    print(f"Saved cleaned + feature-engineered dataset to {out_path}")
    print("\nSummary statistics:")
    print(df.describe().round(2))


if __name__ == "__main__":
    main()

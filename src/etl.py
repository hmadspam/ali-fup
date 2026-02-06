from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---- Config ----

RAW_TO_CANONICAL = {
    "Date": "date",
    "Store ID": "store_id",
    "Product ID": "sku",
    "Category": "category",
    "Region": "region",
    "Inventory Level": "inventory_level",
    "Units Sold": "units_sold",
    "Units Ordered": "units_ordered",
    "Price": "price",
    "Discount": "discount",
    "Weather Condition": "weather_condition",
    "Promotion": "promotion",
    "Competitor Pricing": "competitor_pricing",
    "Seasonality": "seasonality",
    "Epidemic": "epidemic",
    "Demand": "demand",
}

REQUIRED_CANONICAL_COLS = [
    "date",
    "store_id",
    "sku",
    "units_sold",
    "inventory_level",
    "price",
]


@dataclass(frozen=True)
class ETLReport:
    input_rows: int
    output_rows: int
    dropped_bad_dates: int
    dropped_negative_units: int
    dropped_duplicates: int
    nulls_filled_units_sold: int


# ---- Core ETL ----

def load_raw_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    # keep_default_na=True is fine; pandas will parse empty strings as NaN
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Raw CSV loaded but is empty.")

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rename known columns
    missing_raw = [c for c in RAW_TO_CANONICAL.keys() if c not in df.columns]
    if missing_raw:
        raise ValueError(
            "Your CSV is missing expected columns.\n"
            f"Missing: {missing_raw}\n"
            f"Found: {list(df.columns)}"
        )

    df = df.rename(columns=RAW_TO_CANONICAL)

    # strip whitespace from string columns
    for col in ["store_id", "sku", "category", "region", "weather_condition", "seasonality"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def validate_schema(df: pd.DataFrame, required_cols: Iterable[str] = REQUIRED_CANONICAL_COLS) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns after rename: {missing}")

    # sanity: must have at least 30 unique dates for forecasting
    if df["date"].nunique() < 30:
        raise ValueError("Dataset has < 30 unique dates. Forecasting will be meaningless.")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ETLReport]:
    df = df.copy()
    input_rows = len(df)

    # Parse date safely
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    df = df.dropna(subset=["date"])

    # Enforce numeric columns robustly
    numeric_cols = [
        "inventory_level",
        "units_sold",
        "units_ordered",
        "price",
        "discount",
        "promotion",
        "competitor_pricing",
        "epidemic",
        "demand",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill units_sold NaNs with 0 (valid for retail)
    null_units_before = df["units_sold"].isna().sum()
    df["units_sold"] = df["units_sold"].fillna(0)

    # Remove negative units_sold (invalid)
    neg_units = (df["units_sold"] < 0).sum()
    df = df[df["units_sold"] >= 0]

    # store_id and sku must not be null
    df["store_id"] = df["store_id"].astype(str).str.strip()
    df["sku"] = df["sku"].astype(str).str.strip()
    df = df[(df["store_id"] != "") & (df["sku"] != "")]

    # Remove duplicates per (date, store_id, sku)
    before_dupes = len(df)
    df = df.drop_duplicates(subset=["date", "store_id", "sku"], keep="last")
    dropped_duplicates = before_dupes - len(df)

    # Sort for time-series operations
    df = df.sort_values(["store_id", "sku", "date"]).reset_index(drop=True)

    report = ETLReport(
        input_rows=input_rows,
        output_rows=len(df),
        dropped_bad_dates=int(bad_dates),
        dropped_negative_units=int(neg_units),
        dropped_duplicates=int(dropped_duplicates),
        nulls_filled_units_sold=int(null_units_before),
    )

    return df, report


def save_processed_parquet(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def run_etl(raw_path: str | Path, out_path: str | Path) -> ETLReport:
    df = load_raw_csv(raw_path)
    df = standardize_columns(df)
    validate_schema(df)
    df, report = clean_data(df)
    save_processed_parquet(df, out_path)
    return report

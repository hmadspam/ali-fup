from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    target_col: str = "units_sold"
    group_cols: tuple[str, str] = ("store_id", "sku")
    date_col: str = "date"

    lags: tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 14, 28)

    drop_rows_with_nan_lags: bool = True


def _add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df[date_col]

    df["day_of_week"] = d.dt.dayofweek.astype("int16")  # 0=Mon
    df["day_of_month"] = d.dt.day.astype("int16")
    df["week_of_year"] = d.dt.isocalendar().week.astype("int16")
    df["month"] = d.dt.month.astype("int16")
    df["quarter"] = d.dt.quarter.astype("int16")
    df["year"] = d.dt.year.astype("int16")

    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_month_start"] = d.dt.is_month_start.astype("int8")
    df["is_month_end"] = d.dt.is_month_end.astype("int8")

    return df


def _add_lag_features(
    df: pd.DataFrame,
    group_cols: tuple[str, str],
    target_col: str,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    g = df.groupby(list(group_cols), sort=False)[target_col]

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = g.shift(lag).astype("float32")

    return df


def _add_rolling_features(
    df: pd.DataFrame,
    group_cols: tuple[str, str],
    target_col: str,
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    g = df.groupby(list(group_cols), sort=False)[target_col]

    for w in rolling_windows:
        # shift(1) ensures we don't leak today's target into today's features
        roll = g.shift(1).rolling(window=w, min_periods=max(2, w // 3))
        df[f"{target_col}_roll_{w}_mean"] = roll.mean().astype("float32")
        df[f"{target_col}_roll_{w}_std"] = roll.std(ddof=0).astype("float32")
        df[f"{target_col}_roll_{w}_min"] = roll.min().astype("float32")
        df[f"{target_col}_roll_{w}_max"] = roll.max().astype("float32")

    return df


def _add_price_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    # Safe numeric casting
    for col in ["price", "discount", "promotion", "competitor_pricing", "inventory_level", "units_ordered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Promotion already exists (0/1)
    if "promotion" in df.columns:
        df["promotion"] = df["promotion"].fillna(0).astype("int8")

    # Discount: keep numeric and also create discount flag
    if "discount" in df.columns:
        df["discount"] = df["discount"].fillna(0).astype("float32")
        df["has_discount"] = (df["discount"] > 0).astype("int8")

    # Price ratio vs competitor
    if "price" in df.columns and "competitor_pricing" in df.columns:
        eps = 1e-6
        df["price_vs_competitor"] = (df["price"] / (df["competitor_pricing"] + eps)).astype("float32")

    return df


def _add_inventory_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Days of cover using rolling mean demand (7)
    if "inventory_level" not in df.columns:
        return df

    if f"{target_col}_roll_7_mean" in df.columns:
        eps = 1e-6
        df["days_of_cover_7"] = (df["inventory_level"] / (df[f"{target_col}_roll_7_mean"] + eps)).astype("float32")
        df["low_stock_flag"] = (df["days_of_cover_7"] < 3).astype("int8")

    return df


def make_features(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    df = df.copy()

    # Basic validation
    for c in [cfg.date_col, *cfg.group_cols, cfg.target_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col])

    # Sort is mandatory for lags/rolling correctness
    df = df.sort_values([*cfg.group_cols, cfg.date_col]).reset_index(drop=True)

    # Features
    df = _add_date_features(df, cfg.date_col)
    df = _add_price_promo_features(df)
    df = _add_lag_features(df, cfg.group_cols, cfg.target_col, cfg.lags)
    df = _add_rolling_features(df, cfg.group_cols, cfg.target_col, cfg.rolling_windows)
    df = _add_inventory_features(df, cfg.target_col)

    # Categorical cleanup (we'll encode later in training)
    for col in ["category", "region", "weather_condition", "seasonality"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Optional: drop early rows where lag features are NaN
    if cfg.drop_rows_with_nan_lags:
        lag_cols = [f"{cfg.target_col}_lag_{l}" for l in cfg.lags]
        df = df.dropna(subset=lag_cols)

    # Reduce memory
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce").fillna(0).astype("float32")

    return df


def run_feature_pipeline(
    in_path: str | Path = "data/processed/clean_sales.parquet",
    out_path: str | Path = "data/processed/features.parquet",
) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    feat_df = make_features(df)

    feat_df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(feat_df))
    print("Cols:", len(feat_df.columns))
    print("Date range:", feat_df["date"].min(), "->", feat_df["date"].max())


if __name__ == "__main__":
    run_feature_pipeline()

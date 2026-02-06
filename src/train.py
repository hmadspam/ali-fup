from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class TrainConfig:
    features_path: str = "data/processed/features.parquet"
    model_out_path: str = "models/xgb_global.pkl"

    date_col: str = "date"
    target_col: str = "units_sold"

    # Time split (no leakage)
    val_days: int = 30

    random_state: int = 42


def _time_split(df: pd.DataFrame, date_col: str, val_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df[date_col].max()
    cutoff = max_date - pd.Timedelta(days=val_days)

    train_df = df[df[date_col] <= cutoff].copy()
    val_df = df[df[date_col] > cutoff].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Time split failed. Train or validation is empty.")

    return train_df, val_df


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _prepare_xy(df: pd.DataFrame, target_col: str, drop_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].astype("float32")
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    # Convert string or object columns to category codes
    for col in X.columns:
        # Convert pandas StringDtype to category
        if pd.api.types.is_string_dtype(X[col]) or X[col].dtype == "object":
            X[col] = X[col].astype("category")

        if str(X[col].dtype) == "category":
            X[col] = X[col].cat.codes.astype("int32")

    # Final safety: all must be numeric
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        raise ValueError(f"Non-numeric columns still exist: {non_numeric}")

    return X, y


def train_xgb(cfg: TrainConfig = TrainConfig()) -> None:
    df = pd.read_parquet(cfg.features_path)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])

    # ---- CRITICAL: prevent target leakage ----
    # Demand column is suspicious; remove it.
    # Also remove units_ordered (can leak if it's future ordering).
    drop_cols = [
        cfg.target_col,
        cfg.date_col,
        "demand",
        "units_ordered",
    ]

    train_df, val_df = _time_split(df, cfg.date_col, cfg.val_days)

    X_train, y_train = _prepare_xy(train_df, cfg.target_col, drop_cols)
    X_val, y_val = _prepare_xy(val_df, cfg.target_col, drop_cols)

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    val_pred = model.predict(X_val)

    metrics = {
        "rmse": rmse(y_val, val_pred),
        "mape": mape(y_val, val_pred),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "features": X_train.shape[1],
        "val_days": cfg.val_days,
        "cutoff_date": str(val_df[cfg.date_col].min().date()),
    }

    Path(cfg.model_out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X_train.columns),
            "drop_cols": drop_cols,
            "metrics": metrics,
        },
        cfg.model_out_path,
    )

    print("Saved model:", cfg.model_out_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    train_xgb()

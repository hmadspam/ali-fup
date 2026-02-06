import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def generate_forecasts(features_path="data/processed/features.parquet", 
                       model_path="models/xgb_global.pkl", 
                       out_path="data/forecasts/fact_forecast.parquet"):
    """
    Production-ready forecasting script.
    Loads the trained model and applies it to the most recent data points 
    to generate the next day's sales forecast.
    """
    # 1. Load the model and training metadata 
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run training first.")
    
    # Load the dictionary containing the model and the required feature list
    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    feature_cols = artifacts["feature_columns"]
    drop_cols = artifacts["drop_cols"]
    
    # 2. Load feature-engineered data [cite: 103]
    if not Path(features_path).exists():
        raise FileNotFoundError(f"Features not found at {features_path}. Run feature pipeline first.")
    
    df = pd.read_parquet(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. Filter for the most recent data (Current State)
    # We group by store and sku to get the last known day for every product [cite: 141]
    latest_data = df.sort_values("date").groupby(["store_id", "sku"]).last().reset_index()
    
    # 4. Feature Preparation (Production Consistency)
    # Ensure categorical columns are encoded exactly as they were during training
    X = latest_data.copy()
    
    for col in X.columns:
        if pd.api.types.is_string_dtype(X[col]) or X[col].dtype == "object":
            X[col] = X[col].astype("category")
        
        if str(X[col].dtype) == "category":
            X[col] = X[col].cat.codes.astype("int32")

    # Select only the features the model expects
    X_input = X[feature_cols]

    # 5. Generate Predictions [cite: 140]
    # np.maximum(0, ...) ensures we don't predict negative sales
    preds = model.predict(X_input)
    point_forecasts = np.maximum(0, preds)
    
    # 6. Build the Forecast Table [cite: 141]
    forecast_df = latest_data[["store_id", "sku", "date"]].copy()
    forecast_df["point_forecast"] = point_forecasts
    forecast_df["forecast_date"] = forecast_df["date"] + pd.Timedelta(days=1)
    
    # Adding confidence intervals (Heuristic for demo: 10% margin) [cite: 141]
    forecast_df["lo80"] = forecast_df["point_forecast"] * 0.9
    forecast_df["hi80"] = forecast_df["point_forecast"] * 1.1
    forecast_df["model_id"] = "xgb_global_v1"
    
    # 7. Save results [cite: 141]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_parquet(out_path, index=False)
    
    print(f"Success: Generated {len(forecast_df)} forecasts.")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    generate_forecasts()
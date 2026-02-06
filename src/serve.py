from fastapi import FastAPI, HTTPException
import pandas as pd
from pathlib import Path
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Retail Forecast API", description="API for SKU-level demand forecasting and inventory KPIs")

# Define data paths
FORECAST_PATH = "data/forecasts/fact_forecast.parquet"
KPI_PATH = "data/processed/kpis.parquet"

def load_data(path: str):
    if not Path(path).exists():
        raise HTTPException(status_code=500, detail=f"Data file {path} not found. Run the pipeline first.")
    return pd.read_parquet(path)

@app.get("/health")
def health():
    """Returns the API status."""
    return {"status": "ok", "message": "Retail Forecast Service is running"}


@app.get("/forecast")
def get_forecast(store_id: str, sku: str):
    """
    Returns the predicted sales for a specific store and product.
    Example: /forecast?store_id=1&sku=101
    """
    df = load_data(FORECAST_PATH)
    # Filter for the specific store and sku
    result = df[(df["store_id"] == str(store_id)) & (df["sku"] == str(sku))]
    
    if result.empty:
        raise HTTPException(status_code=404, detail="No forecast found for the given Store ID and SKU.")
    
    return result.to_dict(orient="records")[0]

@app.get("/kpi")
def get_kpi(store_id: str, sku: str):
    """
    Returns Inventory KPIs (Stockout Risk, Days of Cover) for a specific item.
    """
    df = load_data(KPI_PATH)
    result = df[(df["store_id"] == str(store_id)) & (df["sku"] == str(sku))]
    
    if result.empty:
        raise HTTPException(status_code=404, detail="No KPIs found for the given Store ID and SKU.")
    
    return result.to_dict(orient="records")[0]
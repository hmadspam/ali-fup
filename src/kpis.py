import pandas as pd
from pathlib import Path

def compute_inventory_kpis(forecast_path="data/forecasts/fact_forecast.parquet",
                           processed_path="data/processed/clean_sales.parquet",
                           out_path="data/processed/kpis.parquet"):
    """
    Calculates business metrics by comparing forecasts against current stock.
    """
    # 1. Load the forecast we just generated
    if not Path(forecast_path).exists():
        raise FileNotFoundError(f"Forecast file not found at {forecast_path}")
    f_df = pd.read_parquet(forecast_path)

    # 2. Load the cleaned sales data to get the latest inventory levels
    if not Path(processed_path).exists():
        raise FileNotFoundError(f"Cleaned data not found at {processed_path}")
    sales_df = pd.read_parquet(processed_path)

    # 3. Get the most recent inventory level for each Store/SKU
    current_inv = sales_df.sort_values("date").groupby(["store_id", "sku"])["inventory_level"].last().reset_index()

    # 4. Merge forecast with current inventory
    kpi_df = pd.merge(f_df, current_inv, on=["store_id", "sku"])

    # 5. Calculate KPIs
    # Stockout Risk: True if current stock is less than predicted demand [cite: 146, 147]
    kpi_df["stockout_risk"] = (kpi_df["inventory_level"] < kpi_df["point_forecast"]).astype(int)

    # Days of Cover: inventory / predicted daily demand [cite: 145]
    # We add a small epsilon (1e-6) to prevent division by zero
    kpi_df["days_of_cover"] = (kpi_df["inventory_level"] / (kpi_df["point_forecast"] + 1e-6)).round(2)

    # Sell-through rate estimate [cite: 148]
    kpi_df["sell_through_est"] = (kpi_df["point_forecast"] / (kpi_df["point_forecast"] + kpi_df["inventory_level"] + 1e-6)).round(4)

    # 6. Save for Power BI [cite: 149]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    kpi_df.to_parquet(out_path, index=False)
    
    print(f"KPIs computed for {len(kpi_df)} items.")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    compute_inventory_kpis()
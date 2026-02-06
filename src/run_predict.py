from src.predict import generate_forecasts

# Define paths consistent with the project structure [cite: 14, 26]
FEATURES_PATH = "data/processed/features.parquet"
MODEL_PATH = "models/xgb_global.pkl"
OUT_PATH = "data/forecasts/fact_forecast.parquet"

if __name__ == "__main__":
    print("Starting Forecast Generation Pipeline...")
    try:
        generate_forecasts(
            features_path=FEATURES_PATH,
            model_path=MODEL_PATH,
            out_path=OUT_PATH
        )
        print("Pipeline Completed Successfully.")
    except Exception as e:
        print(f"Pipeline Failed: {e}")
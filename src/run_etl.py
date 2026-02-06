from src.etl import run_etl

RAW_PATH = "data/raw/sales_data.csv"
OUT_PATH = "data/processed/clean_sales.parquet"

if __name__ == "__main__":
    report = run_etl(RAW_PATH, OUT_PATH)
    print(report)

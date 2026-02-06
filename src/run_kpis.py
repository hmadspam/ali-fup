from src.kpis import compute_inventory_kpis

if __name__ == "__main__":
    print("Calculating Inventory KPIs...")
    try:
        compute_inventory_kpis()
        print("KPI Calculation Complete.")
    except Exception as e:
        print(f"Failed to calculate KPIs: {e}")
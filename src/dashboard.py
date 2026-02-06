import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Retail Intel | Inventory AI",
    page_icon="📦",
    layout="wide"
)

# --- CLEAN UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .status-badge {
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .risk { background-color: #ffebee; color: #c62828; }
    .safe { background-color: #e8f5e9; color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    kpi_path = Path("data/processed/kpis.parquet")
    return pd.read_parquet(kpi_path) if kpi_path.exists() else None

df = load_data()

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    
    if df is not None:
        stores = sorted(df["store_id"].unique())
        selected_stores = st.multiselect("Select Stores", stores, default=stores[:3])
        
        st.markdown("### 📊 View Settings")
        show_risks_only = st.toggle("Focus on Stockout Risks", value=False)
        
        st.divider()
        st.info("This dashboard uses XGBoost forecasts to optimize inventory levels.")

# --- MAIN DASHBOARD ---
if df is not None:
    # Filtering Logic
    filtered_df = df[df["store_id"].isin(selected_stores)]
    if show_risks_only:
        filtered_df = filtered_df[filtered_df["stockout_risk"] == 1]

    # TITLE SECTION
    st.title("📦 Retail Inventory Intelligence")
    st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # ROW 1: TOP METRICS
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Items", len(filtered_df))
    with m2:
        risk_count = len(filtered_df[filtered_df["stockout_risk"] == 1])
        st.metric("At Risk", risk_count, delta=f"{risk_count} SKUs", delta_color="inverse")
    with m3:
        st.metric("Avg Forecast", f"{filtered_df['point_forecast'].mean():.1f}")
    with m4:
        st.metric("Stock Health", f"{((len(filtered_df)-risk_count)/len(filtered_df)*100):.0f}%")

    st.markdown("---")

    # ROW 2: VISUAL ANALYTICS
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📈 Demand Forecast vs. Current Stock")
        # Creating a bar chart that compares Inventory vs Forecast
        fig = px.bar(
            filtered_df.head(20), 
            x="sku", 
            y=["inventory_level", "point_forecast"],
            barmode="group",
            labels={"value": "Units", "variable": "Type", "sku": "Product SKU"},
            color_discrete_sequence=["#1f77b4", "#ff7f0e"],
            template="plotly_white"
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🎯 Risk Distribution")
        risk_pie = px.pie(
            filtered_df, 
            names=filtered_df["stockout_risk"].map({1: "🔴 Risk", 0: "🟢 Safe"}),
            hole=0.6,
            color_discrete_sequence=["#2e7d32", "#c62828"]
        )
        st.plotly_chart(risk_pie, use_container_width=True)

    # ROW 3: DETAILED TABLE
    st.subheader("📋 Detailed Inventory Report")
    
    # Simple Export
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report (CSV)", csv, "inventory_report.csv", "text/csv")

    # Display Styled Table
    st.dataframe(
        filtered_df[["store_id", "sku", "inventory_level", "point_forecast", "days_of_cover", "stockout_risk"]],
        column_config={
            "inventory_level": st.column_config.NumberColumn("Current Stock"),
            "point_forecast": st.column_config.NumberColumn("AI Forecast"),
            "days_of_cover": st.column_config.ProgressColumn("Days left", min_value=0, max_value=30, format="%.1f"),
            "stockout_risk": st.column_config.CheckboxColumn("Critical?")
        },
        use_container_width=True,
        hide_index=True
    )

else:
    st.error("⚠️ Data not found! Please run your ETL and ML pipeline scripts first.")
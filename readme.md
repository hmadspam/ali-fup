Here is a comprehensive, professional `README.md` tailored for your Final Year Project. It is structured to satisfy both technical examiners (who look at the code) and business examiners (who look at the value).

---

# 📦 AI-Powered Retail Demand Forecasting & Inventory Optimization

**Final Year Project (FYP) - End-to-End Machine Learning Solution**

## 📝 Project Overview

This project addresses the "Out-of-Stock" and "Overstock" problems in the retail industry. Using historical sales data, we have developed a predictive engine that forecasts future demand at the Store-SKU level, allowing managers to optimize their supply chain proactively.

### Key Milestones

* **Predictive Engine:** XGBoost model achieving a validation RMSE of ~21.71.
* **Feature Engineering:** Advanced lag features, rolling averages, and seasonal indicators.
* **Production API:** High-performance REST API built with FastAPI.
* **Managerial Dashboard:** Real-time data visualization built with Streamlit.

---

## 🏗️ System Architecture

The system is divided into four distinct layers:

1. **ETL & Feature Layer:** Data cleaning and generation of 41 unique features (Lags, Seasonality, Weather).
2. **Model Layer:** Gradient Boosted Trees (XGBoost) trained on a 2-year sales history.
3. **Service Layer:** FastAPI backend that serves forecasts via JSON endpoints.
4. **Presentation Layer:** Streamlit frontend for inventory monitoring and risk detection.

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.10 or higher
* Manjaro/Linux environment (or WSL2)

### 1. Environment Configuration

```bash
# Clone the repository
git clone https://github.com/hmadspam/ali-fup
cd ali-fyp

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Data Pipeline Execution

Run the following modules in order to process data and train the model:

```bash
python -m src.etl           # Ingest & Clean Data
python -m src.features      # Feature Engineering
python -m src.train         # Model Training & Evaluation
python -m src.run_predict   # Generate Future Forecasts
python -m src.run_kpis      # Compute Inventory KPIs

```

### 3. Launching Services

To view the results, launch both the API and the Dashboard:

**API (Technical View):**

```bash
python -m uvicorn src.serve:app --reload --port 8000

```

*Access at: `http://127.0.0.1:8000/docs*`

**Dashboard (Manager View):**

```bash
streamlit run src/dashboard.py

```

*Access at: `http://localhost:8501*`

---

## 📊 Model Evaluation

The model was evaluated using **Root Mean Squared Error (RMSE)** and **Mean Absolute Percentage Error (MAPE)**.

| Metric | Value |
| --- | --- |
| **RMSE** | 21.71 |
| **MAPE** | 27.97% |
| **Features Used** | 41 |
| **Training Rows** | 70,200 |

---

## 📁 Project Structure

```text
ali-fyp/
├── data/
│   ├── raw/                # Original sales_data.csv
│   ├── processed/          # Features and KPI parquet files
│   └── forecasts/          # Model output (fact_forecast.parquet)
├── models/
│   └── xgb_global.pkl      # Trained XGBoost model artifact
├── src/
│   ├── etl.py              # Data cleaning
│   ├── features.py         # Feature engineering logic
│   ├── train.py            # Training script
│   ├── serve.py            # FastAPI implementation
│   └── dashboard.py        # Streamlit UI
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

```

---

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **ML Framework:** XGBoost, Scikit-Learn
* **Data Handling:** Pandas, PyArrow (Parquet)
* **API:** FastAPI, Uvicorn
* **Visualization:** Streamlit, Plotly

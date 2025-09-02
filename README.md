# Cortex-o1 (Neural Minds)
  Advanced Market Analysis & AI-Powered Prediction Platform — Streamlit app for end-to-end stock exploration, model training, forecasting, and explainability.

[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2)](https://github.com/shap/shap)
[![Plotly](https://img.shields.io/badge/Plotly-Visualizations-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/python/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optional-EB5F0C?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![yfinance](https://img.shields.io/badge/yfinance-Data%20API-0096D6)](https://github.com/ranaroussi/yfinance)
[![Alpha Vantage](https://img.shields.io/badge/Alpha%20Vantage-API-0066CC)](https://www.alphavantage.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](#)


## ✨ Overview
**Neural Minds** is a Streamlit app that brings together:
- 📡 **Market Data**: Pull OHLCV data from **yfinance**, fallback to **Alpha Vantage**, or synthetic data.
- 🧮 **Feature Engineering**: Add MA(20/50), RSI, returns, log returns, volatility, momentum, z-scores, lagged features, and volume signals.
- 🤖 **Modeling**: Train ML models (RF, GBM, Ridge, Lasso, XGBoost optional) inside a scikit-learn pipeline.
- 📊 **Evaluation**: Walk-forward CV with `TimeSeriesSplit` + hold-out backtest.
- 🔮 **Forecasting**: Iterative next-day and multi-day predictions (return% or price).
- 🧩 **Explainability**: Permutation Importance + SHAP waterfall (if installed).
- 🎨 **UI**: Streamlit dashboard with styled header, sidebar controls, API health checks, and interactive tabs.

---

## 🧱 Architecture
- **Data**  
  Fetch OHLCV via yfinance → Alpha Vantage → sample generator. Market metadata: sector, industry, currency, market cap.
- **Features**  
  Rolling indicators, RSI, pct/log returns, rolling vol, z-score, lags of price & returns.
- **Targets**  
  `return` (next-day %) or `price` (next-day level).
- **Models**  
  RF, GBM, Ridge, Lasso, optional XGBoost. Parameter grids for randomized tuning.
- **Evaluation**  
  Walk-forward CV (`TimeSeriesSplit`) and last-20% hold-out backtest.
- **Forecasting**  
  Iterative multi-day business-calendar forecasts with return compounding.
- **Explainability**  
  Global (permutation importance) + Local (SHAP) feature contributions.

---

## 🧪 Feature Set
Computed features include:
- Core: `Open, High, Low, Close, Volume`
- Indicators: `MA_20`, `MA_50`, `RSI`
- Returns: `%Change`, `Log_Return`
- Volatility: `Vol_5`, `Vol_20`
- Momentum: `Mom_5`
- Z-score: `Z_20`
- Volume trend: `Volume_MA`
- Lags: `Close_Lag_{1,2,3,5}`, `Ret_Lag_{1,2,3,5}`

Targets are aligned to **future values** (`shift(-horizon)`).

---

## 🤖 Modeling
- **Pipeline**: `SimpleImputer → StandardScaler → Model`
- **CV**: `TimeSeriesSplit` with fold-wise RMSE, MAE, R²
- **Backtest**: Last 20% of history, RMSE/MAE/R²
- **Forecasts**:
  - Return target → predicts % return, compounds into price, classifies signal (Bullish/Bearish/Neutral).
  - Price target → direct next-day level.

---

## 🔍 Explainable AI
- **Global**: Feature ranking with **Permutation Importance**
- **Local**: SHAP waterfall for last prediction + narrative summary  
  *(If SHAP not installed, falls back to global only)*

---

## 🧭 UI
The app has 5 tabs:
1. **Stock Analysis** — profile, stats, candlestick + indicators  
2. **Predictions** — features, CV table, backtest, forecast block  
3. **Charts** — forward projection, RSI view  
4. **Model Performance** — model summary, feature list, CV means  
5. **Data Table** — processed dataset view/export  

---

## 🛠️ Installation
Python 3.9+

```bash
# 1) Create a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install streamlit pandas numpy plotly scikit-learn shap matplotlib
pip install yfinance requests
pip install xgboost   # optional


### Alpha Vantage key (optional but recommended)
Create a free API key at Alpha Vantage and make it available to the app, for example via environment variable or Streamlit secrets:

- **Environment**: `export ALPHAVANTAGE_API_KEY=your_key`
- **Streamlit secrets**: create `.streamlit/secrets.toml` with
  ```toml
  ALPHAVANTAGE_API_KEY = "your_key"
  ```

Market cap and some metadata rely on this key; price history typically comes from `yfinance`.

---

## ▶️ Run the app
```bash
streamlit run Cortex-o1.py
```
Then open the local URL printed by Streamlit.

---

## ⚠️ Notes & limitations
- Short histories and near‑zero autocorrelation reduce predictive power; trust CV/backtest metrics.
- 1‑day horizon is implemented end‑to‑end. Extending to multi‑step forecasting would require minor changes to feature alignment and plotting.
- Financial data can be messy; if APIs are down or rate‑limited, some metadata (e.g., market cap) may be unavailable.
- **Not financial advice** — use responsibly.

---

## 🧩 File layout
This project is intentionally a **single file** for easy deployment:
```
Cortex-o1.py  # Streamlit app, features, models, plots, explainability
```

---

## ✅ Checklist to reproduce a forecast
1. Install requirements and set the Alpha Vantage key (optional).
2. `streamlit run Cortex-o1-updated-integrated.py`
3. Choose market + ticker, period, target type.
4. Pick a model and CV strategy; run the **Predictions** tab.
5. Inspect CV table, hold‑out backtest, and the one‑step forecast.
6. Open **Explainable AI** to review feature importance and the SHAP waterfall.

---

## Credits
Built with Streamlit, scikit‑learn, SHAP, Plotly, and XGBoost/yfinance/Alpha Vantage.

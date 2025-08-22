# Cortex‑o1 (Neural Minds)
  Advanced Market Analysis & AI‑Powered Prediction Platform - Streamlit app for end‑to‑end stock exploration, model training, and explainability.

[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2)](https://github.com/shap/shap)
[![Plotly](https://img.shields.io/badge/Plotly-Visualizations-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/python/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optional-EB5F0C?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![yfinance](https://img.shields.io/badge/yfinance-Data%20API-0096D6)](https://github.com/ranaroussi/yfinance)
[![Alpha Vantage](https://img.shields.io/badge/Alpha%20Vantage-API-0066CC)](https://www.alphavantage.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](#)


## ✨ What this app does
Cortex‑o1 is a single‑file Streamlit application that lets you:

- Pull historical price/volume data for US and Indian tickers.
- Enrich it with practical technical features (MA_20/50, RSI, returns, volatility, momentum, z‑scores, volume MA, and price/return lags).
- Diagnose data quality and “predictability” before modeling.
- Train ML models (Random Forest, Gradient Boosting, Ridge, Lasso, optional XGBoost) inside a robust scikit‑learn `Pipeline` (imputer → scaler → model).
- Evaluate with walk‑forward cross‑validation and a hold‑out backtest (RMSE/MAE/R²).
- Generate a one‑step‑ahead forecast (return or price) and show signal strength.
- Explain predictions with global (permutation importance) and local (SHAP waterfall) interpretability.
- Visualize candlesticks + indicators, backtests, and forward projections.

Everything lives in **`Cortex-o1-updated-integrated.py`**.

---

## 🧱 Architecture (high level)
- **Data**: historical OHLCV; optional metadata such as sector/industry/currency and market cap.
- **Feature engineering**: rolling means, RSI, price change & log return, rolling volatility, momentum, z‑score, volume MA, and lag features.
- **Targets**: next‑day return (%) or next‑day price level.
- **Model space**: Random Forest, Gradient Boosting, Ridge, Lasso (+ XGBoost if installed).
- **Evaluation**: `TimeSeriesSplit` CV + hold‑out backtest.
- **Explainability**: Permutation Importance (global) + SHAP (local waterfall) with a plain‑English summary.
- **UI**: Streamlit with themed header, an **API Status Check** expander, and five tabs:
  1. **Stock Analysis** (profile, key stats, candlestick + indicators)
  2. **Predictions** (modeling, CV table, backtest, and next‑day forecast)
  3. **Charts** (forecast overlay + RSI lens)
  4. **Model Performance** (feature list & cross‑validation summary)
  5. **Data Table** (raw processed dataframe)

---

## 🧪 Features & indicators
The app computes a stable set of features that exist for most equities:
- `Open, High, Low, Volume`
- `MA_20, MA_50`
- `RSI`
- `Price_Change, Log_Return`
- `Vol_5, Vol_20` (rolling volatility)
- `Mom_5` (momentum)
- `Z_20` (20‑period z‑score)
- `Volume_MA`
- Lags: `Close_Lag_{1,2,3,5}`, `Ret_Lag_{1,2,3,5}` (included when available)

Targets are aligned to the **next** time step (`horizon=1`): either **return %** or **price**.

A lightweight **diagnostics** routine summarizes data size, missingness, variance, autocorrelation, and emits warnings (e.g., short history, weak autocorrelation) along with a composite **predictability score** (0–100).

---

## 🤖 Modeling
- Models: **Random Forest**, **Gradient Boosting**, **Ridge**, **Lasso**, and optional **XGBoost**.
- Training pipeline: `SimpleImputer(strategy="median")` → `StandardScaler()` → model.
- Cross‑validation: walk‑forward `TimeSeriesSplit` (configurable folds).
- Backtest: last ~20% hold‑out, reporting **RMSE**, **MAE**, **R²**.
- One‑step‑ahead forecast:
  - **Return target** → converts to predicted next‑day price and classifies the signal (Strong/Mild Bullish, Neutral, Bearish).
  - **Price target** → reports the next‑day price level.

---

## 🔍 Explainable AI
- **Global**: sklearn **Permutation Importance** over the fitted pipeline; a horizontal bar chart ranks features by contribution.
- **Local**: **SHAP** waterfall for the most recent prediction, plus a short narrative listing the top contributing features and the net bullish/bearish effect.

> SHAP is optional; if unavailable, only global importance will render.

---

## 🧭 UI walkthrough
- **Header & API Status**: A styled “Neural Minds” header and an **API Status Check** expander to test yfinance / Alpha Vantage connectivity.
- **Sidebar controls** (typical):
  - Market (US / India), ticker selector (common tickers pre‑listed), date range/period, target type (return vs price), model choice, CV strategy.
- **Tab 1 – Stock Analysis**: Company/asset profile, sector/industry, currency/market cap (if available), **Key Stats** (52‑week high/low, average volume, RSI with overbought/oversold badges), and a candlestick chart with moving averages.
- **Tab 2 – Predictions**: Feature set preview, CV table (if walk‑forward enabled), backtest chart (actual vs predicted), next‑day prediction block with price/return and signal badges.
- **Tab 3 – Charts**: Forward projection chart and an RSI view to context‑check signals.
- **Tab 4 – Model Performance**: Summary of selected model, input feature list, CV means.
- **Tab 5 – Data Table**: Full processed dataframe; useful for export/debug.

---

## 🛠️ Installation
Tested with Python 3.9+.

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install streamlit pandas numpy plotly scikit-learn
pip install shap matplotlib
pip install xgboost           # optional
pip install yfinance requests # for data sources
```

> If SHAP or XGBoost is not installed, the app gracefully hides those parts.

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
Cortex-o1-updated-integrated.py  # Streamlit app, features, models, plots, explainability
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

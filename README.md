[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2)](https://github.com/shap/shap)
[![Plotly](https://img.shields.io/badge/Plotly-Visualizations-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/python/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optional-EB5F0C?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![yfinance](https://img.shields.io/badge/yfinance-Data%20API-0096D6)](https://github.com/ranaroussi/yfinance)
[![Alpha Vantage](https://img.shields.io/badge/Alpha%20Vantage-API-0066CC)](https://www.alphavantage.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](#)

### Neural Minds
  Advanced Market Analysis & AI-Powered Prediction Platform — Streamlit app for end-to-end stock exploration, model training, forecasting, and explainability.


### ✨ Premium Features
- 🔄 **Multi-API Integration -** Seamless fallback between yfinance & Alpha Vantage
- 📡 **API Status Check -** Automatic health check for data sources before fetching
- 🤖 **Advanced AI Models -** Random Forest, Gradient Boosting, Ridge, Lasso, XGBoost  
- 📊 **Comprehensive Analysis -** Technical indicators + market diagnostics  
- 🎨 **Premium Interface -** Clean, responsive UI with interactive widgets
- 📊 **Evaluation -** Walk-forward CV with `TimeSeriesSplit` + hold-out backtest. 
- 📈 **Real-time Charts -** Plotly-powered OHLC, RSI, and Volume analysis  
- 🔍 **Performance Metrics -** RMSE, MAE, R², CV results & predictability score  
- 🧩 **Explainable AI -** Global + Local model interpretation with plain-English narrative  
- 📥 **Smart Exports -** Downloadable reports for data, charts, signals & CV results
- 🎨 **UI -** Streamlit dashboard with styled header, sidebar controls, API health checks, and interactive tabs.

---

### 🎯 How It Works
1. 📡 **Select Data Source -** *Yahoo Finance (yfinance)* or *Alpha Vantage*  
2. ✅ **API Status Check -** System verifies if chosen API is healthy before fetching  
3. 🌍 **Select Market -** Choose **US Stocks** or **Indian Stocks**  
4. 📊 **Select Stock -** Pick from curated tickers or enter a custom symbol  
5. ⏱️ **Choose Time Period -** Analyze from **1 month → 5 years**  
6. 🧮 **Configure Forecasting -**  
   - Select **Model** (Random Forest, Gradient Boosting, Ridge, Lasso, XGBoost)  
   - Choose **Target Type** → Return (%) or Price (level)  
   - Pick **CV Strategy** → Walk-forward (5 folds) / Hold-out (20%)  
   - Enable **Hyperparameter Tuning** (set iteration budget)  
   - Set **Days to Predict** (1-30)  
7. 🤖 **AI Analysis -** Models learn market patterns & indicators  
8. 🔮 **Predictions -** Forecast returns or prices with confidence  
9. 📈 **Visualize & Explain -** Interactive charts, validation results, signals, narrative explanations  

---

### 📊 Feature Set
The predictive models use a rich set of engineered features that capture both **price action** and **market behavior**:

- 📈 **Moving Averages -** 20-day & 50-day simple moving averages  
- 📊 **RSI (Relative Strength Index) -** Measures overbought/oversold momentum  
- 🌐 **Volatility -** Rolling standard deviation of returns (risk measure)  
- ⚡ **Momentum -** Rate of change of prices to capture trends  
- ⏪ **Lag Features -** Shifted past values of price/returns (memory of past behavior)  
- 🔄 **Z-Scores -** Standardized deviations from rolling mean (mean reversion signal)  
- 📉 **Volume Analysis -** Raw & derived volume-based indicators for market activity

Targets are aligned to **future values** (`shift(-horizon)`)

---

### 🌍 Global Market Coverage

```bash
#### 🇺🇸 US Stocks (41)
**Tech Giants:**  
Apple (AAPL), ASML Holding N.V (ASML), Microsoft (MSFT), Alphabet/Google (GOOGL), Amazon (AMZN), Tesla (TSLA), NVIDIA (NVDA), Meta (META), Netflix (NFLX), Oracle (ORCL), Cisco (CSCO)  
**Finance & Asset Management:**  
JPMorgan (JPM), Goldman Sachs (GS), Morgan Stanley (MS), Citigroup (C), Bank of America (BAC),  
Visa (V), Mastercard (MA), BlackRock (BLK), State Street (STT), Northern Trust (NTRS),  
Berkshire Hathaway (BRK.B), Barclays (BCS), UBS (UBS), Deutsche Bank (DB)  
**Healthcare & Pharma:**  
Johnson & Johnson (JNJ), Pfizer (PFE), Merck (MRK), Eli Lilly (LLY), UnitedHealth (UNH)  
**Energy & Industrials:**  
ExxonMobil (XOM), Chevron (CVX), Boeing (BA), Lockheed Martin (LMT), Northrop Grumman (NOC), Ford (F), General Motors (GM)  
**Consumer & Retail:**  
Walmart (WMT), Procter & Gamble (PG), Coca-Cola (KO), PepsiCo (PEP), Disney (DIS)  
  ```

---

```bash
#### 🇮🇳 Indian Stocks (34)
**Conglomerates & Energy:**  
Reliance (RELIANCE.NS), ONGC (ONGC.NS), Adani Enterprises (ADANIENT.NS), Adani Green (ADANIGREEN.NS), Adani Ports (ADANIPORTS.NS)  
**IT & Tech:**  
TCS (TCS.NS), Infosys (INFY.NS), Wipro (WIPRO.NS), Tech Mahindra (TECHM.NS), HCL Technologies (HCLTECH.NS)  
**Banking & Finance:**  
HDFC Bank (HDFCBANK.NS), ICICI Bank (ICICIBANK.NS), Kotak Bank (KOTAKBANK.NS), SBI (SBIN.NS), Axis Bank (AXISBANK.NS), Bajaj Finance (BAJFINANCE.NS)  
**Consumer & FMCG:**  
Hindustan Unilever (HINDUNILVR.NS), ITC (ITC.NS), Asian Paints (ASIANPAINT.NS), Nestle India (NESTLEIND.NS), Maruti Suzuki (MARUTI.NS)  
**Industrials & Materials:**  
Tata Motors (TATAMOTORS.NS), Mahindra & Mahindra (M&M.NS), Tata Steel (TATASTEEL.NS), JSW Steel (JSWSTEEL.NS), UltraTech Cement (ULTRACEMCO.NS)  
**Healthcare & Pharma:**  
Sun Pharma (SUNPHARMA.NS), Dr. Reddy's (DRREDDY.NS), Cipla (CIPLA.NS), Apollo Hospitals (APOLLOHOSP.NS)  
**Defense & Telecom:**  
Paras Defence (PARAS.NS), HAL (HAL.NS), BEL (BEL.NS), Bharti Airtel (BHARTIARTL.NS)  
  ```

---

### 🛠️ Technical Features
- 📊 **Data Loaded Info -** Number of data points fetched for the selected stock  
- 🧠 **Models Supported -** RF, GBM, Ridge, Lasso, XGBoost  
- 🔁 **Validation -** Walk-forward CV, Hold-out tests, predictability scoring  
- 📊 **Indicators -** Market trend & behavior signals  
- 🚦 **Trading Signals -** Neutral, Mild/Strong Bullish, Mild/Strong Bearish  
- 📈 **Visualizations -** Interactive OHLC & Volume charts, RSI Momentum, Feature Importance  
- ⚡ **Explainable AI -** Global (Permutation Importance) + Local (SHAP Waterfall & Narrative)  
- 📥 **Exports -** Data table (CSV), Charts, CV summary  

---


### 🔮 AI Prediction Workflow
```bash
🔌 API Status Check
└─ Yahoo Finance | Alpha Vantage
    │
    ▼
📊 Data Loaded
└─ # of Data Points Fetched
    │
    ▼
🏦 Market Selection
└─ 🇺🇸 US Stocks | 🇮🇳 Indian Stocks
    │
    ▼
🤖 Machine Learning Models
├─ Random Forest | Gradient Boosting | Ridge | Lasso | XGBoost*
├─ CV Strategy → Walk-forward (5 folds) | Hold-out (20%)
├─ ⚡ Hyperparameter Tuning (1-50 iterations)
└─ 📅 Prediction Horizon (1-30 days)
    │
    ▼
🛠️ Feature Engineering
├─ 📈 Moving Averages (20d, 50d)
├─ 📊 RSI (Momentum Oscillator)
├─ 🌐 Volatility (Std. Dev. of Returns)
├─ ⚡ Momentum (Rate of Change)
├─ ⏪ Lag Features (t-1, t-2, …)
├─ 🔄 Z-Scores (Mean Reversion)
└─ 📉 Volume Analysis
    │
    ▼
🔮 Predictions & Signals
├─ 📊 Price Forecast | Return Forecast
├─ 🚦 Trading Signals → 🟢 Bullish | 🟡 Neutral | 🔴 Bearish
└─ 📥 Exports → CSV | Charts | CV Summary
    │
    ▼
🧩 Explainability
├─ 🌍 Global → Feature Importance
└─ 🎯 Local → SHAP Waterfall + 📝 Narrative Explanation

  ```

---

### 💡 Pro Tips
- 📅 Use **longer timeframes (≥1y)** for more reliable training  
- 🌍 Always **consider macro & global context** with technicals  
- ⏳ Compare **short vs long horizon forecasts**  
- 🧪 Try **both CV strategies** for robustness  
- ⚡ Run ≥20 tuning iterations for stronger models  
- 🛡 Diversify: never rely on one stock or sector


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
The app has 6 tabs:
1. **Stock Analysis** — profile, stats, candlestick + indicators  
2. **Predictions** — features, CV table, backtest, forecast block  
3. **Charts** — forward projection, RSI view  
4. **Model Performance** — model summary, feature list, CV means  
5. **Data Table** — processed dataset view/export
6. **Explainable AI** - **Global** - Permutation Importance and **Local** - SHAP Waterfall & Narrative Explaination

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

---

## ▶️ Run the app
```bash
streamlit run Cortex-o1.py
```
or
```bash
https://neuralminds-gwjl9asfu2xd8mqxjgysyy.streamlit.app/
```

---

## ⚠️ Notes

1. Short histories → weak predictions; rely on CV/backtest.
2. Implemented horizon = 1 (iterative for multi-day).
3. APIs may fail/rate-limit. yfinance is main source.
4. Not financial advice.

---

## 🧩 File layout
This project is intentionally a **single file** for easy deployment:
```
Cortex-o1.py  # Streamlit app, features, models, plots, explainability
```

---

## ✅ Checklist to reproduce a forecast
1. Install requirements and set the Alpha Vantage key (optional).
2. `streamlit run Cortex-o1.py`
3. Choose market + ticker, period, target type.
4. Pick a model and CV strategy; run the **Predictions** tab.
5. Inspect CV table, hold‑out backtest, and the one‑step forecast.
6. Open **Explainable AI** to review feature importance and the SHAP waterfall.

---

## Credits
Built with Streamlit, scikit‑learn, SHAP, Plotly, and XGBoost/yfinance/Alpha Vantage.


"""
Cortex-o1 — Clean, consolidated Streamlit app for stock analysis & forecasting.
- One source of truth per function (no duplicates)
- Robust data loading (yfinance → Alpha Vantage → sample)
- Consistent pipeline step names ("imputer", "scaler", "model")
- Safe secrets handling and graceful fallbacks
- Correct CV / tuning (pipeline-prefixed hyperparams)
- Clear Explainability tab (Permutation Importance, Coeffs, optional SHAP for trees)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests

import streamlit as st

# Optional libs
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.inspection import permutation_importance

import plotly.graph_objects as go

# ---------------------
# Config / Constants
# ---------------------
st.set_page_config(page_title="Cortex-o1 — Clean", layout="wide")

ALPHA_VANTAGE_API_KEY = (
    st.secrets.get("ALPHA_VANTAGE_API_KEY") if hasattr(st, "secrets") else None
) or os.getenv("ALPHA_VANTAGE_API_KEY")

AV_BASE_URL = "https://www.alphavantage.co/query"

# Minimal but extensible mapping for tickers per data source
TICKER_MAP = {
    # yfinance uses .NS, Alpha Vantage generally plain symbol or exchange prefix
    ("RELIANCE", "yfinance"): "RELIANCE.NS",
    ("TCS", "yfinance"): "TCS.NS",
}

CSS = """
<style>
.api-status { padding: .5rem .75rem; border-radius: .5rem; margin: .25rem 0; font-weight: 600; }
.api-working { background: #e7f7ee; color: #0f5132; border: 1px solid #badbcc; }
.api-failed { background: #fde2e1; color: #842029; border: 1px solid #f5c2c7; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------
# Utilities
# ---------------------
def map_ticker_for_source(ticker: str, source: str) -> str:
    base = ticker.split(".")[0].upper()
    return TICKER_MAP.get((base, source), ticker)

@st.cache_data(ttl=300)
def fetch_stock_data_yfinance(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    if not YFINANCE_AVAILABLE:
        return None
    try:
        t = map_ticker_for_source(ticker, "yfinance")
        yf_period = {"1mo":"1mo","3mo":"3mo","6mo":"6mo","1y":"1y","2y":"2y","5y":"5y"}.get(period, "1y")
        df = yf.download(t, period=yf_period, interval="1d", auto_adjust=False)
        if df.empty:
            return None
        df = df.reset_index()
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        df = df[["Date","Open","High","Low","Close","Volume"]]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.attrs = {"source":"yfinance","ticker": t}
        return df
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_stock_data_alpha_vantage(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    if not ALPHA_VANTAGE_API_KEY:
        return None
    try:
        t = map_ticker_for_source(ticker, "alpha_vantage")
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": t,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "full",
            "datatype": "json",
        }
        time.sleep(1)  # be nice to their rate limits
        r = requests.get(AV_BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "Time Series (Daily)" not in data:
            return None
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.columns = ["Open","High","Low","Close","Volume"]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={"index":"Date"})
        days = {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(period, 365)
        start = datetime.now() - timedelta(days=days)
        df = df[df["Date"] >= start]
        df.attrs = {"source":"alpha_vantage","ticker": t}
        return df
    except Exception:
        return None

@st.cache_data(ttl=60)
def load_stock_data_auto(ticker: str, period: str = "1y") -> Tuple[pd.DataFrame, str, List[Tuple[str,str]]]:
    trace: List[Tuple[str, str]] = []
    df = fetch_stock_data_yfinance(ticker, period)
    if df is not None:
        trace.append(("yfinance", "✅ yfinance loaded successfully"))
        return df, "yfinance", trace
    else:
        trace.append(("yfinance", "❌ yfinance failed"))

    df = fetch_stock_data_alpha_vantage(ticker, period)
    if df is not None:
        trace.append(("alpha_vantage", "✅ Alpha Vantage loaded successfully"))
        return df, "alpha_vantage", trace
    else:
        trace.append(("alpha_vantage", "❌ Alpha Vantage failed"))

    df = create_sample_data(ticker, period)
    df.attrs["source"] = "sample_data"
    trace.append(("sample_data", "⚠️ Using sample data (both APIs unavailable)"))
    return df, "sample_data", trace

# ---------------------
# Sample data (fallback)
# ---------------------
def _period_days(period: str) -> int:
    return {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(period, 365)

def create_sample_data(ticker: str, period: str) -> pd.DataFrame:
    days = _period_days(period)
    base_prices = {
        "AAPL": 180, "GOOGL": 140, "MSFT": 330, "BLK": 700, "GS": 340, "STT": 70,
        "RELIANCE": 2500, "TCS": 3500, "INFY": 1500, "HDFCBANK": 1600, "ITC": 450,
    }
    base = ticker.split(".")[0].upper()
    base_price = base_prices.get(base, 1000)

    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    daily_return = 0.08/252
    volatility = 0.02
    rets = rng.normal(daily_return, volatility, days)
    prices = [base_price]
    for i in range(1, days):
        new_p = prices[-1] * (1 + rets[i])
        prices.append(float(np.clip(new_p, base_price*0.5, base_price*3.0)))

    data = []
    for i, close in enumerate(prices):
        if i == 0:
            open_ = close
        else:
            open_ = prices[i-1] * (1 + rng.normal(0, 0.005))
        intraday = abs(rng.normal(0, 0.015))
        high = max(open_, close) * (1 + intraday)
        low = min(open_, close) * (1 - intraday)
        volume = int(np.exp(rng.normal(np.log(500_000), 0.8)))
        data.append({"Date": dates[i], "Open": round(open_,2), "High": round(high,2),
                     "Low": round(low,2), "Close": round(close,2), "Volume": volume})
    df = pd.DataFrame(data)
    df.attrs = {"source":"sample_data", "ticker": ticker}
    return df

# ---------------------
# Feature Engineering & Diagnostics
# ---------------------
def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def process_stock_data(df: pd.DataFrame, ticker: str, source: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    keep = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep]

    def s1(frame: pd.DataFrame, col: str) -> pd.Series:
        if col not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        obj = frame[col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:,0]
        return pd.to_numeric(obj.squeeze(), errors="coerce")

    close = s1(df, "Close")
    volume = s1(df, "Volume") if "Volume" in df.columns else None

    df["MA_20"] = close.rolling(20).mean()
    df["MA_50"] = close.rolling(50).mean()
    df["RSI"] = calculate_rsi(close)
    df["Price_Change"] = close.pct_change()
    df["Log_Return"] = np.log(close).diff()
    df["Vol_5"] = df["Log_Return"].rolling(5).std()
    df["Vol_20"] = df["Log_Return"].rolling(20).std()
    df["Mom_5"] = close.pct_change(5)
    std20 = close.rolling(20).std()
    df["Z_20"] = (close - df["MA_20"]) / (std20 + 1e-9)
    df["Volume_MA"] = (volume.rolling(10).mean() if volume is not None and not volume.isna().all() else np.nan)

    for i in [1,2,3,5]:
        df[f"Close_Lag_{i}"] = close.shift(i)
        df[f"Ret_Lag_{i}"] = df["Price_Change"].shift(i)

    df["Fwd_Return_1d"] = close.pct_change().shift(-1) * 100.0
    df["Fwd_Price_1d"] = close.shift(-1)

    df = df.dropna().reset_index(drop=True)
    df.attrs = {"source": source, "ticker": ticker}
    return df

def prepare_supervised(df: pd.DataFrame, horizon: int = 1, target_type: str = "return") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_columns = [
        "Open","High","Low","Volume","MA_20","MA_50","RSI","Price_Change","Log_Return",
        "Vol_5","Vol_20","Mom_5","Z_20","Volume_MA",
        "Close_Lag_1","Close_Lag_2","Close_Lag_3","Close_Lag_5",
        "Ret_Lag_1","Ret_Lag_2","Ret_Lag_3","Ret_Lag_5",
    ]
    feature_columns = [c for c in feature_columns if c in df.columns]
    X = df[feature_columns].copy()
    if target_type == "return":
        y = df["Close"].pct_change().shift(-horizon) * 100.0
    else:
        y = df["Close"].shift(-horizon)
    X = X.iloc[:-horizon].reset_index(drop=True)
    y = y.iloc[:-horizon].reset_index(drop=True)
    return X, y, feature_columns

def data_diagnostics(df: pd.DataFrame) -> Dict:
    n = len(df)
    diag = {
        "rows": n,
        "date_span_days": int((df["Date"].max() - df["Date"].min()).days) if n else 0,
        "warnings": [],
    }
    miss = df.isna().mean().to_dict()
    diag["missing_max_pct"] = float(max(miss.values())) if miss else 0.0
    var_close = float(df["Close"].pct_change().dropna().var()) if n > 2 else 0.0
    diag["ret_var"] = var_close
    if n > 5:
        r = df["Close"].pct_change().dropna()
        diag["ret_autocorr_lag1"] = float(r.autocorr(lag=1)) if len(r) > 2 else 0.0
    else:
        diag["ret_autocorr_lag1"] = 0.0
    size_score = min(1.0, n/250.0)
    autocorr_score = (diag["ret_autocorr_lag1"] + 1)/2
    noise_penalty = float(np.exp(-5*min(var_close, 0.02)))
    score = 100*size_score*0.5 + 100*autocorr_score*0.3 + 100*noise_penalty*0.2
    diag["predictability_score"] = float(np.clip(score, 0, 100))
    if n < 120:
        diag["warnings"].append("Very little history (<120 rows); models can be unstable.")
    if diag["missing_max_pct"] > 0.05:
        diag["warnings"].append("Missing values >5%; consider different data source or period.")
    if abs(diag["ret_autocorr_lag1"]) < 0.02:
        diag["warnings"].append("Returns show weak autocorrelation; short-term forecasting will be hard.")
    return diag

# ---------------------
# Modeling
# ---------------------
def get_model_space(return_param_grids: bool = False):
    models: Dict[str, object] = {}
    param_grids: Dict[str, Dict[str, List]] = {}

    models["Random Forest"] = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    param_grids["Random Forest"] = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.8],
        "bootstrap": [True, False],
    }

    models["Gradient Boosting"] = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, min_samples_leaf=10, subsample=0.8, random_state=42
    )
    param_grids["Gradient Boosting"] = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [5, 10, 20],
        "subsample": [0.7, 0.8, 0.9],
    }

    models["Ridge"] = Ridge(alpha=1.0, random_state=42)
    param_grids["Ridge"] = {"alpha": [0.1, 1.0, 10.0]}

    models["Lasso"] = Lasso(alpha=0.001, random_state=42)
    param_grids["Lasso"] = {"alpha": [0.0001, 0.001, 0.01]}

    if return_param_grids:
        return models, param_grids
    return models

def time_series_cv_score(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse, mae, r2 = [], [], []
    for tr, te in tscv.split(X):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict(X.iloc[te])
        rmse.append(float(np.sqrt(mean_squared_error(y.iloc[te], p))))
        mae.append(float(mean_absolute_error(y.iloc[te], p)))
        r2.append(float(r2_score(y.iloc[te], p)))
    return {"rmse_mean": float(np.mean(rmse)), "mae_mean": float(np.mean(mae)), "r2_mean": float(np.mean(r2))}

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int = 5,
    do_tune: bool = False,
    tune_iter: int = 20,
) -> Tuple[Pipeline, pd.DataFrame, float, float, float]:
    space, grids = get_model_space(return_param_grids=True)
    if model_name not in space:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(space)}")

    base_model = space[model_name]
    param_grid = grids.get(model_name, {})

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", base_model),
    ])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        folds.append((i,
                      float(mean_squared_error(y.iloc[te], pred, squared=False)),
                      float(mean_absolute_error(y.iloc[te], pred)),
                      float(r2_score(y.iloc[te], pred))))
    cv_table = pd.DataFrame(folds, columns=["Fold","RMSE","MAE","R²"])
    mean_rmse, mean_mae, mean_r2 = cv_table["RMSE"].mean(), cv_table["MAE"].mean(), cv_table["R²"].mean()

    if do_tune and param_grid:
        prefixed = {f"model__{k}": v for k, v in param_grid.items()}
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=prefixed,
            n_iter=tune_iter,
            scoring="neg_root_mean_squared_error",
            cv=3,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y)
        final_pipe = search.best_estimator_
    else:
        final_pipe = pipe.fit(X, y)

    return final_pipe, cv_table, float(mean_rmse), float(mean_mae), float(mean_r2)

def backtest_holdout(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[Dict[str,float], pd.DataFrame]:
    n = len(X)
    split = int(n * (1 - test_size))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    m = {
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred)),
    }
    bt = pd.DataFrame({"Actual": yte.values, "Predicted": pred}, index=yte.index).reset_index(drop=True)
    return m, bt

# ---------------------
# Forecasting
# ---------------------
def next_business_day(date: datetime) -> datetime:
    d = date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

def iterative_forecast(df: pd.DataFrame, pipe: Pipeline, days: int = 1, target_type: str = "return") -> Tuple[List[float], List[datetime]]:
    sim = df.copy().reset_index(drop=True)
    preds: List[float] = []
    last_date = sim["Date"].iloc[-1]
    for _ in range(days):
        proc = process_stock_data(sim[["Date","Open","High","Low","Close","Volume"]].copy(), sim.attrs.get("ticker",""), sim.attrs.get("source",""))
        X_all, _, _ = prepare_supervised(proc, horizon=1, target_type=target_type)
        if X_all.empty:
            break
        x_last = X_all.iloc[[-1]]
        y_hat = float(pipe.predict(x_last)[0])
        preds.append(y_hat)
        new_date = next_business_day(last_date)
        if target_type == "return":
            last_close = float(sim["Close"].iloc[-1])
            new_close = last_close * (1 + y_hat/100.0)
        else:
            new_close = y_hat
        new_row = {
            "Date": new_date,
            "Open": new_close,
            "High": new_close*1.01,
            "Low": new_close*0.99,
            "Close": new_close,
            "Volume": float(sim["Volume"].iloc[-1]),
        }
        sim = pd.concat([sim, pd.DataFrame([new_row])], ignore_index=True)
        last_date = new_date
    fc_dates = pd.bdate_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=days).to_pydatetime().tolist()
    return preds, fc_dates

# ---------------------
# Company Info (simple mapping + fallback)
# ---------------------
STOCK_INFO: Dict[str, Dict[str,str]] = {
    "AAPL": {"name":"Apple Inc.", "sector":"Technology", "industry":"Consumer Electronics", "currency":"USD"},
    "MSFT": {"name":"Microsoft Corporation", "sector":"Technology", "industry":"Software—Infrastructure", "currency":"USD"},
    "RELIANCE": {"name":"Reliance Industries Ltd.", "sector":"Energy", "industry":"Oil & Gas", "currency":"INR"},
    "TCS": {"name":"Tata Consultancy Services Ltd.", "sector":"Technology", "industry":"IT Services", "currency":"INR"},
}

def get_stock_info(ticker: str, data_source: str = "yfinance") -> Dict[str, str]:
    base = ticker.split(".")[0].upper()
    info = STOCK_INFO.get(base, {}).copy()
    # best-effort enrich via yfinance if available
    if YFINANCE_AVAILABLE:
        try:
            yfi = yf.Ticker(map_ticker_for_source(ticker, "yfinance")).info
            info.setdefault("name", yfi.get("longName", base))
            info.setdefault("sector", yfi.get("sector", "Unknown"))
            info.setdefault("industry", yfi.get("industry", "Unknown"))
            info.setdefault("currency", yfi.get("currency", info.get("currency", "USD")))
        except Exception:
            pass
    if not info:
        info = {"name": base, "sector": "Unknown", "industry": "Unknown", "currency": "USD"}
    return info

# ---------------------
# UI
# ---------------------
def main():
    st.title("📈 Cortex‑o1 (clean)")

    # Sidebar
    st.sidebar.header("Inputs")
    ticker = st.sidebar.text_input("Ticker", value="RELIANCE.NS")
    period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
    data_source_choice = st.sidebar.selectbox("Preferred Source", ["Auto (yfinance→AV)", "yfinance", "Alpha Vantage"], index=0)

    st.sidebar.header("Model")
    model_name = st.sidebar.selectbox("Model", list(get_model_space().keys()), index=0)
    target_type = st.sidebar.radio("Target", ["return","price"], index=0, horizontal=True)
    n_splits = st.sidebar.slider("CV folds (TimeSeriesSplit)", 3, 8, 5)
    do_tune = st.sidebar.checkbox("Hyperparameter Tuning (fast)", value=False)
    tune_iter = st.sidebar.slider("Tuning iterations", 5, 50, 20)
    pred_days = st.sidebar.slider("Forecast days", 1, 30, 7)

    run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Fill inputs and click **Run Analysis** to start.")
        return

    # Data Loading
    with st.spinner("Fetching data..."):
        if data_source_choice == "yfinance":
            df = fetch_stock_data_yfinance(ticker, period)
            used_source = "yfinance" if df is not None else None
        elif data_source_choice == "Alpha Vantage":
            df = fetch_stock_data_alpha_vantage(ticker, period)
            used_source = "alpha_vantage" if df is not None else None
        else:
            df, used_source, trace = load_stock_data_auto(ticker, period)
            st.markdown("#### 🔎 API Call Status")
            for src, msg in trace:
                css = "api-working" if "✅" in msg else "api-failed" if "❌" in msg else "api-working"
                st.markdown(f'<div class="api-status {css}">{msg}</div>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.error("Could not fetch data from APIs. Falling back to sample data.")
        df = create_sample_data(ticker, period)
        used_source = "sample_data"

    # Process
    proc = process_stock_data(df, ticker, used_source)
    if proc is None or proc.empty:
        st.error("Unable to process stock data.")
        return

    # Diagnostics
    diag = data_diagnostics(proc)

    # Prepare supervised
    X, y, features = prepare_supervised(proc, horizon=1, target_type=target_type)

    # Train
    final_pipe, cv_table, mean_rmse, mean_mae, mean_r2 = train_model(
        X, y, model_name, n_splits=n_splits, do_tune=do_tune, tune_iter=tune_iter
    )

    # Layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Stock Analysis", "🔮 Predictions", "📈 Charts", "🤖 Model Performance", "📋 Data Table", "🧩 Explainability",
    ])

    # Tab 1 — Stock Analysis
    with tab1:
        info = get_stock_info(ticker, used_source)
        currency_symbol = "$" if info.get("currency","USD") == "USD" else "₹"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{diag['rows']}")
        c2.metric("Data span", f"{diag['date_span_days']} days")
        c3.metric("Max Missing", f"{diag['missing_max_pct']*100:.1f}%")
        c4.metric("Predictability", f"{diag['predictability_score']:.0f}/100")
        if diag["warnings"]:
            st.warning("\n".join([f"• {w}" for w in diag["warnings"]]))

        st.markdown(f"**{info['name']}** — Sector: {info['sector']} | Industry: {info['industry']} | Currency: {info['currency']}")

    # Tab 2 — Predictions
    with tab2:
        st.subheader("One‑step ahead")
        last_row = X.iloc[[-1]]
        y_hat = float(final_pipe.predict(last_row)[0])
        current_price = float(proc["Close"].iloc[-1])
        if target_type == "return":
            next_price = current_price * (1 + y_hat/100.0)
            delta = next_price - current_price
            pct = (delta/current_price)*100 if current_price != 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Current", f"{currency_symbol}{current_price:.2f}")
            c2.metric("Predicted return (1d)", f"{y_hat:.2f}%")
            c3.metric("Predicted price (1d)", f"{currency_symbol}{next_price:.2f}", f"{currency_symbol}{delta:.2f}")
        else:
            delta = y_hat - current_price
            pct = (delta/current_price)*100 if current_price != 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Current", f"{currency_symbol}{current_price:.2f}")
            c2.metric("Predicted price (1d)", f"{currency_symbol}{y_hat:.2f}", f"{currency_symbol}{delta:.2f}")
            c3.metric("Change", f"{pct:.2f}%")

        st.subheader("Multi‑day forecast")
        preds, dates = iterative_forecast(proc, final_pipe, days=pred_days, target_type=target_type)
        if preds:
            fig = go.Figure()
            if target_type == "return":
                price_path = [current_price]
                for r in preds:
                    price_path.append(price_path[-1]*(1 + r/100.0))
                price_path = price_path[1:]
                fig.add_trace(go.Scatter(x=dates, y=price_path, mode="lines+markers", name="Forecasted price"))
            else:
                fig.add_trace(go.Scatter(x=dates, y=preds, mode="lines+markers", name="Forecasted price"))
            fig.update_layout(template="plotly_white", title="Forecasted Price Path", xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Forecast horizon too short to compute.")

    # Tab 3 — Charts
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=proc['Date'], y=proc['Close'], mode='lines', name='Close', line=dict(width=3)))
        if 'MA_20' in proc.columns:
            fig.add_trace(go.Scatter(x=proc['Date'], y=proc['MA_20'], mode='lines', name='MA 20', line=dict(width=2, dash='dash')))
        if 'MA_50' in proc.columns:
            fig.add_trace(go.Scatter(x=proc['Date'], y=proc['MA_50'], mode='lines', name='MA 50', line=dict(width=2, dash='dot')))
        fig.update_layout(template='plotly_white', title=f"{ticker} — Price & MAs", xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)

        if 'Volume' in proc.columns:
            fig_v = go.Figure()
            fig_v.add_trace(go.Bar(x=proc['Date'], y=proc['Volume'], name='Volume', opacity=0.6))
            fig_v.update_layout(template='plotly_white', title=f"{ticker} — Volume", xaxis_title='Date', yaxis_title='Volume')
            st.plotly_chart(fig_v, use_container_width=True)

        if 'RSI' in proc.columns:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=proc['Date'], y=proc['RSI'], mode='lines', name='RSI', line=dict(width=2)))
            fig_r.add_hline(y=70, line_dash='dash', annotation_text='Overbought (70)')
            fig_r.add_hline(y=30, line_dash='dash', annotation_text='Oversold (30)')
            fig_r.update_layout(template='plotly_white', title=f"{ticker} — RSI")
            st.plotly_chart(fig_r, use_container_width=True)

    # Tab 4 — Model Performance
    with tab4:
        st.subheader("TimeSeries CV (walk‑forward)")
        st.dataframe(cv_table, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE (mean)", f"{mean_rmse:.4f}")
        c2.metric("MAE (mean)", f"{mean_mae:.4f}")
        c3.metric("R² (mean)", f"{mean_r2:.3f}")

        st.subheader("Hold‑out backtest (last 20%)")
        bt_metrics, bt_df = backtest_holdout(final_pipe, X, y, test_size=0.2)
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{bt_metrics['rmse']:.4f}")
        c2.metric("MAE", f"{bt_metrics['mae']:.4f}")
        c3.metric("R²", f"{bt_metrics['r2']:.3f}")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(y=bt_df['Actual'], mode='lines', name='Actual'))
        fig_bt.add_trace(go.Scatter(y=bt_df['Predicted'], mode='lines', name='Predicted'))
        fig_bt.update_layout(template='plotly_white', title='Backtest: Actual vs Predicted', xaxis_title='Observations', yaxis_title='Target')
        st.plotly_chart(fig_bt, use_container_width=True)

    # Tab 5 — Data Table
    with tab5:
        st.dataframe(proc.tail(200), use_container_width=True)

    # Tab 6 — Explainability
    with tab6:
        st.subheader("Permutation Importance (test on full data)")
        try:
            final_pipe.fit(X, y)
            pi = permutation_importance(final_pipe, X, y, n_repeats=10, random_state=42)
            importances = pd.DataFrame({
                "feature": features,
                "importance": pi.importances_mean,
            }).sort_values("importance", ascending=False)
            st.dataframe(importances, use_container_width=True)
            fig_pi = go.Figure()
            fig_pi.add_trace(go.Bar(x=importances["feature"], y=importances["importance"]))
            fig_pi.update_layout(template='plotly_white', title='Permutation Importance', xaxis_title='Feature', yaxis_title='Importance')
            st.plotly_chart(fig_pi, use_container_width=True)
        except Exception as e:
            st.warning(f"Permutation importance unavailable: {e}")

        # Coefficients for linear models
        model_obj = final_pipe.named_steps["model"]
        if isinstance(model_obj, (Ridge, Lasso)):
            st.subheader("Linear model coefficients")
            try:
                coefs = pd.DataFrame({"feature": features, "coef": model_obj.coef_}).sort_values("coef", key=np.abs, ascending=False)
                st.dataframe(coefs, use_container_width=True)
            except Exception:
                st.info("Coefficients not available.")

        # Optional SHAP for tree models only (fast)
        if SHAP_AVAILABLE and isinstance(model_obj, (RandomForestRegressor, GradientBoostingRegressor)):
            st.subheader("SHAP values (approx)")
            try:
                # Use model on transformed features
                X_trans = final_pipe.named_steps["scaler"].transform(final_pipe.named_steps["imputer"].transform(X))
                explainer = shap.Explainer(model_obj)
                shap_vals = explainer(X_trans[:200])  # cap for speed
                st.write("Computed SHAP values on first 200 rows.")
            except Exception as e:
                st.info(f"SHAP not available: {e}")

if __name__ == "__main__":
    main()

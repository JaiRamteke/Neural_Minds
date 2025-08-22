
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: xgboost (if available)
try:
    from xgboost import XGBRegressor  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Explainable AI imports (safe)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    PERM_AVAILABLE = True
except Exception:
    PERM_AVAILABLE = False

# Data sources
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Alpha Vantage
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
AV_BASE_URL = 'https://www.alphavantage.co/query'

# Page config
st.set_page_config(page_title="Neural Minds", page_icon="brain.png", layout="wide", initial_sidebar_state="expanded")

# ---------------------
# UI CSS (keep your look & feel)
# ---------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        .main-header { font-size: 3.5rem; font-weight: 700; background: linear-gradient(45deg, #1f77b4, #ff7f0e);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
            text-align: center; margin-bottom: 1rem; font-family: 'Inter', sans-serif; }
        .subtitle { text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 3rem; font-weight: 300; }
        .warning-card { background: #000000; padding: 1.5rem; border-radius: 8px; border: 1px solid #ffeaa7;
            margin-top: 2rem; border-left: 4px solid #fdcb6e; }
        .api-status { padding: 1rem; border-radius: 8px; margin: 1rem 0; }
        .api-working { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .api-failed { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .stButton > button { background: linear-gradient(45deg, #1f77b4, #ff7f0e); color: white; border: none;
            padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 600; font-size: 1rem; transition: all 0.3s ease; width: 100%; }
        .stButton > button:hover { background: linear-gradient(45deg, #1565c0, #f57c00); transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        section[data-testid="stSidebar"] { background:#f9f9f9 !important; color:#000 !important; }
        section[data-testid="stSidebar"] * { color:#000 !important; fill:#000 !important; }
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div{
            background:#ffffff !important; border:1px solid #ddd !important; border-radius:8px !important; }
        div[role="listbox"], ul[role="listbox"]{ background:#ffffff !important; color:#000000 !important;
            border:1px solid #ddd !important; border-radius:8px !important; }
        li[role="option"]{ color:#000 !important; }
        li[role="option"][aria-selected="true"], li[role="option"]:hover{ background:#f0f0f0 !important; }
        section[data-testid="stSidebar"] .stTextInput > div > div,
        section[data-testid="stSidebar"] .stNumberInput > div > div,
        section[data-testid="stSidebar"] .stDateInput > div > div{
            background:#ffffff !important; border:1px solid #ddd !important; border-radius:8px !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------
# Stock dictionaries (same as before, trimmed for brevity)
# ---------------------
RELIABLE_TICKERS = {
    "US Markets": {
        "AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc.", "MSFT": "Microsoft Corporation",
        "BLK": "BlackRock Inc.", "GS": "Goldman Sachs Group Inc.", "STT": "State Street Corporation",
        "TSLA": "Tesla Inc.", "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.", "NFLX": "Netflix Inc.", "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc."
    },
    "Indian Markets": {
        "RELIANCE.NSE": "Reliance Industries", "TCS.NSE": "Tata Consultancy Services", "PARAS.NSE": "Paras Defence and Space Technologies",
        "INFY.NSE": "Infosys Limited", "HDFCBANK.NSE": "HDFC Bank", "WIPRO.NSE": "Wipro Limited", "ITC.NSE": "ITC Limited",
        "SBIN.NSE": "State Bank of India", "TATAMOTORS.NSE": "Tata Motors", "TATASTEEL.NSE": "Tata Steel",
        "KOTAKBANK.NSE": "Kotak Mahindra Bank", "BHARTIARTL.NSE": "Bharti Airtel", "HINDUNILVR.NSE": "Hindustan Unilever"
    }
}

# ---------------------
# Helpers: mapping tickers and API checks
# ---------------------
def map_ticker_for_source(ticker: str, source: str) -> str:
    base = ticker.split('.')[0].upper()
    if source == "yfinance":
        return base + ".NS" if ticker.endswith(".NSE") else base
    if source == "alpha_vantage":
        return base + ".BSE" if ticker.endswith(".NSE") else base
    return ticker

def test_api_connections():
    status = {'yfinance': {'available': YFINANCE_AVAILABLE, 'working': False, 'message': ""},
              'alpha_vantage': {'available': True, 'working': False, 'message': ""}}
    if YFINANCE_AVAILABLE:
        try:
            test_data = yf.Ticker("AAPL").history(period="5d")
            if not test_data.empty:
                status['yfinance']['working'] = True
                status['yfinance']['message'] = "✅ yfinance is working"
            else:
                status['yfinance']['message'] = "❌ yfinance returned no data"
        except Exception as e:
            status['yfinance']['message'] = f"❌ yfinance error: {str(e)[:50]}..."
    else:
        status['yfinance']['message'] = "❌ yfinance not installed"

    try:
        params = {'function': 'TIME_SERIES_DAILY', 'symbol': 'AAPL', 'apikey': ALPHA_VANTAGE_API_KEY, 'outputsize': 'compact'}
        response = requests.get(AV_BASE_URL, params=params, timeout=15)
        data = response.json()
        if 'Time Series (Daily)' in data:
            status['alpha_vantage']['working'] = True
            status['alpha_vantage']['message'] = "✅ Alpha Vantage is working"
        elif 'Note' in data or 'Information' in data:
            status['alpha_vantage']['message'] = "⚠️ Alpha Vantage rate limit exceeded"
        elif 'Error Message' in data:
            status['alpha_vantage']['message'] = f"❌ Alpha Vantage error: {data['Error Message']}"
        else:
            status['alpha_vantage']['message'] = "❌ Unknown Alpha Vantage response"
    except Exception as e:
        status['alpha_vantage']['message'] = f"❌ Alpha Vantage connection failed: {str(e)[:50]}..."
    return status

@st.cache_data(ttl=300)
def fetch_stock_data_yfinance(ticker, period="1y"):
    try:
        ticker_mapped = map_ticker_for_source(ticker, "yfinance")
        yf_period_map = {'1mo':'1mo','3mo':'3mo','6mo':'6mo','1y':'1y','2y':'2y','5y':'5y'}
        yf_period = yf_period_map.get(period, '1y')
        df = yf.download(ticker_mapped, period=yf_period, interval="1d", auto_adjust=False)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        df = df[['Date','Open','High','Low','Close','Volume']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.attrs = {'source':'yfinance','ticker':ticker_mapped}
        return df
    except Exception as e:
        st.error(f"yfinance fetch error: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_stock_data_unified(ticker, period="1y"):
    try:
        mapped_ticker = map_ticker_for_source(ticker, "alpha_vantage")
        time.sleep(1)
        params = {'function': 'TIME_SERIES_DAILY', 'symbol': mapped_ticker, 'apikey': ALPHA_VANTAGE_API_KEY,
                  'outputsize': 'full', 'datatype': 'json'}
        response = requests.get(AV_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'Error Message' in data or 'Time Series (Daily)' not in data:
            return None
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={'index': 'Date'})
        days = get_period_days(period)
        start_date = datetime.now() - timedelta(days=days)
        df = df[df['Date'] >= start_date]
        df['Date'] = pd.to_datetime(df['Date'])
        df.attrs = {'source': 'alpha_vantage'}
        return df
    except Exception:
        return None

def load_stock_data_auto(ticker, period="1y"):
    trace = []
    if YFINANCE_AVAILABLE:
        df_yf = fetch_stock_data_yfinance(ticker, period)
        if df_yf is not None:
            trace.append(("yfinance", "✅ yfinance loaded successfully"))
            return df_yf, "yfinance", trace
        else:
            trace.append(("yfinance", "❌ yfinance failed (no/invalid data)"))
    else:
        trace.append(("yfinance", "❌ yfinance not installed"))
    df_av = fetch_stock_data_unified(ticker, period)
    if df_av is not None:
        trace.append(("alpha_vantage", "✅ Alpha Vantage loaded successfully"))
        return df_av, "alpha_vantage", trace
    else:
        trace.append(("alpha_vantage", "❌ Alpha Vantage failed (no/invalid data)"))
    df_sample = create_sample_data(ticker, period)
    df_sample.attrs['source'] = 'sample_data'
    trace.append(("sample_data", "⚠️ Using sample data (both APIs unavailable)"))
    return df_sample, "sample_data", trace

def get_period_days(period):
    return {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825}.get(period,365)

def create_sample_data(ticker, period):
    days = get_period_days(period)
    base_prices = {'AAPL': 180, 'GOOGL': 140, 'MSFT': 330, 'BLK': 700, 'GS': 340, 'STT': 70,
                   'TSLA': 250, 'AMZN': 140, 'NVDA': 450, 'META': 300, 'NFLX': 400, 'JPM': 150, 'V': 230,
                   'RELIANCE': 2500, 'TCS': 3500, 'PARAS': 700, 'INFY': 1500, 'HDFCBANK': 1600, 'WIPRO': 400,
                   'ITC': 450, 'SBIN': 600, 'TATAMOTORS': 650, 'TATASTEEL': 120, 'KOTAKBANK': 1900,
                   'BHARTIARTL': 850, 'HINDUNILVR': 2500}
    base_name = ticker.split('.')[0].upper()
    base_price = base_prices.get(base_name, 1000)
    np.random.seed(hash(ticker) % 2**32)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    daily_return = 0.08 / 252
    volatility = 0.02
    returns = np.random.normal(daily_return, volatility, days)
    prices = [base_price]
    for i in range(1, days):
        new_price = prices[-1] * (1 + returns[i])
        new_price = max(new_price, base_price * 0.5)
        new_price = min(new_price, base_price * 3.0)
        prices.append(new_price)
    data = []
    for i, close_price in enumerate(prices):
        daily_vol = abs(np.random.normal(0, 0.015))
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        intraday_range = abs(np.random.normal(0, daily_vol))
        high = max(open_price, close_price) * (1 + intraday_range)
        low = min(open_price, close_price) * (1 - intraday_range)
        high = max(open_price, close_price, high)
        low = min(open_price, close_price, low)
        base_volume = 1000000 if base_price < 1000 else 100000
        volume = int(np.random.lognormal(np.log(base_volume), 0.8))
        data.append({'Date': dates[i], 'Open': round(open_price, 2), 'High': round(high, 2),
                     'Low': round(low, 2), 'Close': round(close_price, 2), 'Volume': volume})
    df = pd.DataFrame(data)
    df.attrs = {'source': 'sample_data', 'ticker': ticker}
    return df

# ---------------------
# Feature engineering & diagnostics
# ---------------------
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_stock_data(df, ticker, source):
    if df is None or df.empty:
        return None
    if 'Date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
    # Technicals
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Price_Change'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close']).diff()
    df['Vol_5'] = df['Log_Return'].rolling(5).std()
    df['Vol_20'] = df['Log_Return'].rolling(20).std()
    df['Mom_5'] = df['Close'].pct_change(5)
    df['Z_20'] = (df['Close'] - df['MA_20']) / (df['Close'].rolling(20).std() + 1e-9)
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    for i in [1,2,3,5]:
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Ret_Lag_{i}'] = df['Price_Change'].shift(i)
    df = df.dropna().reset_index(drop=True)
    df.attrs = {'source': source, 'ticker': ticker, 'last_updated': datetime.now()}
    return df

def data_diagnostics(df):
    """Return a dict of diagnostics and a predictability score."""
    diag = {}
    n = len(df)
    diag['rows'] = n
    diag['date_span_days'] = int((df['Date'].max() - df['Date'].min()).days) if n else 0
    # missingness
    miss = df.isna().mean().to_dict()
    diag['missing_max_pct'] = float(max(miss.values())) if miss else 0.0
    # variance of features
    var_close = float(df['Close'].pct_change().dropna().var()) if n > 2 else 0.0
    diag['ret_var'] = var_close
    # autocorrelation of returns (lag1)
    if n > 5:
        r = df['Close'].pct_change().dropna()
        if len(r) > 2:
            diag['ret_autocorr_lag1'] = float(r.autocorr(lag=1))
        else:
            diag['ret_autocorr_lag1'] = 0.0
    else:
        diag['ret_autocorr_lag1'] = 0.0
    # simple predictability score (0-100)
    # higher with more data, higher autocorr, moderate variance (avoid extremely noisy)
    size_score = min(1.0, n/250.0)
    autocorr_score = (diag['ret_autocorr_lag1'] + 1)/2  # map [-1,1] -> [0,1]
    noise_penalty = np.exp(-5*min(var_close, 0.02))  # more noise -> lower score
    score = 100 * size_score * 0.5 + 100 * autocorr_score * 0.3 + 100 * noise_penalty * 0.2
    diag['predictability_score'] = float(np.clip(score, 0, 100))
    # flags
    diag['warnings'] = []
    if n < 120:
        diag['warnings'].append("Very little history (<120 rows); models can be unstable.")
    if diag['missing_max_pct'] > 0.05:
        diag['warnings'].append("Missing values >5%; consider different data source or period.")
    if abs(diag['ret_autocorr_lag1']) < 0.02:
        diag['warnings'].append("Returns show weak autocorrelation; short-term forecasting will be hard.")
    return diag

# ---------------------
# Supervised dataset construction
# ---------------------
def prepare_supervised(df, horizon=1, target_type="return"):
    """
    Build X, y with target aligned to the **next step**.
    target_type: 'return' (percent) or 'price' (level)
    """
    # Features (keep a stable list that exists)
    feature_columns = [
        'Open','High','Low','Volume','MA_20','MA_50','RSI','Price_Change','Log_Return',
        'Vol_5','Vol_20','Mom_5','Z_20','Volume_MA'
    ]
    for i in [1,2,3,5]:
        if f'Close_Lag_{i}' in df.columns: feature_columns.append(f'Close_Lag_{i}')
        if f'Ret_Lag_{i}' in df.columns: feature_columns.append(f'Ret_Lag_{i}')
    feature_columns = [c for c in feature_columns if c in df.columns]

    X = df[feature_columns].copy()

    if target_type == "return":
        # percent return of next day
        y = df['Close'].pct_change().shift(-horizon) * 100.0
    else:
        y = df['Close'].shift(-horizon)

    # drop the last 'horizon' rows (no target) and align
    X = X.iloc[:-horizon].reset_index(drop=True)
    y = y.iloc[:-horizon].reset_index(drop=True)
    return X, y, feature_columns

# ---------------------
# Model training, CV, selection
# ---------------------
def get_model_space():
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.001, random_state=42)
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
    return models

def time_series_cv_score(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_list, mae_list, r2_list = [], [], []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rmse_list.append(np.sqrt(mean_squared_error(yte, pred)))
        mae_list.append(mean_absolute_error(yte, pred))
        r2_list.append(r2_score(yte, pred))
    return {"rmse_mean": float(np.mean(rmse_list)), "mae_mean": float(np.mean(mae_list)), "r2_mean": float(np.mean(r2_list))}

def auto_select_model(X, y, selected_names, n_splits=5, do_tune=False, tune_iter=20, target_type="return"):
    space = get_model_space()
    candidates = {k:v for k,v in space.items() if (k in selected_names) or ("Auto" in selected_names)}
    results = []
    best = None
    best_score = np.inf  # minimize RMSE
    for name, mdl in candidates.items():
        scores = time_series_cv_score(mdl, X, y, n_splits=n_splits)
        results.append({"model": name, **scores})
        if scores["rmse_mean"] < best_score:
            best_score = scores["rmse_mean"]
            best = (name, mdl)
    # Optional fast tuning on the current best tree models
    tuned_pipe = None
    best_name, best_model = best
    if do_tune and best_name in ["Random Forest", "Extra Trees", "Gradient Boosting"]:
        param_grid = {}
        if best_name == "Random Forest":
            param_grid = {"m__n_estimators": [200,400,600,800],
                          "m__max_depth": [None, 6, 10, 14],
                          "m__min_samples_leaf": [1,2,4]}
        elif best_name == "Extra Trees":
            param_grid = {"m__n_estimators": [300,500,800],
                          "m__max_depth": [None, 6, 10, 14]}
        elif best_name == "Gradient Boosting":
            param_grid = {"m__n_estimators": [200,400,600],
                          "m__learning_rate": [0.03,0.05,0.08],
                          "m__max_depth": [2,3,4]}
        tuned_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", best_model)])
        try:
            rsearch = RandomizedSearchCV(tuned_pipe, param_distributions=param_grid, n_iter=min(tune_iter, max(3, tune_iter)),
                                         scoring="neg_root_mean_squared_error", n_jobs=-1, cv=TimeSeriesSplit(n_splits=n_splits),
                                         random_state=42, verbose=0)
            rsearch.fit(X, y)
            tuned_pipe = rsearch.best_estimator_
            best_score = -rsearch.best_score_
        except Exception as e:
            tuned_pipe = None
    # Final training (best or tuned) using full data
    final_pipe = tuned_pipe if tuned_pipe is not None else Pipeline([("imp", SimpleImputer(strategy="median")),
                                                                     ("sc", StandardScaler()),
                                                                     ("m", best_model)])
    final_pipe.fit(X, y)
    cv_table = pd.DataFrame(results).sort_values("rmse_mean")
    return best_name, final_pipe, cv_table, best_score

def backtest_holdout(pipe, X, y, test_size=0.2):
    n = len(X)
    split = int(n*(1-test_size))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred))
    }
    bt = pd.DataFrame({"Actual": yte.values, "Predicted": pred}, index=yte.index).reset_index(drop=True)
    return metrics, bt

# ---------------------
# Forecasting helpers
# ---------------------
def next_business_day(date):
    d = date + timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d += timedelta(days=1)
    return d

def iterative_forecast(df, pipe, days=1, target_type="return"):
    """Iteratively forecast 'days' into the future by recomputing features each step."""
    df_sim = df.copy().reset_index(drop=True)
    preds = []
    last_date = df_sim['Date'].iloc[-1]
    for _ in range(days):
        # Recompute features on the fly to get the latest row
        proc = process_stock_data(df_sim[['Date','Open','High','Low','Close','Volume']].copy(),
                                  df_sim.attrs.get('ticker', ''), df_sim.attrs.get('source',''))
        X_all, _, feats = prepare_supervised(proc, horizon=1, target_type=target_type)
        if X_all.empty: break
        x_last = X_all.iloc[[-1]]
        y_hat = float(pipe.predict(x_last)[0])
        preds.append(y_hat)
        # Convert return to price or use predicted price directly
        new_date = next_business_day(last_date)
        if target_type == "return":
            last_close = float(df_sim['Close'].iloc[-1])
            new_close = last_close * (1 + y_hat/100.0)
        else:
            new_close = y_hat
        # naive OHLC for placeholder
        new_open = new_close
        new_high = new_close * 1.01
        new_low = new_close * 0.99
        new_volume = float(df_sim['Volume'].iloc[-1])
        df_sim = pd.concat([df_sim, pd.DataFrame([{"Date": new_date, "Open": new_open, "High": new_high,
                                                   "Low": new_low, "Close": new_close, "Volume": new_volume}])],
                           ignore_index=True)
        last_date = new_date
    # Build forecast dataframe
    fc_dates = [next_business_day(df['Date'].iloc[-1] + timedelta(days=i)) for i in range(days)]
    return preds, fc_dates

# ---------------------
# Stock info (unchanged mapping)
# ---------------------
def get_stock_info(ticker):
    stock_info = {
        'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'currency': 'USD'},
        'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software', 'currency': 'USD'},
        'BLK': {'name': 'BlackRock, Inc.', 'sector': 'Financial Services', 'industry': 'Asset Management', 'currency': 'USD'},
        'GS': {'name': 'Goldman Sachs Group, Inc.', 'sector': 'Financial Services', 'industry': 'Capital Markets', 'currency': 'USD'},
        'STT': {'name': 'State Street Corporation', 'sector': 'Financial Services', 'industry': 'Asset Management', 'currency': 'USD'},
        'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet Services', 'currency': 'USD'},
        'AMZN': {'name': 'Amazon.com, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail', 'currency': 'USD'},
        'META': {'name': 'Meta Platforms, Inc.', 'sector': 'Communication Services', 'industry': 'Social Media', 'currency': 'USD'},
        'TSLA': {'name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'USD'},
        'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'industry': 'Semiconductors', 'currency': 'USD'},
        'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services', 'industry': 'Banks—Diversified', 'currency': 'USD'},
        'V': {'name': 'Visa Inc.', 'sector': 'Financial Services', 'industry': 'Credit Services', 'currency': 'USD'},
        'WMT': {'name': 'Walmart Inc.', 'sector': 'Consumer Defensive', 'industry': 'Discount Stores', 'currency': 'USD'},
        # Indian
        'RELIANCE': {'name': 'Reliance Industries Limited', 'sector': 'Energy', 'industry': 'Oil & Gas', 'currency': 'INR'},
        'TCS': {'name': 'Tata Consultancy Services', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
        'PARAS': {'name': 'Paras Defence and Space Technologies Ltd.', 'sector': 'Industrials', 'industry': 'Defense & Aerospace', 'currency': 'INR'},
        'INFY': {'name': 'Infosys Limited', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
        'HDFCBANK': {'name': 'HDFC Bank Limited', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
        'ICICIBANK': {'name': 'ICICI Bank Limited', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
        'HINDUNILVR': {'name': 'Hindustan Unilever Limited', 'sector': 'Consumer Defensive', 'industry': 'Household & Personal Products', 'currency': 'INR'},
        'BHARTIARTL': {'name': 'Bharti Airtel Limited', 'sector': 'Communication Services', 'industry': 'Telecom Services', 'currency': 'INR'},
        'SBIN': {'name': 'State Bank of India', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
        'ITC': {'name': 'ITC Limited', 'sector': 'Consumer Defensive', 'industry': 'Tobacco & FMCG', 'currency': 'INR'},
        'KOTAKBANK': {'name': 'Kotak Mahindra Bank Limited', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
    }
    base_ticker = ticker.split('.')[0].upper()
    info = stock_info.get(base_ticker, {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown', 'currency': 'USD'})
    info['market_cap'] = 'N/A'
    return info

# Safe display helper
def safe_stat(df, col, func, label, fmt="{:.2f}", currency_symbol=""):
    try:
        if df is not None and col in df.columns and not df[col].dropna().empty:
            val = func(df[col].dropna())
            if pd.notna(val):
                st.write(f"- {label}: {currency_symbol}{fmt.format(val)}")
                return
    except Exception:
        pass
    st.write(f"- {label}: Data not available")

# Explainable AI tab (compatible with Pipeline)
def render_explainable_ai_tab(pipe, df):
    st.markdown("### 🧠 Explainable AI")
    st.write("Feature attributions via **SHAP** when supported; otherwise **Permutation Importance**.")
    try:
        X_all, y_all, feature_names = prepare_supervised(df, horizon=1, target_type=st.session_state.get("target_type","return"))
    except Exception as e:
        st.error(f"Failed to prepare features for explanation: {e}")
        return
    if X_all is None or X_all.empty:
        st.error("No features available for explanation.")
        return
    try:
        # pull the wrapped model if possible
        model = getattr(pipe.named_steps.get("m", None), "feature_importances_", None)
    except Exception:
        model = None

    # SHAP for tree-based where possible
    used_shap = False
    if SHAP_AVAILABLE and hasattr(pipe.named_steps.get("m", None), "predict"):
        try:
            # we use a tree explainer when underlying model is tree-based
            inner = pipe.named_steps.get("m", None)
            name = inner.__class__.__name__.lower()
            if any(k in name for k in ["forest","tree","boost"]):
                Xs = pipe.named_steps["sc"].transform(pipe.named_steps["imp"].transform(X_all.values))
                explainer = shap.TreeExplainer(inner)
                shap_vals = explainer.shap_values(Xs)
                fig_bar = plt.figure(figsize=(8,5))
                shap.summary_plot(shap_vals, X_all, feature_names=feature_names, plot_type="bar", show=False)
                st.pyplot(fig_bar, clear_figure=True)
                used_shap = True
        except Exception as e:
            st.warning(f"SHAP failed ({e}). Falling back to permutation importance.")
    if not used_shap and PERM_AVAILABLE:
        try:
            Xs = pipe.named_steps["sc"].transform(pipe.named_steps["imp"].transform(X_all.values))
            inner = pipe.named_steps.get("m", None)
            result = permutation_importance(inner, Xs, y_all, n_repeats=8, random_state=42, n_jobs=-1)
            imp_df = pd.DataFrame({"feature": feature_names, "importance_mean": result.importances_mean,
                                   "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)
            fig_imp = px.bar(imp_df.head(15), x="importance_mean", y="feature", error_x="importance_std",
                             orientation="h", title="Top Features by Permutation Importance", template="plotly_white")
            fig_imp.update_layout(yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Permutation importance failed: {e}")

# ---------------------
# App
# ---------------------
def main():
    st.markdown('<h1 class="main-header">Neural Minds</h1>', unsafe_allow_html=True)
    st.markdown("""<p style='text-align:center;font-size:20px;font-weight:500;
        background:-webkit-linear-gradient(45deg,#4facfe,#00f2fe);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;margin-top:-10px;margin-bottom:20px;'>Advanced Market Analysis & AI-Powered Prediction Platform</p>""",
        unsafe_allow_html=True)

    # API status expander
    with st.expander("🔍 API Status Check", expanded=False):
        if st.button("🔄 Test API Connections", type="primary"):
            with st.spinner("Testing API connections..."):
                api_status = test_api_connections()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 yfinance Status")
                if api_status['yfinance']['working']:
                    st.markdown(f'<div class="api-status api-working">{api_status["yfinance"]["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="api-status api-failed">{api_status["yfinance"]["message"]}</div>', unsafe_allow_html=True)
            with col2:
                st.subheader("🔑 Alpha Vantage Status")
                if api_status['alpha_vantage']['working']:
                    st.markdown('<div class="api-status api-working">✅ Alpha Vantage is working</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="api-status api-failed">❌ Alpha Vantage error: {api_status["alpha_vantage"]["message"]}</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("""
            <div class="api-badge" style="background:linear-gradient(90deg,#4facfe 0%,#00f2fe 100%);
                color:#fff;padding:8px 18px;border-radius:25px;font-size:15px;font-weight:600;display:inline-block;
                box-shadow:0px 4px 10px rgba(0,0,0,0.2);">💎 Premium API Access Enabled</div>""",
            unsafe_allow_html=True)

        # Data source
        st.markdown("#### 📡 Data Source")
        data_source_choice = st.selectbox("Select Data Source",
            ["yfinance", "Alpha Vantage", "Auto (yfinance → Alpha Vantage → Sample)"], index=0)

        # Stock selection
        st.markdown("#### 📈 Stock Selection")
        market = st.selectbox("Select Market", ["US Stocks", "Indian Stocks", "Custom Ticker"])
        if market == "US Stocks":
            stock_options = RELIABLE_TICKERS["US Markets"]
            selected_stock = st.selectbox("Select Stock", list(stock_options.keys()))
            ticker = selected_stock
            st.info(f"📊 Selected: {stock_options[selected_stock]}")
        elif market == "Indian Stocks":
            stock_options = RELIABLE_TICKERS["Indian Markets"]
            selected_stock = st.selectbox("Select Stock", list(stock_options.keys()))
            ticker = selected_stock
            st.info(f"🇮🇳 Selected: {stock_options[selected_stock]}")
        else:
            ticker = st.text_input("Enter Stock Ticker", value="AAPL",
                                   help="Examples: AAPL (US), RELIANCE.NSE (Indian stocks with .NSE extension)")
            if ticker:
                st.info("🇮🇳 Indian stock format detected" if ticker.endswith(".NSE") else "🇺🇸 US stock format detected")

        # Time period
        st.markdown("#### 📅 Time Period")
        period = st.selectbox("Select Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)

        # New model controls
        st.markdown("### 🤖 Select Models for Forecasting")
        available = list(get_model_space().keys())
        if XGB_AVAILABLE:
            tooltip = "Auto will test all listed models with walk‑forward CV and pick the best by RMSE."
        else:
            tooltip = "Auto will test all built‑in models with walk‑forward CV and pick the best by RMSE."
        model_choices = st.multiselect("Models", ["Auto (Select Best)"] + available, default=["Auto (Select Best)"],
                                       help=tooltip)

        st.markdown("#### 🎯 Target Type")
        target_type = st.radio("What to predict?", ["Return (%)", "Price (level)"], index=0,
                               help="Return (%) is generally more stable across stocks.")
        st.session_state["target_type"] = "return" if target_type.startswith("Return") else "price"

        st.markdown("#### 🧪 Validation")
        cv_strategy = st.radio("CV Strategy", ["Walk‑forward (5 folds)", "Hold‑out (20%)"], index=0)
        do_tune = st.checkbox("Fast Hyperparameter Tuning", value=False)
        tune_iter = st.slider("Tuning Budget (iterations)", 5, 50, 20)

        # Prediction settings
        st.markdown("#### 🔮 Prediction Settings")
        prediction_days = st.slider("Days to Predict", 1, 30, 7)

        predict_button = st.button("🚀 Predict Stock Price", type="primary", use_container_width=True)

    # Action
    if predict_button:
        if not ticker:
            st.error("Please enter a stock ticker symbol!")
            return

        # Tabs layout
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Stock Analysis","🔮 Predictions","📈 Charts",
                                                      "🤖 Model Performance","📋 Data Table","🧩 Explainable AI"])

        # Load data
        with st.spinner(f"🔄 Fetching stock data from {data_source_choice}..."):
            if data_source_choice.startswith("yfinance"):
                df = fetch_stock_data_yfinance(ticker, period) if YFINANCE_AVAILABLE else None
                used_source = "yfinance" if df is not None else None
                if df is None:
                    st.warning("⚠️ yfinance failed, trying Alpha Vantage...")
                    df = fetch_stock_data_unified(ticker, period)
                    used_source = "alpha_vantage" if df is not None else None
            elif data_source_choice.startswith("Alpha Vantage"):
                df = fetch_stock_data_unified(ticker, period)
                used_source = "alpha_vantage" if df is not None else None
                if df is None and YFINANCE_AVAILABLE:
                    st.warning("⚠️ Alpha Vantage failed, trying yfinance...")
                    df = fetch_stock_data_yfinance(ticker, period)
                    used_source = "yfinance" if df is not None else None
            else:
                df, used_source, trace = load_stock_data_auto(ticker, period)
                st.markdown("#### 🔎 API Call Status")
                for src, msg in trace:
                    css_class = "api-working" if "✅" in msg else "api-failed"
                    st.markdown(f'<div class="api-status {css_class}">{msg}</div>', unsafe_allow_html=True)

        if df is None or df.empty:
            st.error("❌ Unable to fetch real data. Using sample data.")
            df = create_sample_data(ticker, period)
            used_source = "sample_data"

        # Process & diagnostics
        data_source = df.attrs.get('source', used_source)
        df = process_stock_data(df, ticker, data_source)
        if df is None or df.empty:
            st.error("❌ Unable to process stock data. Please try again.")
            return

        stock_info = get_stock_info(ticker)
        currency = stock_info.get('currency', 'USD')
        currency_symbol = '$' if currency == 'USD' else 'INR '

        current_price_val = float(df['Close'].iloc[-1]) if 'Close' in df.columns else None

        # Volatility (annualized)
        volatility = None
        if 'Close' in df.columns and len(df) > 2:
            r = df['Close'].pct_change().dropna()
            if not r.empty:
                volatility = r.std() * np.sqrt(252)

        # Diagnostics
        diag = data_diagnostics(df)

        # ---------------- Tab1: Stock Analysis ----------------
        with tab1:
            st.markdown(f"### 📋 {stock_info['name']} ({ticker})")
            if data_source != 'sample_data':
                st.info(f"📡 Data Source: {data_source.title()}")
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.metric("Current Price", f"{currency_symbol}{current_price_val:.2f}" if current_price_val else "—")
            with c2:
                if len(df) > 1:
                    price_change = float(df['Close'].iloc[-1] - df['Close'].iloc[-2])
                    pct = price_change/float(df['Close'].iloc[-2])*100.0 if float(df['Close'].iloc[-2])!=0 else 0.0
                else:
                    price_change, pct = 0.0, 0.0
                st.metric("Price Change", f"{currency_symbol}{price_change:.2f}", f"{pct:.2f}%")
            with c3:
                vol = int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else None
                st.metric("Volume", f"{vol:,.0f}" if vol else "—")
            with c4:
                st.metric("Volatility (annualized %)", f"{volatility*100:.2f}%" if volatility is not None else "—")

            st.markdown("### 🧰 Data Quality & Predictability")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.metric("Rows", f"{diag['rows']:,}")
                st.metric("Date Span", f"{diag['date_span_days']} days")
            with cc2:
                st.metric("Max Missing % (any column)", f"{diag['missing_max_pct']*100:.1f}%")
                st.metric("Return Variance", f"{diag['ret_var']:.6f}")
            with cc3:
                st.metric("Lag‑1 Autocorr (returns)", f"{diag['ret_autocorr_lag1']:.3f}")
                st.metric("Predictability Score", f"{diag['predictability_score']:.0f}/100")
            if diag['warnings']:
                st.warning(" • " + "\n • ".join(diag['warnings']))

            # Details
            st.markdown("### 📊 Stock Details")
            col1,col2 = st.columns(2)
            with col1:
                st.write(f"**Sector:** {stock_info['sector']}")
                st.write(f"**Industry:** {stock_info['industry']}")
            with col2:
                st.write(f"**Market Cap:** {stock_info['market_cap']}")
                st.write(f"**Currency:** {stock_info['currency']}")

            # Key stats
            st.markdown("### 📈 Key Statistics")
            k1,k2,k3,k4 = st.columns(4)
            with k1:
                st.metric("52W High", f"{currency_symbol}{float(df['High'].max()):.2f}" if not df.empty else "—")
            with k2:
                st.metric("52W Low", f"{currency_symbol}{float(df['Low'].min()):.2f}" if not df.empty else "—")
            with k3:
                st.metric("Avg Volume", f"{float(df['Volume'].mean()):,.0f}" if not df.empty else "—")
            with k4:
                if 'RSI' in df.columns and not df['RSI'].isna().all():
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

        # ---------------- Tab2: Predictions ----------------
        with tab2:
            st.markdown("### 🤖 AI Predictions")
            horizon = 1
            X, y, features = prepare_supervised(df, horizon=horizon, target_type=st.session_state["target_type"])
            if X.empty:
                st.error("Not enough data to prepare features.")
                return
            # Auto select or manual set
            nfolds = 5 if cv_strategy.startswith("Walk") else 3
            if "Auto" in model_choices or len([m for m in model_choices if m != "Auto (Select Best)"]) == 0:
                with st.spinner("Selecting best model via walk‑forward CV..."):
                    best_name, final_pipe, cv_table, best_rmse = auto_select_model(
                        X, y, selected_names=model_choices, n_splits=nfolds, do_tune=do_tune,
                        tune_iter=tune_iter, target_type=st.session_state["target_type"]
                    )
            else:
                name = model_choices[0]
                mdl = get_model_space()[name]
                final_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler()),
                                       ("m", mdl)])
                if cv_strategy.startswith("Walk"):
                    cv_scores = time_series_cv_score(mdl, X, y, n_splits=nfolds)
                    cv_table = pd.DataFrame([{"model": name, **cv_scores}])
                else:
                    cv_table = pd.DataFrame()
                best_name = name

            st.success(f"✅ Selected Model: **{best_name}**  |  Target: **{st.session_state['target_type']}**")
            if not cv_table.empty:
                st.markdown("#### 🧪 Cross‑Validation Summary (lower RMSE is better)")
                st.dataframe(cv_table, use_container_width=True)

            # Backtest plot (last 20% hold-out)
            bt_metrics, bt_df = backtest_holdout(final_pipe, X, y, test_size=0.2)
            st.markdown("#### 📉 Backtest on Recent Hold‑out")
            c1,c2,c3 = st.columns(3)
            c1.metric("RMSE", f"{bt_metrics['rmse']:.4f}")
            c2.metric("MAE", f"{bt_metrics['mae']:.4f}")
            c3.metric("R²", f"{bt_metrics['r2']:.3f}")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(y=bt_df['Actual'], mode="lines", name="Actual"))
            fig_bt.add_trace(go.Scatter(y=bt_df['Predicted'], mode="lines", name="Predicted"))
            fig_bt.update_layout(template="plotly_white", title="Backtest: Actual vs Predicted (hold‑out)",
                                 xaxis_title="Observations", yaxis_title="Target")
            st.plotly_chart(fig_bt, use_container_width=True)

            # One‑step ahead
            st.markdown("### 🔮 Next Day Prediction")
            # Use last row features
            X_all, _, _ = prepare_supervised(df, horizon=1, target_type=st.session_state["target_type"])
            last_row = X_all.iloc[[-1]]
            y_hat = float(final_pipe.predict(last_row)[0])
            if st.session_state["target_type"] == "return":
                current_price_num = float(df['Close'].iloc[-1])
                next_price = current_price_num * (1 + y_hat/100.0)
                delta = next_price - current_price_num
                pct = (delta/current_price_num)*100.0 if current_price_num!=0 else 0.0
                c1,c2,c3 = st.columns(3)
                c1.metric("Current Price", f"{currency_symbol}{current_price_num:.2f}")
                c2.metric("Predicted Return (1d)", f"{y_hat:.2f}%")
                c3.metric("Predicted Price (1d)", f"{currency_symbol}{next_price:.2f}", f"{currency_symbol}{delta:.2f}")
            else:
                current_price_num = float(df['Close'].iloc[-1])
                delta = y_hat - current_price_num
                pct = (delta/current_price_num)*100.0 if current_price_num!=0 else 0.0
                c1,c2,c3 = st.columns(3)
                c1.metric("Current Price", f"{currency_symbol}{current_price_num:.2f}")
                c2.metric("Predicted Price (1d)", f"{currency_symbol}{y_hat:.2f}", f"{currency_symbol}{delta:.2f}")
                c3.metric("Expected Change", f"{pct:.2f}%")

            # Multi‑day iterative forecast
            st.markdown("### 📈 Multi‑day Forecast")
            preds, fc_dates = iterative_forecast(df, final_pipe, days=prediction_days, target_type=st.session_state["target_type"])
            if preds:
                if st.session_state["target_type"] == "return":
                    # Convert cumulative returns to price path
                    price_path = [float(df['Close'].iloc[-1])]
                    for r in preds:
                        price_path.append(price_path[-1]*(1+r/100.0))
                    price_path = price_path[1:]
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=fc_dates, y=price_path, mode="lines+markers", name="Forecasted Price"))
                    fig_fc.update_layout(template="plotly_white", title="Forecasted Price Path", xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})")
                else:
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=fc_dates, y=preds, mode="lines+markers", name="Forecasted Price"))
                    fig_fc.update_layout(template="plotly_white", title="Forecasted Price Path", xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})")
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.info("Forecast horizon too short to compute.")

        # ---------------- Tab3: Charts (keep as before) ----------------
        with tab3:
            st.markdown("### 📈 Stock Price Charts")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price',
                                     line=dict(color='#1f77b4', width=3)))
            if 'MA_20' in df.columns and not df['MA_20'].isna().all():
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_20'], mode='lines', name='20-Day MA',
                                         line=dict(color='#ff7f0e', width=2, dash='dash')))
            if 'MA_50' in df.columns and not df['MA_50'].isna().all():
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_50'], mode='lines', name='50-Day MA',
                                         line=dict(color='#2ca02c', width=2, dash='dot')))
            fig.update_layout(title=f"{ticker} Stock Price with Moving Averages",
                              xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})",
                              hovermode='x unified', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='rgba(31, 119, 180, 0.6)'))
            fig_volume.update_layout(title=f"{ticker} Trading Volume", xaxis_title="Date", yaxis_title="Volume", template='plotly_white')
            st.plotly_chart(fig_volume, use_container_width=True)
            if 'RSI' in df.columns and not df['RSI'].isna().all():
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='#d62728', width=3)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff7f0e", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#2ca02c", annotation_text="Oversold (30)")
                fig_rsi.update_layout(title=f"{ticker} RSI (Relative Strength Index)", xaxis_title="Date", yaxis_title="RSI",
                                      yaxis=dict(range=[0, 100]), template='plotly_white')
                st.plotly_chart(fig_rsi, use_container_width=True)

        # ---------------- Tab4: Model Performance ----------------
        with tab4:
            st.markdown("### 🤖 Model Performance Details")
            st.info("Cross‑validation results are shown in the **Predictions** tab table. Below are hold‑out metrics:")
            c1,c2,c3 = st.columns(3)
            c1.metric("RMSE (hold‑out)", f"{bt_metrics['rmse']:.4f}")
            c2.metric("MAE (hold‑out)", f"{bt_metrics['mae']:.4f}")
            c3.metric("R² (hold‑out)", f"{bt_metrics['r2']:.3f}")
            st.markdown("### 🎯 Guidance")
            if bt_metrics['r2'] > 0.6:
                st.success("Excellent model performance – predictions are generally reliable.")
            elif bt_metrics['r2'] > 0.4:
                st.warning("Moderate performance – use predictions with caution.")
            else:
                st.error("Low performance – consider longer history, return target, or different data source.")
            st.markdown("### 📌 Why performance varies & fixes applied")
            st.write("""
            - **Proper target alignment**: We now predict the **next‑day** return or price, avoiding leakage.
            - **Return‑based modeling**: Default target is **Return (%)**, which is comparable across stocks.
            - **Walk‑forward CV**: Uses time‑aware folds for fair evaluation across regimes.
            - **Auto Model Selection**: Tests multiple algorithms & picks the best, with optional fast tuning.
            - **Iterative multi‑day**: Forecasts step‑by‑step, recomputing features at each step.
            - **Diagnostics**: Predictability score flags tickers with inherently poor short‑term signal.
            """)

        # ---------------- Tab5: Data Table ----------------
        with tab5:
            st.markdown("### 📋 Historical Data")
            display_df = df.tail(50).copy()
            if 'Date' in display_df.columns:
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for c in ['MA_20','RSI']:
                if c in display_df.columns: display_columns.append(c)
            st.dataframe(display_df[display_columns], use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button(label="📥 Download Data as CSV", data=csv,
                               file_name=f"{ticker}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv", type="primary")
            st.markdown("### 📊 Data Statistics")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**💰 Price Statistics:**")
                safe_stat(df, "High", np.max, "Highest Price", "{:.2f}", currency_symbol)
                safe_stat(df, "Low", np.min, "Lowest Price", "{:.2f}", currency_symbol)
                safe_stat(df, "Close", np.mean, "Average Price", "{:.2f}", currency_symbol)
                try:
                    if "High" in df.columns and "Low" in df.columns:
                        price_range = float(df["High"].max()) - float(df["Low"].min())
                        st.write(f"- Price Range: {currency_symbol}{price_range:.2f}")
                    else:
                        st.write("- Price Range: Data not available")
                except Exception:
                    st.write("- Price Range: Data not available")
            with c2:
                st.markdown("**📊 Trading Statistics:**")
                safe_stat(df, "Volume", np.mean, "Average Volume", "{:,.0f}")
                safe_stat(df, "Volume", np.max, "Max Volume", "{:,.0f}")
                st.write(f"- Total Data Points: {len(df):,}" if df is not None else "- Total Data Points: Data not available")
                try:
                    date_min, date_max = df['Date'].min(), df['Date'].max()
                    st.write(f"- Date Range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
                except Exception:
                    st.write("- Date Range: Data not available")
                if volatility is not None:
                    st.write(f"- Volatility (annualized): {volatility*100:.2f}%")
                else:
                    st.write("- Volatility: Data not available")

        # ---------------- Tab6: Explainable AI ----------------
        with tab6:
            render_explainable_ai_tab(final_pipe, df)

        st.markdown("""
        <div class="warning-card">
            <strong>⚠️ Important Disclaimer:</strong><br>
            This application is designed for educational and research purposes only.
            Stock price predictions are inherently uncertain and should never be used as the sole basis for investment decisions.
            <br><br>
            <strong>🔍 Please Note:</strong>
            <ul>
                <li>Past performance does not guarantee future results</li>
                <li>Market conditions can change rapidly and unpredictably</li>
                <li>Always consult with qualified financial advisors</li>
                <li>Conduct your own thorough research before making investment decisions</li>
                <li>Only invest what you can afford to lose</li>
            </ul>
            <br>
            <strong>📊 Data Sources:</strong> This application utilizes multiple data sources including Alpha Vantage & yfinance,
            and may fall back to sample data when live APIs are unavailable.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <h2 style='text-align:center;font-size:40px;font-weight:800;
            background:-webkit-linear-gradient(45deg,#4facfe,#00f2fe);-webkit-background-clip:text;
            -webkit-text-fill-color:transparent;margin-bottom:20px;'>🧠 Cortex-o1 Predictive Model</h2>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                ### ✨ Premium Features:
                - 🔄 **Multi-API Integration**: Seamless data fetching from Alpha Vantage & yfinance
                - 🤖 **Auto Model Selection** with walk‑forward CV
                - 📊 **Technical Indicators** & enhanced features (returns, volatility, momentum)
                - 🎯 **Return or Price Targets** with proper next‑day alignment
                - 📈 **Backtests & Forecasts**: Hold‑out plot and multi‑day iterative forecasting
                - 🧪 **Diagnostics**: Predictability score and data quality warnings
            """)
        with col2:
            st.markdown("""
                ### 🎯 How It Works:
                1. Choose your **stock** and **period**
                2. Select **target** (Return or Price) and **CV strategy**
                3. Use **Auto (Select Best)** or pick a model
                4. Generate **predictions** and review **backtests**
                5. Inspect **Explainable AI** to understand drivers
            """)
        st.markdown("---")
        st.markdown("👈 Use the **sidebar** to configure your settings and begin exploring the power of **AI‑driven stock prediction!**")

if __name__ == "__main__":
    main()

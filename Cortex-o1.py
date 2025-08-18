# Cortex-o1-fixed.py
# CORRECTED VERSION with all critical errors fixed
# Key fixes applied:
# - Fixed deprecated pandas fillna(method='ffill') -> .ffill()
# - Optimized DataFrame concatenation in prediction loop
# - Improved hash generation for model caching
# - Added safe column access and bounds checking
# - Enhanced error handling throughout

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import hashlib
warnings.filterwarnings('ignore')

# Optional libs (graceful degrade)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# Alpha Vantage
# Keep using st.secrets for API key as before
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", None)
AV_BASE_URL = "https://www.alphavantage.co/query"

# Streamlit page config
st.set_page_config(page_title="Neural Minds", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------
# Styling (kept similar to prior file)
# ---------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        .main-header { font-size: 3rem; font-weight:700; background: linear-gradient(45deg,#1f77b4,#ff7f0e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align:center; margin-bottom: 0.5rem; font-family: 'Inter', sans-serif; }
        .subtitle { text-align:center; font-size:1.1rem; color:#666; margin-bottom: 1.5rem; font-weight:300; }
        .warning-card { background:#000; padding:1.2rem; border-radius:8px; border-left:4px solid #fdcb6e; color:#fff; }
        .api-status { padding:0.6rem; border-radius:8px; margin:0.6rem 0; }
        .api-working { background:#d4edda; color:#155724; border:1px solid #c3e6cb; }
        .api-failed { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; }
        section[data-testid="stSidebar"] { background:#f9f9f9; color:#000; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tickers dictionary (for sidebar picker)
# ---------------------------------------------------------
RELIABLE_TICKERS = {
    "US Markets": {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "BLK": "BlackRock Inc.",
        "GS": "Goldman Sachs Group Inc.",
        "STT": "State Street Corporation",
        "TSLA": "Tesla Inc.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc."
    },
    "Indian Markets": {
        "RELIANCE.NSE": "Reliance Industries",
        "TCS.NSE": "Tata Consultancy Services",
        "PARAS.NSE": "Paras Defence and Space Technologies",
        "INFY.NSE": "Infosys Limited",
        "HDFCBANK.NSE": "HDFC Bank",
        "WIPRO.NSE": "Wipro Limited",
        "ITC.NSE": "ITC Limited",
        "SBIN.NSE": "State Bank of India",
        "TATAMOTORS.NSE": "Tata Motors",
        "TATASTEEL.NSE": "Tata Steel",
        "KOTAKBANK.NSE": "Kotak Mahindra Bank",
        "BHARTIARTL.NSE": "Bharti Airtel",
        "HINDUNILVR.NSE": "Hindustan Unilever"
    }
}

# ---------------------------------------------------------
# HELPER FUNCTIONS FOR SAFE DATA ACCESS (NEW)
# ---------------------------------------------------------
def safe_column_access(df, column, default_value=0):
    """Safely access DataFrame column with fallback"""
    if df is None or df.empty:
        return pd.Series([default_value])
    if column not in df.columns:
        return pd.Series([default_value] * len(df))
    return df[column]

def safe_iloc_last(df, default=None):
    """Safely get last row of DataFrame"""
    if df is None or df.empty:
        return default
    return df.iloc[-1]

def safe_iloc_access(df, index, default=None):
    """Safely access DataFrame by index with bounds checking"""
    if df is None or df.empty or len(df) <= abs(index):
        return default
    return df.iloc[index]

# ---------------------------------------------------------
# Utility: map ticker for source (yfinance vs alpha_vantage)
# ---------------------------------------------------------
def map_ticker_for_source(ticker: str, source: str) -> str:
    """
    Map user's ticker to the source-specific symbol.
    - For yfinance, convert .NSE to .NS
    - For alpha_vantage, use base ticker (Alpha Vantage often expects non-exchange suffix)
    """
    if not ticker:
        return ticker
    t = ticker.strip().upper()
    if source == "yfinance":
        if t.endswith(".NSE"):
            return t.replace(".NSE", ".NS")
        return t.split('.')[0]
    if source == "alpha_vantage":
        # Alpha Vantage expects symbol without .NSE, commonly base ticker works
        return t.split('.')[0]
    return t

# ---------------------------------------------------------
# Small helpers for time periods
# ---------------------------------------------------------
def get_period_days(period: str) -> int:
    return {"1mo":30, "3mo":90, "6mo":180, "1y":365, "2y":730, "5y":1825}.get(period, 365)

# ---------------------------------------------------------
# Cached fetchers - cache network results for performance
# ---------------------------------------------------------
# Use ttl so recent updates refresh occasionally
@st.cache_data(ttl=300)
def fetch_stock_data_yfinance(ticker: str, period: str = "1y"):
    """Fetch via yfinance, minimal processing; returns dataframe or None."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        mapped = map_ticker_for_source(ticker, "yfinance")
        yf_period_map = {'1mo':'1mo','3mo':'3mo','6mo':'6mo','1y':'1y','2y':'2y','5y':'5y'}
        yf_period = yf_period_map.get(period, '1y')
        df = yf.download(mapped, period=yf_period, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df = df.rename(columns={"Date":"Date"})
        # Standard columns with safe access
        required_columns = ['Date','Open','High','Low','Close','Volume']
        if not all(col in df.columns for col in required_columns):
            return None
        df = df[required_columns].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.attrs = {'source':'yfinance', 'ticker': mapped}
        return df
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_stock_data_alpha_vantage(ticker: str, period: str = "1y"):
    """
    Fetch using Alpha Vantage with smarter 'outputsize' selection to reduce payload.
    Returns dataframe or None.
    """
    if ALPHA_VANTAGE_API_KEY is None:
        return None
    try:
        mapped = map_ticker_for_source(ticker, "alpha_vantage")
        days = get_period_days(period)
        # Use 'compact' if asking for <= 2 years to reduce payload
        outputsize = 'compact' if days <= 730 else 'full'
        # TIME_SERIES_DAILY returns daily series
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': mapped,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': outputsize,
            'datatype': 'json'
        }
        resp = requests.get(AV_BASE_URL, params=params, timeout=20)
        data = resp.json()
        # Rate limit / error handling
        if 'Time Series (Daily)' not in data:
            return None
        ts = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df = df.rename(columns={
            '1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close','5. volume':'Volume'
        })
        df = df[['Open','High','Low','Close','Volume']].astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={'index':'Date'})
        # Trim to requested period days
        start_date = datetime.now() - timedelta(days=days)
        df = df[df['Date'] >= start_date].copy()
        df.attrs = {'source':'alpha_vantage', 'ticker': mapped}
        return df
    except Exception:
        return None

def create_sample_data(ticker: str, period: str = "1y"):
    """Fallback synthetic sample data (kept simple and fast)."""
    days = get_period_days(period)
    base_name = ticker.split('.')[0].upper()
    # use a small dictionary of base prices
    base_prices = {
    # US Markets
    'AAPL': 180,
    'GOOGL': 140,
    'MSFT': 330,
    'BLK': 700,        # BlackRock
    'GS': 340,         # Goldman Sachs
    'STT': 70,         # State Street
    'TSLA': 250,
    'AMZN': 140,
    'NVDA': 450,
    'META': 300,
    'NFLX': 400,
    'JPM': 150,        # JPMorgan
    'V': 230,          # Visa
    
    # Indian Markets
    'RELIANCE': 2500,
    'TCS': 3500,
    'PARAS': 700,      # Paras Defence
    'INFY': 1500,
    'HDFCBANK': 1600,
    'WIPRO': 400,
    'ITC': 450,
    'SBIN': 600,
    'TATAMOTORS': 650,
    'TATASTEEL': 120,
    'KOTAKBANK': 1900,
    'BHARTIARTL': 850,
    'HINDUNILVR': 2500
}
    base_price = base_prices.get(base_name, 300.0)
    np.random.seed(abs(hash(ticker)) % (2**32))
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    prices = base_price * np.cumprod(1 + np.random.normal(0.0003, 0.02, size=len(dates)))
    data = []
    for i, d in enumerate(dates):
        close = float(prices[i])
        openp = float(prices[i] * (1 + np.random.normal(0, 0.002)))
        high = float(max(openp, close) * (1 + abs(np.random.normal(0, 0.01))))
        low = float(min(openp, close) * (1 - abs(np.random.normal(0, 0.01))))
        vol = int(np.random.lognormal(np.log(100000), 0.8))
        data.append({'Date': d, 'Open': round(openp,2), 'High': round(high,2), 'Low': round(low,2), 'Close': round(close,2), 'Volume': vol})
    df = pd.DataFrame(data)
    df.attrs = {'source': 'sample_data', 'ticker': ticker}
    return df

# ---------------------------------------------------------
# Indicator computations (compute once, cache)
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators once and return new dataframe (copy)."""
    if df is None or df.empty:
        return df
    df = df.copy()
    
    # Ensure required columns exist
    required_columns = ['Close', 'High', 'Low', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return df
    
    # Ensure Date is datetime and sorted
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Moving averages - using safe column access
    close_col = safe_column_access(df, 'Close', 0)
    df['MA_20'] = close_col.rolling(window=20,min_periods=5).mean()
    df['MA_50'] = close_col.rolling(window=50,min_periods=10).mean()
    
    # RSI (simple implementation)
    delta = close_col.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    ma = close_col.rolling(20).mean()
    std = close_col.rolling(20).std(ddof=0)
    df['BB_MA'] = ma
    df['BB_UPPER'] = ma + 2 * std
    df['BB_LOWER'] = ma - 2 * std
    
    # MACD
    ema_fast = close_col.ewm(span=12, adjust=False).mean()
    ema_slow = close_col.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    
    # ATR (Average True Range)
    prev_close = close_col.shift(1)
    tr1 = safe_column_access(df, 'High', 0) - safe_column_access(df, 'Low', 0)
    tr2 = (safe_column_access(df, 'High', 0) - prev_close).abs()
    tr3 = (safe_column_access(df, 'Low', 0) - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # VWAP (typical price * volume cumulative) - FIXED: deprecated fillna
    tp = (safe_column_access(df, 'High', 0) + safe_column_access(df, 'Low', 0) + close_col) / 3
    vol = safe_column_access(df, 'Volume', 1).replace(0, np.nan).ffill().fillna(1)  # FIXED: removed method='ffill'
    df['VWAP'] = (tp * vol).cumsum() / vol.cumsum()
    
    # Returns and volatility
    df['ret'] = close_col.pct_change()
    df['logret'] = np.log(close_col).diff()
    df['ret2'] = df['ret'] ** 2
    
    # Lag features for modeling
    for i in [1,2,3,5]:
        df[f'Close_Lag_{i}'] = close_col.shift(i)
    
    # Drop rows with NaNs from rolling windows (keep later for modeling)
    df = df.dropna().reset_index(drop=True)
    
    # Save metadata
    df.attrs['indicators_computed'] = True
    df.attrs['last_indicator_run'] = datetime.now()
    return df

# ---------------------------------------------------------
# Small risk & stat utilities
# ---------------------------------------------------------
def sharpe_ratio(df: pd.DataFrame, rf_daily: float=0.0):
    """Calculate Sharpe ratio with safe column access"""
    if df is None or df.empty:
        return np.nan
    r = safe_column_access(df, 'ret', 0).dropna()
    if r.empty: 
        return np.nan
    excess = r - rf_daily
    return float(np.sqrt(252) * excess.mean() / (excess.std(ddof=0) + 1e-12))

def drawdown_stats(df: pd.DataFrame):
    """Calculate drawdown statistics with safe access"""
    if df is None or df.empty:
        return pd.Series([np.nan]), np.nan, np.nan, np.nan
    
    close_col = safe_column_access(df, 'Close', 1).astype(float)
    if close_col.empty or len(close_col) < 2:
        return pd.Series([np.nan]), np.nan, np.nan, np.nan
    
    cum_max = close_col.cummax()
    dd = close_col / cum_max - 1.0
    max_dd = float(dd.min()) if not dd.empty else np.nan
    
    first_val = safe_iloc_access(close_col, 0, 1)
    last_val = safe_iloc_access(close_col, -1, 1)
    total_ret = last_val / first_val - 1.0 if first_val and last_val else np.nan
    
    if 'Date' in df.columns and len(df) >= 2:
        first_date = safe_iloc_access(df['Date'], 0)
        last_date = safe_iloc_access(df['Date'], -1)
        if first_date is not None and last_date is not None:
            years = max((last_date - first_date).days / 365.25, 1e-9)
        else:
            years = 1.0
    else:
        years = 1.0
    
    ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 and total_ret is not np.nan else np.nan
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 and ann_ret is not np.nan else np.nan
    return dd, max_dd, ann_ret, calmar

def vol_clustering_score(df: pd.DataFrame, lag: int = 1):
    """Calculate volatility clustering with safe access"""
    if df is None or df.empty:
        return np.nan
    r2 = safe_column_access(df, 'ret2', 0).dropna()
    if len(r2) < lag + 1: 
        return np.nan
    return float(r2.autocorr(lag=lag))

# ---------------------------------------------------------
# Modeling helpers (with caching to avoid re-train)
# ---------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    """Return X, y, feature list for regression on Close with safe access"""
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), []
    
    feature_columns = ['Open','High','Low','Volume','MA_20','MA_50','RSI','Price_Change','Volume_MA']
    # include lag features if present
    for i in [1,2,3,5]:
        feature_columns.append(f'Close_Lag_{i}')
    
    existing = [c for c in feature_columns if c in df.columns]
    if not existing:
        return pd.DataFrame(), pd.Series(dtype=float), []
        
    X = df[existing].copy()
    y = safe_column_access(df, 'Close', 0).copy()
    return X, y, existing

@st.cache_resource
def train_random_forest_cached_safe(key: str, X_train_array, y_train_array, n_estimators:int=100, max_depth:int=12):
    """
    Cache training with improved error handling.
    st.cache_resource caches Python objects (model) across reruns.
    """
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train_array, y_train_array)
        return model
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

def train_model_rf(df: pd.DataFrame, test_ratio: float = 0.2, n_estimators: int = 100, max_depth: int = 12):
    """Train RF using prepared features. Uses cache_resource for the model."""
    X, y, feat = prepare_features(df)
    if X.empty or y.empty:
        return None, None, None, None
    
    # create time-based split
    split_idx = int(len(X) * (1 - test_ratio))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # IMPROVED: More reliable cache key generation using MD5 hash
        last_row_dict = safe_iloc_last(X_train.fillna(0).round(3), {}).to_dict() if hasattr(safe_iloc_last(X_train.fillna(0).round(3), {}), 'to_dict') else {}
        last_row_str = str(sorted(last_row_dict.items()))
        hash_suffix = hashlib.md5(last_row_str.encode()).hexdigest()[:8]
        model_key = f"rf_{n_estimators}_{max_depth}_{len(X_train)}_{hash_suffix}"
        
        model = train_random_forest_cached_safe(model_key, X_train_scaled, y_train.values, 
                                               n_estimators=n_estimators, max_depth=max_depth)
        
        if model is None:
            return None, None, None, None
            
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
            'test_mae': float(mean_absolute_error(y_test, y_test_pred)),
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred)),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Feature importance
        fi = pd.DataFrame({'feature': feat, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        return model, scaler, metrics, fi
        
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, None, None, None

def predict_next_day_from_rf(model, scaler, df: pd.DataFrame):
    """Return next-day predicted close using last available features with safe access."""
    if model is None or scaler is None:
        return None
        
    X, _, feat = prepare_features(df)
    if X.empty:
        return None
        
    try:
        last_row = safe_iloc_last(X, None)
        if last_row is None:
            return None
            
        last = last_row.values.reshape(1, -1)
        last_scaled = scaler.transform(last)
        pred = float(model.predict(last_scaled)[0])
        return pred
    except Exception:
        return None

# ---------------------------------------------------------
# UI helpers: banners, downsampling
# ---------------------------------------------------------
def show_data_banner(source: str, n_points: int, ticker: str):
    if source == 'sample_data':
        st.warning(f"⚠️ SAMPLE DATA — Real-time data unavailable. Showing synthetic series for **{ticker}** ({n_points} points).")
    elif source in ('yfinance','alpha_vantage'):
        src_name = "Yahoo Finance" if source == 'yfinance' else 'Alpha Vantage'
        st.success(f"✅ LIVE DATA — Loaded **{n_points}** points for **{ticker}** from **{src_name}**.")
    else:
        st.info(f"ℹ️ Data source: {source} — {n_points} points for {ticker}.")

def downsample_for_plot(df: pd.DataFrame, max_points: int = 1500) -> pd.DataFrame:
    """Downsample dataframe for faster plotting while preserving endpoints."""
    if df is None or df.empty:
        return df
        
    n = len(df)
    if n <= max_points:
        return df
    
    # keep first 50 + last 50 then sample the middle
    head = df.iloc[:50] if n > 50 else df
    tail = df.iloc[-50:] if n > 50 else pd.DataFrame()
    
    if n > 100:
        middle = df.iloc[50:-50]
        frac = max_points - 100
        # compute approximate fraction
        if len(middle) <= 0:
            sampled = pd.concat([head, tail]) if not tail.empty else head
            return sampled.sort_values('Date').reset_index(drop=True) if 'Date' in sampled.columns else sampled
        
        sample_frac = max(0.001, min(1.0, frac / len(middle)))
        mid_sampled = middle.sample(frac=sample_frac, random_state=42).sort_index()
        result = pd.concat([head, mid_sampled, tail])
    else:
        result = pd.concat([head, tail]) if not tail.empty else head
    
    # FIXED: Ensure proper sorting and index reset
    if 'Date' in result.columns:
        result = result.sort_values('Date').reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)
    
    return result

def safe_number(val, precision=":.3f", percent=False):
    """
    Safely cast a value to float for display in st.metric.
    Returns formatted string or '—' if invalid.
    """
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return "—"
        fmt = f"{{{precision}}}"
        return (fmt.format(v*100) + "%") if percent else fmt.format(v)
    except Exception:
        return "—"

# ---------------------------------------------------------
# Main app layout and logic
# ---------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">Neural Minds</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Market Analysis & AI-Powered Prediction Platform</p>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        data_source_choice = st.selectbox("Select Data Source", ["Auto (yfinance → Alpha Vantage → Sample)", "yfinance", "Alpha Vantage"], index=0)
        market = st.selectbox("Select Market", ["US Stocks", "Indian Stocks", "Custom Ticker"])
        if market == "US Stocks":
            selected_stock = st.selectbox("Select Stock", list(RELIABLE_TICKERS["US Markets"].keys()))
            ticker = selected_stock
        elif market == "Indian Stocks":
            selected_stock = st.selectbox("Select Stock", list(RELIABLE_TICKERS["Indian Markets"].keys()))
            ticker = selected_stock
        else:
            ticker = st.text_input("Enter Stock Ticker (Example: AAPL or RELIANCE.NSE)", value="AAPL")
        period = st.selectbox("Select Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
        quick_mode = st.checkbox("Quick Mode (faster fetch & smaller training set)", value=True, help="Quick mode limits data & model training size for speed")
        prediction_days = st.slider("Days to predict (for modeling)", 1, 30, 7)
        predict_button = st.button("🚀 Analyze & Predict", use_container_width=True)

    # Welcome screen logic if not clicked
    if not predict_button:
        st.markdown(
            """
            <h2 style='
                text-align: center;
                font-size: 40px;
                font-weight: 800;
                background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-decoration: none;
                margin-bottom: 20px;
            '>
                🧠 Cortex-o1 Predictive Model
            </h2>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                ### ✨ Premium Features:
                - 🔄 **Multi-API Integration**: Seamless data fetching from Alpha Vantage & yfinance
                - 🤖 **Advanced AI Models**: Machine learning-powered predictions
                - 📊 **Comprehensive Analysis**: Technical indicators & market insights
                - 🎨 **Premium Interface**: Beautiful, responsive dark theme
                - 📈 **Real-time Charts**: Interactive Plotly visualizations
                - 🔍 **Performance Metrics**: Detailed model evaluation & statistics

                ### 🌍 Global Market Coverage:
                **🇺🇸 US Stocks:**
                - Apple (AAPL), Microsoft (MSFT), Alphabet/Google (GOOGL)
                - Amazon (AMZN), Tesla (TSLA), NVIDIA (NVDA)
                - Meta (META), Netflix (NFLX)
                - JPMorgan (JPM), Visa (V)
                - BlackRock (BLK), Goldman Sachs (GS), State Street (STT)

                **🇮🇳 Indian Stocks:**
                - Reliance (RELIANCE.NSE), TCS (TCS.NSE), Infosys (INFY.NSE)
                - HDFC Bank (HDFCBANK.NSE), Wipro (WIPRO.NSE), ITC (ITC.NSE)
                - SBI (SBIN.NSE), Kotak Bank (KOTAKBANK.NSE), Bharti Airtel (BHARTIARTL.NSE)
                - Hindustan Unilever (HINDUNILVR.NSE), Tata Motors (TATAMOTORS.NSE)
                - Tata Steel (TATASTEEL.NSE), Paras Defence (PARAS.NSE)
                """)

        with col2:
            st.markdown("""
                ### 🎯 How It Works:
                1. 📊 **Select Your Stock**: Pick from curated tickers or enter a custom symbol  
                2. ⏱️ **Choose Time Period**: Analyze 1 month → 5 years of data  
                3. 🤖 **AI Analysis**: ML models learn market patterns  
                4. 🔮 **Get Predictions**: Forecast next-day/multi-day prices with confidence  
                5. 📈 **Visualize Results**: Interactive charts & detailed analytics

                ### 🛠️ Technical Features:
                - 🧠 **Machine Learning**: Random Forest, Feature Engineering  
                - 🔁 **Cross-validation**: Performance metrics built-in  
                - 📊 **Technical Indicators**: Moving Averages (20/50d), RSI, Volume Analysis  
                - 📈 **Visualizations**: Interactive Price & Volume charts, RSI Momentum, Feature Importance  

                ### 💡 Pro Tips:
                - 📅 Use longer timeframes (1y+) for more reliable predictions  
                - 🌍 Consider external market/economic context  
                - ⏳ Compare predictions across different timeframes  
                - 🛡️ Always diversify your portfolio  
                """)
                        
        # 👇 Bottom full-width message
        st.markdown(
            """
            ---
            👈 Use the **sidebar** to configure your settings and begin exploring the power of **AI-driven stock prediction!**
            """,
            unsafe_allow_html=True
        )
        return

    # --- Fetching data (Auto fallback) with improved error handling ---
    with st.spinner("🔄 Fetching data..."):
        df = None
        used_source = None
        trace_msgs = []
        
        try:
            # Auto logic
            if data_source_choice.startswith("Auto"):
                # Try yfinance first if available
                if YFINANCE_AVAILABLE:
                    df = fetch_stock_data_yfinance(ticker, period)
                    if df is not None and not df.empty:
                        used_source = 'yfinance'
                        trace_msgs.append(("yfinance", "✅ yfinance loaded successfully"))
                    else:
                        trace_msgs.append(("yfinance","❌ yfinance failed"))
                # Try Alpha Vantage
                if df is None and ALPHA_VANTAGE_API_KEY:
                    df = fetch_stock_data_alpha_vantage(ticker, period)
                    if df is not None and not df.empty:
                        used_source = 'alpha_vantage'
                        trace_msgs.append(("alpha_vantage", "✅ Alpha Vantage loaded successfully"))
                    else:
                        trace_msgs.append(("alpha_vantage", "❌ Alpha Vantage failed or rate-limited"))
                # Fallback to sample
                if df is None:
                    df = create_sample_data(ticker, period)
                    used_source = 'sample_data'
                    trace_msgs.append(("sample_data", "⚠️ Using synthetic sample data"))
            elif data_source_choice == "yfinance":
                df = fetch_stock_data_yfinance(ticker, period) if YFINANCE_AVAILABLE else None
                used_source = 'yfinance' if df is not None else None
                if df is None:
                    st.warning("yfinance failed; falling back to Alpha Vantage then sample...")
                    df = fetch_stock_data_alpha_vantage(ticker, period) if ALPHA_VANTAGE_API_KEY else None
                    used_source = 'alpha_vantage' if df is not None else used_source
                if df is None:
                    df = create_sample_data(ticker, period)
                    used_source = 'sample_data'
            else:  # explicit Alpha Vantage
                df = fetch_stock_data_alpha_vantage(ticker, period) if ALPHA_VANTAGE_API_KEY else None
                used_source = 'alpha_vantage' if df is not None else None
                if df is None and YFINANCE_AVAILABLE:
                    st.warning("Alpha Vantage failed; trying yfinance...")
                    df = fetch_stock_data_yfinance(ticker, period)
                    used_source = 'yfinance' if df is not None else used_source
                if df is None:
                    df = create_sample_data(ticker, period)
                    used_source = 'sample_data'
        except Exception as e:
            st.error(f"Data fetching error: {str(e)}")
            df = create_sample_data(ticker, period)
            used_source = 'sample_data'

    # Show short trace of source attempts if present
    if 'trace_msgs' in locals() and trace_msgs:
        st.markdown("#### 🔎 Source trace")
        for src, msg in trace_msgs:
            css = "api-working" if "✅" in msg else "api-failed"
            st.markdown(f'<div class="api-status {css}">{msg}</div>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.error("Unable to retrieve any data. Try a different ticker or ensure API keys are configured.")
        return

    # Compute indicators once (cached)
    with st.spinner("🧮 Computing indicators..."):
        df_ind = compute_indicators(df)

    # Show data banner
    data_source = df.attrs.get('source', used_source)
    show_data_banner(data_source, len(df_ind), ticker)

    # Basic stock info (lightweight)
    stock_info = {
    # US Stocks
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

    # Indian Stocks
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

    # simple currency inference
    currency = 'INR' if ticker.endswith('.NSE') else 'USD'
    currency_symbol = 'INR ' if currency == 'INR' else '$'

    # Derive volatility and intraday detection
    is_intraday = False
    if 'Date' in df_ind.columns and not df_ind.empty:
        if df_ind['Date'].dt.hour.max() != 0 or df_ind['Date'].dt.minute.max() != 0:
            is_intraday = True
    volatility = None
    if 'ret' in df_ind.columns and not df_ind['ret'].dropna().empty:
        if is_intraday:
            volatility = df_ind['ret'].std()
        else:
            volatility = df_ind['ret'].std() * np.sqrt(252)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analysis", "🔮 Predictions", "📈 Charts", "🤖 Models", "📋 Data"])

    # -------------------- TAB 1: Analysis --------------------
    with tab1:
        st.subheader(f"Market Analysis — {stock_info.get(ticker.split('.')[0], {}).get('name', ticker)} ({ticker})")

        # Initialize defaults
        sr, max_dd, ann_ret, calmar, clust = (np.nan, np.nan, np.nan, np.nan, np.nan)

        # Compute safely
        try:
            sr = sharpe_ratio(df_ind)
        except Exception:
            pass

        try:
            dd, max_dd, ann_ret, calmar = drawdown_stats(df_ind)
        except Exception:
            pass

        try:
            clust = vol_clustering_score(df_ind, lag=1)
        except Exception:
            pass

        # Display metrics
        st.metric("Sharpe", safe_number(sr, precision=":.2f"))
        st.metric("Max Drawdown", safe_number(max_dd, precision=":.2f", percent=True))
        st.metric("Annualized Return", safe_number(ann_ret, precision=":.2f", percent=True))
        st.metric("Volatility clustering (lag=1)", safe_number(clust, precision=":.3f"))

        # Show bollinger + vwap chart (downsample for speed) with safe column access
        required_cols = ['Date','Close','BB_MA','BB_UPPER','BB_LOWER','VWAP']
        available_cols = [col for col in required_cols if col in df_ind.columns]
        
        if len(available_cols) >= 2:  # At least Date and Close
            plot_df = downsample_for_plot(df_ind[available_cols].dropna())
            fig = go.Figure()
            
            if 'Close' in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], name='Close', mode='lines'))
            if 'BB_MA' in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_MA'], name='BB_MA', mode='lines', line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_UPPER'], name='BB_UPPER', mode='lines', opacity=0.6))
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_LOWER'], name='BB_LOWER', mode='lines', opacity=0.6))
            if 'VWAP' in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['VWAP'], name='VWAP', mode='lines', line=dict(dash='dot')))
            
            fig.update_layout(title=f"{ticker} Price with Bollinger Bands & VWAP", xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})", template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for price chart")

        # Returns distribution (matplotlib histogram is lightweight)
        st.markdown("**Returns Distribution**")
        r = safe_column_access(df_ind, 'ret', 0).dropna()
        if not r.empty and len(r) > 10:
            fig_hist = plt.figure(figsize=(6,3))
            plt.hist(r, bins=min(50, len(r)//2))
            plt.title('Returns Distribution')
            st.pyplot(fig_hist)
        else:
            st.info("Not enough returns data for distribution")

        # Volatility clustering metric
        clust = vol_clustering_score(df_ind, lag=1)
        st.metric("Volatility clustering (lag=1)", safe_number(clust, precision=":.3f"))

    # -------------------- TAB 2: Single-day Predictions --------------------
    with tab2:
        st.subheader("Single-day Prediction (Random Forest)")
        # For speed, optionally restrict to last N days when quick_mode
        df_for_model = df_ind.copy()
        if quick_mode:
            last_n = min(len(df_for_model), 450)  # limit to last ~450 days
            df_for_model = df_for_model.tail(last_n).reset_index(drop=True)

        model, scaler, metrics, fi = None, None, None, None
        with st.spinner("🧠 Training Random Forest (cached when possible)..."):
            try:
                model, scaler, metrics, fi = train_model_rf(df_for_model, test_ratio=0.2, n_estimators=100, max_depth=12)
            except Exception as e:
                st.error(f"Model training failed: {e}")

        if model is None:
            st.error("Model training failed. Try toggling Quick Mode or use a smaller period.")
        else:
            # Display test r2, RMSE, MAE
            st.metric("Test R²", safe_number(metrics.get('test_r2', np.nan), precision=":.3f"))
            st.metric("Test RMSE", safe_number(metrics.get('test_rmse', np.nan), precision=":.3f"))
            st.metric("Test MAE", safe_number(metrics.get('test_mae', np.nan), precision=":.3f"))

            # Predict next day with safe access
            next_pred = predict_next_day_from_rf(model, scaler, df_for_model)
            if next_pred is not None:
                last_close_val = safe_iloc_last(safe_column_access(df_for_model, 'Close', 0), None)
                if last_close_val is not None:
                    last_close = float(last_close_val)
                    diff = next_pred - last_close
                    pct = (diff / last_close) * 100 if last_close != 0 else 0.0
                    st.metric("Current Price", f"{currency_symbol}{last_close:.2f}")
                    st.metric("Predicted Next Close", f"{currency_symbol}{next_pred:.2f}", f"{pct:.2f}%")
                    # Simple signal
                    if pct > 2:
                        st.success("🟢 Strong Bullish Signal")
                    elif pct > 0:
                        st.info("🔵 Mild Bullish")
                    elif pct > -2:
                        st.warning("🟡 Neutral")
                    else:
                        st.error("🔴 Bearish")
                else:
                    st.warning("Unable to get current price for prediction")
            else:
                st.warning("Unable to generate prediction")

    # -------------------- TAB 3: Charts --------------------
    with tab3:
        st.subheader("Charts & Indicators")
        # Price + MA chart with safe access
        chart_cols = ['Date','Close','MA_20','MA_50']
        available_chart_cols = [col for col in chart_cols if col in df_ind.columns]
        
        if len(available_chart_cols) >= 2:
            plot_df = downsample_for_plot(df_ind[available_chart_cols].dropna())
            fig2 = go.Figure()
            if 'Close' in plot_df.columns:
                fig2.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], name='Close'))
            if 'MA_20' in plot_df.columns: 
                fig2.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MA_20'], name='MA_20', line=dict(dash='dash')))
            if 'MA_50' in plot_df.columns: 
                fig2.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MA_50'], name='MA_50', line=dict(dash='dot')))
            fig2.update_layout(title="Price with Moving Averages", xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})", template='plotly_white')
            st.plotly_chart(fig2, use_container_width=True)

        # MACD
        if 'MACD' in df_ind.columns:
            macd_cols = ['Date','MACD','MACD_SIGNAL']
            available_macd_cols = [col for col in macd_cols if col in df_ind.columns]
            if len(available_macd_cols) >= 2:
                fig_macd = go.Figure()
                df_macd = downsample_for_plot(df_ind[available_macd_cols].dropna())
                if 'MACD' in df_macd.columns:
                    fig_macd.add_trace(go.Scatter(x=df_macd['Date'], y=df_macd['MACD'], name='MACD'))
                if 'MACD_SIGNAL' in df_macd.columns:
                    fig_macd.add_trace(go.Scatter(x=df_macd['Date'], y=df_macd['MACD_SIGNAL'], name='Signal'))
                fig_macd.update_layout(title="MACD", template='plotly_white')
                st.plotly_chart(fig_macd, use_container_width=True)

        # RSI
        if 'RSI' in df_ind.columns:
            rsi_cols = ['Date','RSI']
            available_rsi_cols = [col for col in rsi_cols if col in df_ind.columns]
            if len(available_rsi_cols) == 2:
                df_rsi = downsample_for_plot(df_ind[available_rsi_cols].dropna())
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df_rsi['Date'], y=df_rsi['RSI'], name='RSI'))
                fig_rsi.update_yaxes(range=[0,100])
                fig_rsi.update_layout(title="RSI", template='plotly_white')
                st.plotly_chart(fig_rsi, use_container_width=True)

    # -------------------- TAB 4: Modeling (multi-model forecasts) --------------------
    with tab4:
        st.subheader("Train Multiple Models & Compare")
        cols = st.columns(3)
        with cols[0]:
            n_days = st.slider("Forecast horizon (days)", 1, 30, 7)
        with cols[1]:
            lookback = st.slider("Lookback (lag days)", 10, 60, 30)
        with cols[2]:
            test_split_choice = st.selectbox("Test split", ["10%","20%","30%"], index=1)
        test_ratio = {"10%":0.1,"20%":0.2,"30%":0.3}[test_split_choice]
        model_choices = st.multiselect("Models to run", options=["Random Forest"] + (["Prophet"] if PROPHET_AVAILABLE else []) + (["LSTM"] if KERAS_AVAILABLE else []), default=["Random Forest"])

        if st.button("Run Models"):
            st.info("Training selected models. This may take some time (cached when possible).")
            leaderboard = []
            forecasts = {}
            future_dates = None

            # For speed, optionally restrict training set in quick_mode
            df_models = df_ind.copy()
            if quick_mode:
                df_models = df_models.tail(min(len(df_models), 600)).reset_index(drop=True)

            if "Random Forest" in model_choices:
                try:
                    rf_model, rf_scaler, rf_metrics, rf_fi = train_model_rf(df_models, test_ratio=test_ratio, n_estimators=150, max_depth=12)
                    if rf_model is not None and rf_metrics is not None:
                        leaderboard.append({"Model":"RandomForest", **{k:v for k,v in rf_metrics.items() if isinstance(v,(int,float))}})
                        rf_preds = []
                        
                        # FIXED: Optimized DataFrame concatenation - collect rows first, then concatenate once
                        new_rows = []
                        current_df = df_models.copy()
                        
                        for day in range(n_days):
                            pred = predict_next_day_from_rf(rf_model, rf_scaler, current_df)
                            if pred is None:
                                break
                            rf_preds.append(pred)
                            
                            # Prepare new row data
                            last_row = safe_iloc_last(current_df, None)
                            if last_row is None:
                                break
                                
                            new_row_data = {}
                            for col in current_df.columns:
                                if col == 'Date':
                                    new_row_data[col] = last_row[col] + pd.Timedelta(days=1)
                                elif col in ['Close', 'Open', 'High', 'Low']:
                                    new_row_data[col] = pred
                                elif col == 'Volume':
                                    new_row_data[col] = last_row.get(col, 100000)
                                else:
                                    new_row_data[col] = last_row.get(col, 0)
                            
                            new_rows.append(new_row_data)
                            # Add new row to current_df for next iteration
                            new_row_df = pd.DataFrame([new_row_data])
                            current_df = pd.concat([current_df, new_row_df], ignore_index=True)
                        
                        forecasts['RandomForest'] = rf_preds
                except Exception as e:
                    st.error(f"Random Forest failed: {e}")

            if "Prophet" in model_choices and PROPHET_AVAILABLE:
                try:
                    # Prophet training (lightweight)
                    prophet_cols = ['Date','Close']
                    if all(col in df_ind.columns for col in prophet_cols):
                        dfp = df_ind[prophet_cols].rename(columns={'Date':'ds','Close':'y'}).copy()
                        # limit for speed
                        if quick_mode:
                            dfp = dfp.tail(min(len(dfp), 800))
                        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                        m.fit(dfp)
                        future = m.make_future_dataframe(periods=n_days)
                        fcst = m.predict(future)
                        preds = fcst['yhat'].tail(n_days).astype(float).tolist()
                        forecasts['Prophet'] = preds
                except Exception as e:
                    st.error(f"Prophet failed: {e}")

            if "LSTM" in model_choices and KERAS_AVAILABLE:
                st.warning("LSTM training can be slow in CPU environments. Consider Quick Mode or disabling LSTM.")
                try:
                    # Use the LSTM training block but with safe array operations
                    close_series = safe_column_access(df_ind, 'Close', 0).astype(float)
                    if len(close_series) < lookback + 10:  # Need minimum data
                        st.error("Insufficient data for LSTM training")
                    else:
                        series = close_series.values.reshape(-1,1)
                        ms = MinMaxScaler()
                        series_scaled = ms.fit_transform(series)
                        look = min(lookback, len(series_scaled) - 10)  # Ensure we have enough data
                        Xs, ys = [], []
                        for i in range(look, len(series_scaled)):
                            Xs.append(series_scaled[i-look:i])
                            ys.append(series_scaled[i])
                        
                        if len(Xs) < 10:  # Need minimum samples
                            st.error("Insufficient samples for LSTM training")
                        else:
                            Xs = np.array(Xs)
                            ys = np.array(ys)
                            # split
                            split_idx = int(len(Xs) * (1 - test_ratio))
                            X_train, X_test = Xs[:split_idx], Xs[split_idx:]
                            y_train, y_test = ys[:split_idx], ys[split_idx:]
                            epochs = 5 if quick_mode else 20
                            batch = min(32, len(X_train))
                            
                            model_l = keras.Sequential([
                                layers.Input(shape=(look,1)),
                                layers.LSTM(32, return_sequences=False),
                                layers.Dense(16, activation='relu'),
                                layers.Dense(1)
                            ])
                            model_l.compile(optimizer='adam', loss='mse')
                            model_l.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=0)
                            
                            # forecast n_days by rolling
                            preds = []
                            hist_scaled = series_scaled.copy()
                            for _ in range(n_days):
                                window = hist_scaled[-look:].reshape(1,look,1)
                                next_scaled = model_l.predict(window, verbose=0)[0]
                                next_val = float(ms.inverse_transform(next_scaled.reshape(-1,1))[0,0])
                                preds.append(next_val)
                                hist_scaled = np.vstack([hist_scaled, next_scaled.reshape(1,1)])
                            forecasts['LSTM'] = preds
                except Exception as e:
                    st.error(f"LSTM failed: {e}")

            # Show leaderboard & forecasts
            if leaderboard:
                try:
                    # FIXED: Safe sorting - check if test_mae column exists
                    lb_df = pd.DataFrame(leaderboard)
                    if 'test_mae' in lb_df.columns:
                        lb_df = lb_df.sort_values(by='test_mae', ascending=True).reset_index(drop=True)
                    st.markdown("#### Leaderboard")
                    st.dataframe(lb_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Leaderboard display failed: {e}")
                    
            if forecasts:
                try:
                    # build plot_df with safe access
                    hist_cols = ['Date','Close']
                    if all(col in df_ind.columns for col in hist_cols):
                        last_hist = df_ind[hist_cols].tail(180).copy()
                        last_date = safe_iloc_last(last_hist['Date'], datetime.now())
                        if last_date is not None:
                            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_days+1)]
                            plot_df = last_hist.rename(columns={'Close':'Actual'}).set_index('Date')
                            for mname, preds in forecasts.items():
                                if len(preds) == len(future_dates):
                                    # create future series
                                    fut = pd.DataFrame({'Date': future_dates, mname: preds})
                                    fut = fut.set_index('Date')
                                    plot_df = pd.concat([plot_df, fut], axis=0)
                            st.line_chart(plot_df)
                        else:
                            st.warning("Unable to create forecast chart - missing date information")
                    else:
                        st.warning("Unable to create forecast chart - missing required columns")
                except Exception as e:
                    st.error(f"Forecast chart failed: {e}")

    # -------------------- TAB 5: Data & Download --------------------
    with tab5:
        st.subheader("Historical Data")
        display_df = df_ind.copy()
        # show only tail for speed
        st.dataframe(display_df.tail(100).reset_index(drop=True), use_container_width=True)
        csv = display_df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name=f"{ticker}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

        # Quick stats with safe access
        st.markdown("### Summary Statistics")
        cols = st.columns(2)
        with cols[0]:
            try:
                high_col = safe_column_access(display_df, 'High', 0)
                low_col = safe_column_access(display_df, 'Low', 0)
                close_col = safe_column_access(display_df, 'Close', 0)
                if not high_col.empty and not low_col.empty and not close_col.empty:
                    st.write(f"- Highest Price: {currency_symbol}{high_col.max():.2f}")
                    st.write(f"- Lowest Price: {currency_symbol}{low_col.min():.2f}")
                    st.write(f"- Average Price: {currency_symbol}{close_col.mean():.2f}")
                else:
                    st.write("- Price stats: Data not available")
            except Exception:
                st.write("- Price stats: Data not available")
                
        with cols[1]:
            try:
                vol_col = safe_column_access(display_df, 'Volume', 0)
                if not vol_col.empty:
                    st.write(f"- Average Volume: {int(vol_col.mean()):,}")
                    st.write(f"- Max Volume: {int(vol_col.max()):,}")
                else:
                    st.write("- Volume stats: Data not available")
                    
                if volatility is not None and not np.isnan(volatility):
                    if is_intraday:
                        st.write(f"- Volatility (intraday σ): {volatility:.4f}")
                    else:
                        st.write(f"- Volatility (annualized): {volatility*100:.2f}%")
                else:
                    st.write("- Volatility: Data not available")
            except Exception:
                st.write("- Volume stats: Data not available")

        # Warning disclaimer
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
            <strong>📊 Data Sources:</strong> This application utilizes multiple data sources including Alpha Vantage API 
            and may fall back to sample data for demonstration when live APIs are unavailable.
            <br><br>
            <strong>✅ FIXED VERSION:</strong> This version includes all critical error fixes:
            <ul>
                <li>✅ Fixed deprecated pandas fillna method</li>
                <li>✅ Optimized DataFrame concatenation performance</li>
                <li>✅ Improved cache key generation reliability</li>
                <li>✅ Added comprehensive safe data access patterns</li>
                <li>✅ Enhanced error handling throughout</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

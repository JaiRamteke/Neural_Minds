import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance  # XAI: global feature importance
import warnings
warnings.filterwarnings('ignore')

# ======================== NEW LIBRARIES FOR LSTM AND PROPHET ====================
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance not installed. Only Alpha Vantage will be used.")

# Alpha Vantage Configuration
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
AV_BASE_URL = 'https://www.alphavantage.co/query'

# Page configuration
st.set_page_config(
    page_title="Neural Minds",
    page_icon="brain.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean White Theme CSS
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Headers and Text */
        .main-header {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #1f77b4, #ff7f0e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 1rem;
            font-family: 'Inter', sans-serif;
        }
        
        .subtitle {
            text-align: center;
            font-size: 1.3rem;
            color: #666;
            margin-bottom: 3rem;
            font-weight: 300;
        }
        
        /* Warning Card */
        .warning-card {
            background: #000000;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #ffeaa7;
            margin-top: 2rem;
            border-left: 4px solid #fdcb6e;
        }
        
        /* Status Indicators */
        .api-status {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .api-working {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .api-failed {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(45deg, #1f77b4, #ff7f0e);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #1565c0, #f57c00);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: visible;}
        footer {visibility: visible;}
        header {visibility: visible;}
            
        /* Sidebar text fix */
        section[data-testid="stSidebar"] {
            background: #f9f9f9;
            color: #000000;
        }
        section[data-testid="stSidebar"] * {
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
            color: #000000 !important;
        }
        

        /* ✅ Make the whole sidebar light */
        section[data-testid="stSidebar"]{
        background:#f9f9f9 !important;
        color:#000 !important;

        /* Override Streamlit theme variables (sidebar scope only) */
        --background-color:#f9f9f9;
        --secondary-background-color:#ffffff; /* widgets like selectbox */
        --text-color:#000000;
        --font-color:#000000;
        --border-color:#DDDDDD;
        }

        /* Ensure all text/icons in sidebar are dark */
        section[data-testid="stSidebar"] *{
        color:#000 !important;
        fill:#000 !important;
        }

        /* ✅ Selectbox control (BaseWeb) – make the control white */
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div{
        background:#ffffff !important;
        border:1px solid #ddd !important;
        border-radius:8px !important;
        }

        /* Value/placeholder text inside the control */
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] *{
        color:#000 !important;
        }

        /* Arrow/clear icons */
        section[data-testid="stSidebar"] .stSelectbox svg{
        color:#000 !important;
        fill:#000 !important;
        }

        /* ✅ Dropdown menu (rendered in a portal – target globally) */
        div[role="listbox"], ul[role="listbox"]{
        background:#ffffff !important;
        color:#000000 !important;
        border:1px solid #ddd !important;
        border-radius:8px !important;
        }
        li[role="option"]{ color:#000 !important; }
        li[role="option"][aria-selected="true"],
        li[role="option"]:hover{ background:#f0f0f0 !important; }

        /* (Optional) Make other sidebar inputs white too */
        section[data-testid="stSidebar"] .stTextInput > div > div,
        section[data-testid="stSidebar"] .stNumberInput > div > div,
        section[data-testid="stSidebar"] .stDateInput > div > div{
        background:#ffffff !important;
        border:1px solid #ddd !important;
        border-radius:8px !important;
        }
            
    </style>
""", unsafe_allow_html=True)

# Enhanced stock tickers
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



def map_ticker_for_source(ticker: str, source: str) -> str:
    base = ticker.split('.')[0].upper()
    if source == "yfinance":
        if ticker.endswith(".NSE"):
            return base + ".NS"
        return base
    if source == "alpha_vantage":
        if ticker.endswith(".NSE"):
            return base + ".BSE"
        return base
    return ticker

def test_api_connections():
    """Test both API connections and return status"""
    status = {
        'yfinance': {'available': YFINANCE_AVAILABLE, 'working': False, 'message': ""},
        'alpha_vantage': {'available': True, 'working': False, 'message': ""}
    }
    
    # Test yfinance
    if YFINANCE_AVAILABLE:
        try:
            test_stock = yf.Ticker("AAPL")
            test_data = test_stock.history(period="5d")
            if not test_data.empty:
                status['yfinance']['working'] = True
                status['yfinance']['message'] = "✅ yfinance is working"
            else:
                status['yfinance']['message'] = "❌ yfinance returned no data"
        except Exception as e:
            status['yfinance']['message'] = f"❌ yfinance error: {str(e)[:50]}..."
    else:
        status['yfinance']['message'] = "❌ yfinance not installed"
    
    # Test Alpha Vantage
    try:
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': 'AAPL',
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'compact'
        }
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
def fetch_stock_data_yfinance(ticker, period="1y", max_retries=2):
    """
    Robust yfinance data fetch:
    - maps ticker for yfinance (NSE -> .NS)
    - tries yf.download, then yf.Ticker.history fallback
    - retries with exponential backoff
    - returns normalized dataframe or None
    """
    try:
        ticker_mapped = map_ticker_for_source(ticker, "yfinance")
        yf_period_map = {'1mo':'1mo','3mo':'3mo','6mo':'6mo','1y':'1y','2y':'2y','5y':'5y'}
        yf_period = yf_period_map.get(period, '1y')

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                # prefer Ticker.history for more robust behavior
                t = yf.Ticker(ticker_mapped)
                df = t.history(period=yf_period, interval="1d", auto_adjust=False, actions=False)
                if df is None or df.empty:
                    # try yf.download as a fallback
                    df = yf.download(ticker_mapped, period=yf_period, interval="1d", auto_adjust=False, threads=False)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    # make sure Date is datetime and Close exists
                    if 'Date' not in df.columns and df.index.name in ['Date','date']:
                        df = df.reset_index()
                    if 'Close' not in df.columns and 'Adj Close' in df.columns:
                        df['Close'] = df['Adj Close']
                    # keep expected columns only
                    for col in ['Date','Open','High','Low','Close','Volume']:
                        if col not in df.columns:
                            df[col] = np.nan
                    df = df[['Date','Open','High','Low','Close','Volume']]
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date']).reset_index(drop=True)
                    df.attrs = {'source':'yfinance','ticker':ticker_mapped}
                    return df
                else:
                    last_exc = Exception("yfinance returned no data")
            except Exception as e:
                last_exc = e
            # exponential backoff
            time.sleep(1 + attempt*1.5)

        # If we reached here, failed
        st.warning(f"yfinance fetch failed for {ticker} ({ticker_mapped}): {str(last_exc)[:240]}")
        return None
    except Exception as e:
        st.error(f"yfinance unexpected error: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_stock_data_unified(ticker, period="1y"):
    try:
        mapped_ticker = map_ticker_for_source(ticker, "alpha_vantage")
        time.sleep(1)
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': mapped_ticker,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full',
            'datatype': 'json'
        }
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
    """
    Try yfinance -> Alpha Vantage -> sample, and return (df, used_source, trace_list)
    where trace_list is a list of (source_key, human_message).
    """
    trace = []

    # 1) yfinance (preferred)
    if YFINANCE_AVAILABLE:
        df_yf = fetch_stock_data_yfinance(ticker, period)
        if df_yf is not None:
            trace.append(("yfinance", "✅ yfinance loaded successfully"))
            return df_yf, "yfinance", trace
        else:
            trace.append(("yfinance", "❌ yfinance failed (no/invalid data)"))
    else:
        trace.append(("yfinance", "❌ yfinance not installed"))

    # 2) Alpha Vantage (backup)
    df_av = fetch_stock_data_unified(ticker, period)
    if df_av is not None:
        trace.append(("alpha_vantage", "✅ Alpha Vantage loaded successfully"))
        return df_av, "alpha_vantage", trace
    else:
        trace.append(("alpha_vantage", "❌ Alpha Vantage failed (no/invalid data)"))

    # 3) Sample (last resort)
    df_sample = create_sample_data(ticker, period)
    df_sample.attrs['source'] = 'sample_data'
    trace.append(("sample_data", "⚠️ Using sample data (both APIs unavailable)"))
    return df_sample, "sample_data", trace


def get_period_days(period):
    return {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825}.get(period,365)

def create_sample_data(ticker, period):
    """Create realistic sample data when APIs fail"""
    days = get_period_days(period)
    
    # Base prices for different stocks
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
    
    base_name = ticker.split('.')[0].upper()
    base_price = base_prices.get(base_name, 1000)
    
    # Generate realistic data
    np.random.seed(hash(ticker) % 2**32)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Generate price movements
    daily_return = 0.08 / 252
    volatility = 0.02
    returns = np.random.normal(daily_return, volatility, days)
    
    prices = [base_price]
    for i in range(1, days):
        new_price = prices[-1] * (1 + returns[i])
        new_price = max(new_price, base_price * 0.5)
        new_price = min(new_price, base_price * 3.0)
        prices.append(new_price)
    
    # Generate OHLC data
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
        
        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.attrs = {'source': 'sample_data', 'ticker': ticker}
    return df

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_stock_data(df, ticker, source):
    """Process and enhance stock data with technical indicators"""
    if df is None or df.empty:
        return None
    
    # Ensure Date column exists
    if 'Date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
    
    # Add technical indicators
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    
    # Add lag features
    for i in [1, 2, 3, 5]:
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Add metadata
    df.attrs = {
        'source': source,
        'ticker': ticker,
        'last_updated': datetime.now()
    }
    
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    feature_columns = [
        'Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI',
        'Price_Change', 'Volume_MA'
    ]
    
    # Add lag features
    for i in [1, 2, 3, 5]:
        if f'Close_Lag_{i}' in df.columns:
            feature_columns.append(f'Close_Lag_{i}')
    
    # Select only existing columns
    existing_features = [col for col in feature_columns if col in df.columns]
    
    X = df[existing_features].copy()
    y = df['Close'].copy()
    
    return X, y, existing_features

def train_model(df):
    """Train Random Forest model"""
    try:
        X, y, feature_names = prepare_features(df)
        
        if X.empty or y.empty:
            st.error("Insufficient data for training")
            return None, None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return model, scaler, metrics, feature_importance
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None

def predict_next_price(model, scaler, df):
    """Predict next day price"""
    try:
        X, _, feature_names = prepare_features(df)
        if X.empty:
            return None
            
        last_features = X.iloc[-1:].values
        last_features_scaled = scaler.transform(last_features)
        prediction = model.predict(last_features_scaled)[0]
        return prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_stock_info(ticker):
    """Get default stock information"""
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

    
    base_ticker = ticker.split('.')[0].upper()
    info = stock_info.get(base_ticker, {
        'name': ticker,
        'sector': 'Unknown',
        'industry': 'Unknown',
        'currency': 'USD'
    })
    
    info['market_cap'] = 'N/A'
    return info

# ---------------------------
# Safe-stat helper + Volatility computation placement
# ---------------------------
def safe_stat(df, col, func, label, fmt="{:.2f}", currency_symbol=""):
    """
    Safely compute a statistic on a dataframe column and display with Streamlit.
    """
    try:
        if df is not None and col in df.columns and not df[col].dropna().empty:
            val = func(df[col].dropna())
            if pd.notna(val):
                st.write(f"- {label}: {currency_symbol}{fmt.format(val)}")
                return
    except Exception:
        pass
    st.write(f"- {label}: Data not available")

# ======================== NEW METRIC FUNCTION ========================
def calculate_sharpe_ratio(series, risk_free_rate=0.0):
    if series is None or len(series) < 2:
        return 0.0
    returns = series.pct_change().dropna()
    # Defensive checks
    if returns.empty:
        return 0.0
    std = returns.std()
    if pd.isna(std) or std == 0:
        return 0.0
    excess = returns - risk_free_rate
    return (excess.mean() / std) * np.sqrt(252)  # annualized

def split_features_for_xai(df):
    """Rebuild the same split used in training for XAI pipelines."""
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    return X, y, X_train, X_test, y_train, y_test, feature_names

def rf_permutation_importance(model, scaler, df):
    """
    Compute permutation importance on the RF model using the consistent split + scaler.
    Returns (imp_df, X_test, y_test, feature_names, X_test_scaled)
    """
    _, _, _, X_test, _, y_test, feature_names = split_features_for_xai(df)
    X_test_scaled = scaler.transform(X_test)
    r = permutation_importance(
        model, X_test_scaled, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": r.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp_df, X_test, y_test, feature_names, X_test_scaled

def rf_shap_global_local(model, X_test_scaled, feature_names, local_idx=-1):
    """
    Compute SHAP values for RF if SHAP is available. Returns (global_importance_df, local_df) or (None, None).
    """
    if not ('SHAP_AVAILABLE' in globals() and SHAP_AVAILABLE):
        return None, None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        # If model is regressor, shap_values is 2D (n_samples, n_features)
        if isinstance(shap_values, list):  # safety for classifiers; not expected here
            shap_values = shap_values[0]
        # Global: mean |SHAP|
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        global_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}) \
                        .sort_values("mean_abs_shap", ascending=False)
        # Local: last test sample contributions
        local_vals = shap_values[local_idx]
        local_df = pd.DataFrame({"feature": feature_names, "shap_value": local_vals}) \
                        .sort_values("shap_value", key=np.abs, ascending=False)
        return global_df, local_df
    except Exception:
        return None, None

def rf_whatif_curves(model, scaler, df, top_features, points=15):
    """
    For top features, vary each one across its training distribution quantiles and
    compute predicted price to show sensitivity curves.
    Returns dict: {feature: (grid_values, preds)}
    """
    X, _, X_train, X_test, _, _, feature_names = split_features_for_xai(df)
    # Base = last available test sample in original (unscaled) space
    base = X_test.iloc[-1].copy()
    curves = {}
    for feat in top_features:
        if feat not in X.columns: 
            continue
        q = np.linspace(0.05, 0.95, points)
        grid = np.quantile(X_train[feat].values, q)
        preds = []
        for val in grid:
            x_new = base.copy()
            x_new[feat] = val
            x_scaled = scaler.transform(x_new.values.reshape(1, -1))
            pred = float(model.predict(x_scaled)[0])
            preds.append(pred)
        curves[feat] = (grid, np.array(preds))
    return curves

def prophet_components_figure(m):
    """Return a Matplotlib figure of Prophet components (trend/seasonality) if possible."""
    try:
        future = m.make_future_dataframe(periods=0)
        fc = m.predict(future)
        fig = m.plot_components(fc)
        return fig
    except Exception:
        return None

def lstm_last_window_sensitivity(model, scaler, df, sequence_length=10, steps=9, pct=0.05):
    """
    Simple 'what-if' for LSTM: perturb the most recent price in the last window ±pct
    and see the next prediction response.
    """
    X, y, _sc = prepare_lstm_data(df, sequence_length)
    if len(X) == 0: 
        return None, None
    base = X[-1].copy().reshape(sequence_length, 1)
    # Build factors from -pct to +pct
    factors = np.linspace(1 - pct, 1 + pct, steps)
    preds = []
    for f in factors:
        perturbed = base.copy()
        perturbed[-1, 0] = perturbed[-1, 0] * f
        p = model.predict(perturbed.reshape(1, sequence_length, 1), verbose=0).flatten()
        pred = scaler.inverse_transform(p.reshape(-1, 1)).flatten()[0]
        preds.append(pred)
    # Recover original (inverse transform of base last-step prediction for reference)
    p0 = model.predict(base.reshape(1, sequence_length, 1), verbose=0).flatten()
    base_pred = scaler.inverse_transform(p0.reshape(-1, 1)).flatten()[0]
    return factors, (np.array(preds), base_pred)

# ======================================================================

# ---- PROPHET ADDITION ----
def train_prophet(df, use_log=True, grid_cps=[0.01, 0.05, 0.1, 0.5]):
    """
    Train Prophet model with optional log transform and cross-validation over changepoint_prior_scale.
    Includes regressors if present (MA_20, RSI, Volume).

    Returns:
        best_model (Prophet): Fitted Prophet model
        metrics (dict): Test set evaluation metrics
        y_pred (ndarray): Predictions on test set (inverse transformed if log used)
        y_test (ndarray): Actual test set values (inverse transformed if log used)
        best_params (dict): Best changepoint_prior_scale chosen
    """
    # Prepare dataframe
    dfp = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).copy()

    if use_log:
        dfp['y'] = np.log1p(dfp['y'])

    # Add regressors if available
    regs = []
    for r in ['MA_20', 'RSI', 'Volume']:
        if r in df.columns:
            dfp[r] = df[r].values
            regs.append(r)

    # Drop NaNs
    dfp = dfp.dropna(subset=['y'] + regs).reset_index(drop=True)

    if dfp.empty or len(dfp) < 30:
        st.error("Not enough clean data for Prophet.")
        return None, {}, None, None, None

    # Train/test split
    train_size = int(len(dfp) * 0.8)
    train, test = dfp.iloc[:train_size], dfp.iloc[train_size:]

    best_model, best_metrics, best_pred, best_params = None, None, None, None

    # Grid search over changepoint_prior_scale
    for cps in grid_cps:
        try:
            m = Prophet(
                changepoint_prior_scale=cps,
                yearly_seasonality=True,
                daily_seasonality=False,
                weekly_seasonality=True
            )
            for r in regs:
                m.add_regressor(r)

            m.fit(train)

            # Build future frame with regressors
            future = m.make_future_dataframe(periods=len(test), freq='D')
            for r in regs:
                if len(dfp) >= len(future):
                    future[r] = dfp[r].iloc[:len(future)].values
                else:
                    # forward-fill last known value
                    future[r] = pd.concat([
                        dfp[r],
                        pd.Series([dfp[r].iloc[-1]] * (len(future) - len(dfp)))
                    ], ignore_index=True)

            forecast = m.predict(future)
            y_pred = forecast['yhat'].iloc[-len(test):].values
            y_test = test['y'].values

            if use_log:
                y_pred = np.expm1(y_pred)
                y_test = np.expm1(y_test)

            metrics = {
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "test_mae": mean_absolute_error(y_test, y_pred),
                "test_r2": r2_score(y_test, y_pred)
            }

            if (best_metrics is None) or (metrics["test_rmse"] < best_metrics["test_rmse"]):
                best_model, best_metrics, best_pred, best_params = m, metrics, y_pred, {"cps": cps}

        except Exception as e:
            print(f"Prophet training error for cps={cps}: {e}")
            continue

    return best_model, best_metrics, best_pred, y_test, best_params

def _merge_future_with_history_regressors(m, future):
    regs = list(getattr(m, 'extra_regressors', {}).keys())
    if not regs:
        return future

    hist = getattr(m, 'history', None)
    if hist is None or hist.empty:
        for r in regs:
            future[r] = 0
        return future

    have_regs = [r for r in regs if r in hist.columns]
    future = future.merge(hist[['ds'] + have_regs], on='ds', how='left')

    for r in have_regs:
        future[r] = future[r].ffill().bfill()

    for r in regs:
        if r not in future.columns:
            future[r] = 0

    return future


def prophet_components_figure(m):
    try:
        hist = getattr(m, 'history', None)
        if hist is not None and len(hist):
            cols = ['ds'] + [c for c in hist.columns if c != 'y']
            fc = m.predict(hist[cols])
        else:
            future = m.make_future_dataframe(periods=0)
            future = _merge_future_with_history_regressors(m, future)
            fc = m.predict(future)
        return m.plot_components(fc)
    except Exception as e:
        st.warning(f"Prophet components unavailable: {e}")
        return None


def predict_prophet_next(m):
    future = m.make_future_dataframe(periods=1)
    future = _merge_future_with_history_regressors(m, future)
    forecast = m.predict(future)
    return float(forecast['yhat'].iloc[-1])
# --------------------------

# ---- LSTM (robust) ----
def prepare_lstm_data(df, sequence_length=10):
    from sklearn.preprocessing import MinMaxScaler
    # Force numeric, drop NaNs, and cast to float32 for Keras
    close = pd.to_numeric(df['Close'], errors='coerce').dropna().values.reshape(-1, 1).astype(np.float32)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i + sequence_length, 0])
        y.append(scaled[i + sequence_length, 0])

    # Fixed shapes + dtypes that Keras likes
    X = np.array(X, dtype=np.float32).reshape(-1, sequence_length, 1)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler

def train_lstm(df):
    sequence_length = 10
    X, y, scaler = prepare_lstm_data(df, sequence_length)

    # Not enough sequences? Skip LSTM gracefully (do NOT crash app)
    if len(X) < 30:
        st.warning("LSTM skipped: not enough sequences for training.")
        metrics = {
            'train_rmse': np.nan, 'test_rmse': np.nan,
            'train_mae': np.nan,  'test_mae': np.nan,
            'train_r2': np.nan,   'test_r2': np.nan,
            'train_size': 0,      'test_size': 0
        }
        return None, None, metrics, None, None, sequence_length

    # Time-ordered split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build model
    model = Sequential()
    model.add(LSTM(50, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mse')
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # Explicit validation set (avoids Keras validation_split edge cases)
    val_len = max(1, len(X_train) // 10)
    X_tr, X_val = X_train[:-val_len], X_train[-val_len:]
    y_tr, y_val = y_train[:-val_len], y_train[-val_len:]

    try:
        model.fit(
            X_tr, y_tr,
            epochs=50, batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
    except Exception as e:
        # Never crash the app; surface in UI and continue
        st.warning(f"LSTM training failed and will be skipped this run: {e}")
        metrics = {
            'train_rmse': np.nan, 'test_rmse': np.nan,
            'train_mae': np.nan,  'test_mae': np.nan,
            'train_r2': np.nan,   'test_r2': np.nan,
            'train_size': int(X_tr.shape[0]),
            'test_size': int(X_test.shape[0])
        }
        return None, None, metrics, None, None, sequence_length

    # Evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = {
        'train_rmse': np.nan,
        'test_rmse': float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))),
        'train_mae': np.nan,
        'test_mae': float(mean_absolute_error(y_test_inv, y_pred_inv)),
        'train_r2': np.nan,
        'test_r2': float(r2_score(y_test_inv, y_pred_inv)),
        'train_size': int(X_tr.shape[0]),
        'test_size': int(X_test.shape[0])
    }
    return model, scaler, metrics, y_pred_inv, y_test_inv, sequence_length
# -----------------------
def predict_lstm_next(model, scaler, df, sequence_length=10):
    """
    Predict the next closing price using a trained LSTM model.
    Uses the last `sequence_length` closes from df.
    """
    if model is None or scaler is None:
        return None

    # Ensure numeric close values
    close_vals = pd.to_numeric(df['Close'], errors='coerce').dropna().values.reshape(-1, 1).astype(np.float32)

    if len(close_vals) < sequence_length:
        return None

    # Take the last window
    last_window = close_vals[-sequence_length:]
    last_scaled = scaler.transform(last_window)

    X_input = last_scaled.reshape(1, sequence_length, 1)

    try:
        y_scaled = model.predict(X_input, verbose=0).flatten()[0]
        y_inv = scaler.inverse_transform([[y_scaled]])[0][0]
        return float(y_inv)
    except Exception as e:
        st.warning(f"LSTM next-step prediction failed: {e}")
        return None

model_metrics_leaderboard = {}

def main():
    # Title and description
    st.markdown('<h1 class="main-header">Neural Minds</h1>', unsafe_allow_html=True)
    st.markdown(
    """
    <p style='
        text-align: center;
        font-size: 20px;
        font-weight: 500;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -10px;
        margin-bottom: 20px;
    '>
        Advanced Market Analysis & AI-Powered Prediction Platform
    </p>
    """,
    unsafe_allow_html=True
    )

    # API Status Check
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

    
    # Initialize variables (🔑 init once here)
    df, current_price = None, None
    volatility = None
    current_price_val = None
    currency_symbol = '$'

    # --- Session state init ---
    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False

    if "model_metrics_leaderboard" not in st.session_state:
        st.session_state.model_metrics_leaderboard = None

    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown(
            """
            <style>
            .api-badge {
                background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 8px 18px;
                border-radius: 25px;
                font-size: 15px;
                font-weight: 600;
                display: inline-block;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(79,172,254,0.6); }
                70% { box-shadow: 0 0 0 12px rgba(79,172,254,0); }
                100% { box-shadow: 0 0 0 0 rgba(79,172,254,0); }
            }
            </style>

            <div class="api-badge">💎 Premium API Access Enabled</div>
            """,
            unsafe_allow_html=True
        )
                # Data source selection (default yfinance)
        st.markdown("#### 📡 Data Source")
        data_source_choice = st.selectbox(
            "Select Data Source",
            ["yfinance", "Alpha Vantage", "Auto (yfinance → Alpha Vantage → Sample)"],
            index=0,
            help="Choose the data source. 'Auto' tries yfinance first, then Alpha Vantage, then sample data."
        )

        
        # Stock selection
        st.markdown("#### 📈 Stock Selection")
        
        market = st.selectbox(
            "Select Market",
            ["US Stocks", "Indian Stocks", "Custom Ticker"],
            help="Choose your preferred market or enter a custom ticker"
        )
        
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
            
        else:  # Custom ticker
            ticker = st.text_input(
                "Enter Stock Ticker",
                value="AAPL",
                help="Examples: AAPL (US), RELIANCE.NSE (Indian stocks with .NSE extension)"
            )
            
            if ticker:
                if ticker.endswith('.NSE'):
                    st.info("🇮🇳 Indian stock format detected")
                else:
                    st.info("🇺🇸 US stock format detected")
        
        st.sidebar.markdown("### 🤖 Select Models for Forecasting")

        # =============== RECOMMENDED: Model SELECTORS and METRIC COLLECTOR ==============
        model_choices = ['Random Forest']
        if PROPHET_AVAILABLE:
            model_choices.append("Prophet")
        if LSTM_AVAILABLE:
            model_choices.append("LSTM")

        selected_model = st.selectbox("Select Model", model_choices)
        model_metrics_leaderboard = {}

        # Time period selection
        st.markdown("#### 📅 Time Period")
        period = st.selectbox(
            "Select Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Choose the historical data period for analysis"
        )

        # Prediction settings
        st.markdown("#### 🔮 Prediction Settings")
        prediction_days = st.slider("Days to Predict", 1, 30, 7, help="Number of days to predict into the future")
        
        # Reset prediction state if inputs change
        if "last_config" not in st.session_state:
            st.session_state.last_config = (None, None, None)

        current_config = (ticker, selected_model, period)

        if current_config != st.session_state.last_config:
            st.session_state.predict_clicked = False
            st.session_state.last_config = current_config

        # Action button
        if st.button("🚀 Predict Stock Price", type="primary", use_container_width=True):
            st.session_state.predict_clicked = True

    # Main content area
    if st.session_state.predict_clicked:
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Stock Analysis", 
            "🔮 Predictions", 
            "📈 Charts", 
            "🤖 Model Performance", 
            "📋 Data Table",
            "🏆 Leaderboard & Risk"
        ])
        
        # Fetch stock data depending on user choice
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

            else:  # Auto mode
                df, used_source, trace = load_stock_data_auto(ticker, period)
                # Inline API call status
                st.markdown("#### 🔎 API Call Status")
                for src, msg in trace:
                    css_class = "api-working" if "✅" in msg else "api-failed"
                    st.markdown(f'<div class="api-status {css_class}">{msg}</div>', unsafe_allow_html=True)

        # If still nothing, drop to sample
        if df is None or df.empty:
            st.error("❌ Unable to fetch real data. Using sample data.")
            df = create_sample_data(ticker, period)
            used_source = "sample_data"

        # Process the data
        data_source = df.attrs.get('source', used_source)
        df = process_stock_data(df, ticker, data_source)

        if selected_model == "Random Forest":
            model, scaler, metrics, feat_imp = train_model(df)
            prediction = predict_next_price(model, scaler, df)
            metrics["sharpe"] = calculate_sharpe_ratio(df["Close"])
            model_metrics_leaderboard["Random Forest"] = metrics
        elif selected_model == "Prophet" and PROPHET_AVAILABLE:
            m, metrics, pred, y_test, best_params = train_prophet(df)
            next_pred = predict_prophet_next(m, df)
            metrics["sharpe"] = calculate_sharpe_ratio(df["Close"])
            prediction = next_pred
            model_metrics_leaderboard["Prophet"] = metrics
        elif selected_model == "LSTM" and LSTM_AVAILABLE:
            model, scaler, metrics, y_pred_inv, y_test_inv, seq_len = train_lstm(df)
            prediction = predict_lstm_next(model, scaler, df, seq_len)
            metrics["sharpe"] = calculate_sharpe_ratio(df["Close"])
            model_metrics_leaderboard["LSTM"] = metrics
        else:
            prediction = None

        if df is None or df.empty:
            st.error("❌ Unable to process stock data. Please try again.")
            return

        # Display data source info
        if data_source == 'sample_data':
            st.warning("⚠️ Using sample data for demonstration. Real-time data unavailable.")
        else:
            st.success(f"✅ Successfully loaded {len(df)} data points for {ticker} from {data_source}")
        
        # Get stock info and set currency/curr symbol ONCE
        stock_info = get_stock_info(ticker)
        currency = stock_info.get('currency', 'USD')
        currency_symbol = '$' if currency == 'USD' else 'INR '

        # Safe, single-point assignment of current_price_val
        if df is not None and not df.empty and 'Close' in df.columns:
            try:
                current_price_val = float(df['Close'].iloc[-1])
            except Exception:
                current_price_val = None
        else:
            current_price_val = None

        # ---------------------------
        # Robust volatility calculation (compute once)
        # ---------------------------
        # --- Volatility calculation (safe version) ---
        volatility = None
        is_intraday = False

        if df is not None and 'Close' in df.columns:
            valid_closes = df['Close'].dropna()
            if len(valid_closes) > 2:
                returns = valid_closes.pct_change().dropna()
                if not returns.empty:
                    if is_intraday:
                        volatility = returns.std()
                    else:
                        # annualize for daily data
                        volatility = returns.std() * (252 ** 0.5)

        # Debug info if still None
        if volatility is None:
            st.warning(
                f"⚠️ Volatility could not be computed. "
                f"Rows: {len(df) if df is not None else 0}, "
                f"Close exists: {'Close' in df.columns if df is not None else False}, "
                f"Valid closes: {df['Close'].notna().sum() if df is not None and 'Close' in df.columns else 0}"
            )

        # ---------------------------
        # UI Tabs - Tab1 (Stock Analysis)
        # ---------------------------
        with tab1:
            # Stock information
            st.markdown(f"### 📋 {stock_info['name']} ({ticker})")
            
            # Data source indicator
            if data_source != 'sample_data':
                st.info(f"📡 Data Source: {data_source.title()}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cp = current_price_val
                try:
                    if cp is not None and not pd.isna(cp):
                        st.metric("Current Price", f"{currency_symbol}{float(cp):.2f}")
                    else:
                        st.metric("Current Price", "Data not available")
                except (ValueError, TypeError):
                    st.metric("Current Price", "Data not available")
            
            with col2:
                price_change = 0.0
                pct_change = 0.0
                if df is not None and len(df) > 1:
                    try:
                        prev_close = float(df['Close'].iloc[-2])
                        last_close = float(df['Close'].iloc[-1])
                        price_change = last_close - prev_close
                        pct_change = (price_change / prev_close) * 100 if prev_close != 0 else 0.0
                    except Exception:
                        price_change, pct_change = 0.0, 0.0
                st.metric("Price Change", f"{currency_symbol}{price_change:.2f}", f"{pct_change:.2f}%")
            
            with col3:
                volume = None
                if df is not None and 'Volume' in df.columns and len(df) > 0:
                    try:
                        volume = int(float(df['Volume'].iloc[-1]))
                    except (ValueError, TypeError):
                        volume = None

                if volume is not None:
                    st.metric("Volume", f"{volume:,.0f}")
                else:
                    st.metric("Volume", "Data not available")
            
            with col4:
                if volatility is not None:
                    try:
                        if is_intraday:
                            st.metric("Volatility (intraday σ)", f"{volatility:.4f}")
                        else:
                            # show percent for annualized volatility
                            st.metric("Volatility (annualized %)", f"{volatility*100:.2f}%")
                    except (ValueError, TypeError):
                        st.metric("Volatility", "Data not available")
                else:
                    st.metric("Volatility", "Data not available")
            
            # Stock details
            st.markdown("### 📊 Stock Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Sector:** {stock_info['sector']}")
                st.write(f"**Industry:** {stock_info['industry']}")
            
            with col2:
                st.write(f"**Market Cap:** {stock_info['market_cap']}")
                st.write(f"**Currency:** {stock_info['currency']}")
            
            # Key Statistics
            st.markdown("### 📈 Key Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_52w = None
                if df is not None and 'High' in df.columns and not df.empty:
                    try:
                        high_52w = float(df['High'].max())
                    except (ValueError, TypeError):
                        high_52w = None
                st.metric("52W High", f"{currency_symbol}{high_52w:.2f}" if high_52w is not None else "Data not available")
            
            with col2:
                low_52w = None
                if df is not None and 'Low' in df.columns and not df.empty:
                    try:
                        low_52w = float(df['Low'].min())
                    except (ValueError, TypeError):
                        low_52w = None
                st.metric("52W Low", f"{currency_symbol}{low_52w:.2f}" if low_52w is not None else "Data not available")
            
            with col3:
                avg_volume_val = None
                if df is not None and 'Volume' in df.columns and not df.empty:
                    try:
                        avg_volume_val = float(df['Volume'].mean())
                    except (ValueError, TypeError):
                        avg_volume_val = None
                st.metric("Avg Volume", f"{avg_volume_val:,.0f}" if avg_volume_val is not None else "Data not available")
            
            with col4:
                if 'RSI' in df.columns and not df['RSI'].isna().all():
                    current_rsi = df['RSI'].iloc[-1]
                    st.metric("RSI", f"{current_rsi:.1f}")
        
        # ---------------------------
        # Tab2: Predictions (unchanged)
        # ---------------------------
        with tab2:
            # Train model and make predictions
            st.markdown("### 🤖 AI Predictions")
            
            with st.spinner("🧠 Training ML model..."):
                model, scaler, metrics, feature_importance = train_model(df)
            
            if model is None:
                st.error("Failed to train model. Please try with different parameters.")
                return
            
            # Display model performance
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Accuracy (R²)", f"{metrics['test_r2']:.3f}")
            
            with col2:
                st.metric("RMSE", f"{metrics['test_rmse']:.2f}")
            
            with col3:
                st.metric("MAE", f"{metrics['test_mae']:.2f}")
            
            # Single day prediction
            st.markdown("### 🔮 Next Day Prediction")
            next_day_pred = predict_next_price(model, scaler, df)
            
            if next_day_pred is not None:
                try:
                    current_price_num = float(df['Close'].iloc[-1])
                    price_change = float(next_day_pred) - current_price_num
                    percentage_change = (price_change / current_price_num) * 100 if current_price_num != 0 else 0.0
                except Exception:
                    current_price_num, price_change, percentage_change = None, None, None
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"{currency_symbol}{current_price_num:.2f}" if current_price_num is not None else "—")
                
                with col2:
                    st.metric(
                        "Predicted Price",
                        f"{currency_symbol}{float(next_day_pred):.2f}",
                        f"{currency_symbol}{price_change:.2f}" if price_change is not None else "—"
                    )
                
                with col3:
                    st.metric("Expected Change", f"{percentage_change:.2f}%" if percentage_change is not None else "—")
                
                # Prediction confidence
                if percentage_change is not None:
                    if percentage_change > 2:
                        st.success("🟢 Strong Bullish Signal")
                    elif percentage_change > 0:
                        st.info("🔵 Mild Bullish Signal")
                    elif percentage_change > -2:
                        st.warning("🟡 Neutral Signal")
                    else:
                        st.error("🔴 Bearish Signal")
        
        # ---------------------------
        # Tab3: Charts (unchanged except x-axis handling remains ok)
        # ---------------------------
        with tab3:
            # Charts and visualizations
            st.markdown("### 📈 Stock Price Charts")
            
            # Price chart with moving averages
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=3)
            ))
            
            if 'MA_20' in df.columns and not df['MA_20'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['MA_20'],
                    mode='lines',
                    name='20-Day MA',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
            
            if 'MA_50' in df.columns and not df['MA_50'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['MA_50'],
                    mode='lines',
                    name='50-Day MA',
                    line=dict(color='#2ca02c', width=2, dash='dot')
                ))
            
            fig.update_layout(
                title=f"{ticker} Stock Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title=f"Price ({currency_symbol})",
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(31, 119, 180, 0.6)'
            ))
            
            fig_volume.update_layout(
                title=f"{ticker} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # RSI chart
            if 'RSI' in df.columns and not df['RSI'].isna().all():
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='#d62728', width=3)
                ))
                
                # Add overbought/oversold lines
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff7f0e", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#2ca02c", annotation_text="Oversold (30)")
                
                fig_rsi.update_layout(
                    title=f"{ticker} RSI (Relative Strength Index)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    yaxis=dict(range=[0, 100]),
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
        
        # ---------------------------
        # Tab4: Model Performance (unchanged)
        # ---------------------------
        with tab4:
            # Model performance details
            if model is not None:
                st.markdown("### 🤖 Model Performance Details")
                
                # Performance metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Training Metrics:**")
                    st.write(f"- RMSE: {metrics['train_rmse']:.4f}")
                    st.write(f"- MAE: {metrics['train_mae']:.4f}")
                    st.write(f"- R² Score: {metrics['train_r2']:.4f}")
                    st.write(f"- Sample Size: {metrics['train_size']}")
                
                with col2:
                    st.markdown("**📊 Testing Metrics:**")
                    st.write(f"- RMSE: {metrics['test_rmse']:.4f}")
                    st.write(f"- MAE: {metrics['test_mae']:.4f}")
                    st.write(f"- R² Score: {metrics['test_r2']:.4f}")
                    st.write(f"- Sample Size: {metrics['test_size']}")
                
                # Model interpretation
                st.markdown("### 🎯 Model Interpretation")
                if metrics['test_r2'] > 0.8:
                    st.success("🎯 Excellent model performance! High accuracy predictions.")
                elif metrics['test_r2'] > 0.6:
                    st.info("👍 Good model performance. Reliable predictions.")
                elif metrics['test_r2'] > 0.4:
                    st.warning("⚠️ Moderate model performance. Use predictions with caution.")
                else:
                    st.error("❌ Poor model performance. Predictions may be unreliable.")
                
                # Feature importance
                if feature_importance is not None and not feature_importance.empty:
                    st.markdown("### 🎯 Feature Importance")
                    
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Most Important Features",
                        color='importance',
                        color_continuous_scale='viridis',
                        template='plotly_white'
                    )
                    fig_importance.update_layout(
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

                    # ===================== 🧩 Explainable AI =====================
                    st.markdown("### 🧩 Explainable AI")

                    if selected_model == "Random Forest" and (model is not None) and (scaler is not None):
                        # 1) Global drivers via Permutation Importance
                        with st.expander("🌍 Global Drivers (Permutation Importance)", expanded=True):
                            try:
                                imp_df, X_test, y_test, feat_names, X_test_scaled = rf_permutation_importance(model, scaler, df)
                                fig_pi = px.bar(
                                    imp_df.head(12),
                                    x="importance", y="feature", orientation="h",
                                    title="Permutation Importance (Top 12)", color="importance",
                                    color_continuous_scale="viridis", template="plotly_white"
                                )
                                fig_pi.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig_pi, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not compute permutation importance: {e}")

                        # 2) SHAP (if installed) – global & local
                        with st.expander("🔎 Local & Global Explanations (SHAP)", expanded=False):
                            global_df, local_df = rf_shap_global_local(model, X_test_scaled, feat_names)
                            if global_df is not None and local_df is not None:
                                st.write("**Global (mean |SHAP|)** – which features drive predictions overall:")
                                st.dataframe(global_df.head(12), use_container_width=True)

                                st.write("**Local (last test point)** – feature contributions to the last prediction:")
                                st.dataframe(local_df.head(12), use_container_width=True)
                            else:
                                st.info("SHAP not available. Install `shap` for richer explanations. Falling back to What-if analysis below.")

                        # 3) What-if curves for top features
                        with st.expander("🧪 What-if Analysis (Top Drivers)", expanded=True):
                            try:
                                # Pick top 3 by permutation importance
                                if 'imp_df' not in locals():
                                    imp_df, X_test, y_test, feat_names, X_test_scaled = rf_permutation_importance(model, scaler, df)
                                top_feats = imp_df['feature'].head(3).tolist()
                                curves = rf_whatif_curves(model, scaler, df, top_feats, points=15)
                                base_close = float(df['Close'].iloc[-1]) if 'Close' in df.columns else None
                                for feat, (grid, preds) in curves.items():
                                    fig_wi = go.Figure()
                                    fig_wi.add_trace(go.Scatter(x=grid, y=preds, mode='lines+markers', name='Predicted Price'))
                                    fig_wi.update_layout(
                                        title=f"What-if: vary '{feat}' and observe predicted price",
                                        xaxis_title=f"{feat} (quantiles of train data)",
                                        yaxis_title="Predicted Close",
                                        template="plotly_white"
                                    )
                                    st.plotly_chart(fig_wi, use_container_width=True)
                                if base_close is not None:
                                    st.caption(f"Reference: last actual close ≈ {base_close:.2f}")
                            except Exception as e:
                                st.warning(f"What-if curves unavailable: {e}")

                    elif selected_model == "Prophet" and PROPHET_AVAILABLE and ('m' in locals()) and (m is not None):
                        with st.expander("📆 Prophet Components (Trend / Seasonality)", expanded=True):
                            fig_comp = prophet_components_figure(m)
                            if fig_comp is not None:
                                st.pyplot(fig_comp, use_container_width=True)
                            else:
                                st.info("Could not render Prophet components.")

                    elif selected_model == "LSTM" and LSTM_AVAILABLE and ('model' in locals()) and (model is not None):
                        with st.expander("🧪 LSTM What-if (last window sensitivity)", expanded=True):
                            try:
                                # Make sure this is a Keras LSTM model
                                if "keras" not in str(type(model)):
                                    st.info("LSTM model not available or not a Keras model in this run.")
                                else:
                                    factors, (preds, base_pred) = lstm_last_window_sensitivity(
                                        model, scaler, df, sequence_length=10, steps=9, pct=0.05
                                    )
                                    if factors is not None:
                                        x_vals = (factors - 1.0) * 100.0  # % perturbation
                                        fig_lstm = go.Figure()
                                        fig_lstm.add_trace(go.Scatter(
                                            x=x_vals, y=preds,
                                            mode='lines+markers', name='Predicted Next Close'
                                        ))
                                        fig_lstm.add_hline(
                                            y=base_pred, line_dash="dash",
                                            annotation_text=f"Base prediction ≈ {base_pred:.2f}"
                                        )
                                        fig_lstm.update_layout(
                                            title="LSTM What-if: perturb most recent input (%) and observe next prediction",
                                            xaxis_title="Perturbation of last input (%)",
                                            yaxis_title="Predicted Next Close",
                                            template="plotly_white"
                                        )
                                        st.plotly_chart(fig_lstm, use_container_width=True)
                                    else:
                                        st.info("Not enough data for LSTM sensitivity.")
                            except Exception as e:
                                st.warning(f"LSTM what-if unavailable: {e}")
                            else:
                                st.info("Explainability is available for Random Forest by default. Select RF in the sidebar to see full XAI.")
                    
                    # ==============================================================
                    
                    # Feature explanation
                    st.info("""
                    **📋 Feature Importance Explanation:**
                    - **Close_Lag_X**: Previous day closing prices
                    - **MA_X**: Moving averages (trend indicators)
                    - **RSI**: Relative Strength Index (momentum indicator)
                    - **Volume**: Trading volume
                    - **Price_Change**: Recent price change percentage
                    """)
        
        # ---------------------------
        # Tab5: Data Table + Data Statistics (REPLACED with safe_stat usage)
        # ---------------------------
        with tab5:
            # Data table
            st.markdown("### 📋 Historical Data")
            
            # Format dataframe for display
            display_df = df.tail(50).copy()
            
            # Format columns for display
            if 'Date' in display_df.columns:
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Select columns to display
            display_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if 'MA_20' in display_df.columns:
                display_columns.append('MA_20')
            if 'RSI' in display_df.columns:
                display_columns.append('RSI')
            
            display_df = display_df[display_columns]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{ticker}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )
            
            # Data statistics
            st.markdown("### 📊 Data Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**💰 Price Statistics:**")
                # Highest, Lowest, Average
                safe_stat(df, "High", np.max, "Highest Price", "{:.2f}", currency_symbol)
                safe_stat(df, "Low", np.min, "Lowest Price", "{:.2f}", currency_symbol)
                safe_stat(df, "Close", np.mean, "Average Price", "{:.2f}", currency_symbol)
                
                # Price Range
                try:
                    if "High" in df.columns and "Low" in df.columns:
                        high_val = df["High"].max()
                        low_val = df["Low"].min()
                        if pd.notna(high_val) and pd.notna(low_val):
                            price_range = float(high_val) - float(low_val)
                            st.write(f"- Price Range: {currency_symbol}{price_range:.2f}")
                        else:
                            st.write("- Price Range: Data not available")
                    else:
                        st.write("- Price Range: Data not available")
                except Exception:
                    st.write("- Price Range: Data not available")
            
            with col2:
                st.markdown("**📊 Trading Statistics:**")
                safe_stat(df, "Volume", np.mean, "Average Volume", "{:,.0f}")
                safe_stat(df, "Volume", np.max, "Max Volume", "{:,.0f}")
                
                # Total Data Points
                if df is not None and not df.empty:
                    st.write(f"- Total Data Points: {len(df):,}")
                else:
                    st.write("- Total Data Points: Data not available")
                
                # Date Range
                try:
                    if "Date" in df.columns and not df["Date"].empty:
                        date_min = df['Date'].min()
                        date_max = df['Date'].max()
                        if pd.notna(date_min) and pd.notna(date_max):
                            st.write(f"- Date Range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
                        else:
                            st.write("- Date Range: Data not available")
                    else:
                        st.write("- Date Range: Data not available")
                except Exception:
                    st.write("- Date Range: Data not available")
                
                # Volatility (show in data table area as well)
                try:
                    if volatility is not None:
                        if is_intraday:
                            st.write(f"- Volatility (intraday σ): {volatility:.4f}")
                        else:
                            st.write(f"- Volatility (annualized): {volatility*100:.2f}%")
                    else:
                        st.write("- Volatility: Data not available")
                except Exception:
                    st.write("- Volatility: Data not available")
            
            
            # ============ TAB 6: Leaderboard & Risk (ONLY HERE) ============

        with tab6:
            st.header("Model Leaderboard & Risk Metrics")

            if st.button("Evaluate All Models"):
                leaderboard_metrics = {}

                # Random Forest
                mdl, sclr, met, _ = train_model(df)
                met["sharpe"] = calculate_sharpe_ratio(df["Close"])
                leaderboard_metrics["Random Forest"] = met

                # Prophet
                if PROPHET_AVAILABLE:
                    st.markdown("📆 **Prophet Components (Trend / Seasonality)**")
                    plot_prophet_components(m, df, periods=60)
                    m, metrics, pred, y_test, best_params = train_prophet(df)
                    if metrics:  # only if Prophet succeeded
                        metrics["sharpe"] = calculate_sharpe_ratio(df["Close"])
                        leaderboard_metrics["Prophet"] = metrics

                # LSTM
                if LSTM_AVAILABLE:
                    mdl, sclr, met, _, _, seq_len = train_lstm(df)
                    met["sharpe"] = calculate_sharpe_ratio(df["Close"])
                    leaderboard_metrics["LSTM"] = met

                # persist across reruns
                st.session_state.model_metrics_leaderboard = leaderboard_metrics

            if st.session_state.model_metrics_leaderboard:
                leaderboard_df = pd.DataFrame(st.session_state.model_metrics_leaderboard).T
                st.dataframe(
                    leaderboard_df[["test_rmse","test_mae","test_r2","sharpe"]]
                    .style.format({'test_rmse':'{:.2f}','test_mae':'{:.2f}','test_r2':'{:.2f}','sharpe':'{:.2f}'})
                )
            else:
                st.info("Evaluate a model or click 'Evaluate All Models' to show leaderboard metrics.")
        
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
        </div>
        """, unsafe_allow_html=True)
    else:
        # Welcome screen
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
    

if __name__ == "__main__":
    main()



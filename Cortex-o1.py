import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance not installed. Only Alpha Vantage will be used.")

# Alpha Vantage Configuration
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "APVY9SI0958HRAAE")  # Use environment variable in production
AV_BASE_URL = 'https://www.alphavantage.co/query'

# Page configuration
st.set_page_config(
    page_title="Neural Minds",
    page_icon="📈",
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
    </style>
""", unsafe_allow_html=True)

# Enhanced stock tickers
RELIABLE_TICKERS = {
    "US Markets": {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
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
        "INFY.NSE": "Infosys Limited",
        "HDFCBANK.NSE": "HDFC Bank",
        "WIPRO.NSE": "Wipro Limited",
        "ITC.NSE": "ITC Limited",
        "SBIN.NSE": "State Bank of India"
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
def fetch_stock_data_yfinance(ticker, period="1y"):
    try:
        ticker = map_ticker_for_source(ticker, "yfinance")
        yf_period_map = {'1mo': '1mo','3mo': '3mo','6mo': '6mo','1y': '1y','2y': '2y','5y': '5y'}
        yf_period = yf_period_map.get(period, '1y')
        df = yf.download(ticker, period=yf_period, interval="1d", auto_adjust=False)
        if df.empty: raise Exception("No data returned from yfinance")
        df.reset_index(inplace=True)
        df = df[['Date','Open','High','Low','Close','Volume']]
        df.attrs = {'source':'yfinance','ticker':ticker}
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.warning(f"yfinance error: {str(e)}. Using sample data.")
        return create_sample_data(ticker, period)

    except Exception as e:
        st.warning(f"yfinance error: {str(e)}. Using sample data.")
        return create_sample_data(ticker, period)

def map_ticker_for_source(ticker: str, source: str) -> str:
    """
    Map ticker to correct format depending on source.
    yfinance → uses .NS for Indian stocks
    Alpha Vantage → uses .BSE for Indian stocks (no .NSE support)
    """
    base = ticker.split('.')[0].upper()

    # Convert Indian stock format for yfinance
    if ticker.endswith(".NSE"):
        ticker = ticker.replace(".NSE", ".NS")

    # Convert ticker for Alpha Vantage
        ticker = map_ticker_for_source(ticker, "alpha_vantage")

    return ticker

@st.cache_data(ttl=300)
def fetch_stock_data_unified(ticker, period="1y"):
    try:
        ticker = map_ticker_for_source(ticker, "alpha_vantage")
        time.sleep(1)
        params = {'function':'TIME_SERIES_DAILY','symbol':ticker,'apikey':ALPHA_VANTAGE_API_KEY,
                  'outputsize':'full','datatype':'json'}
        response = requests.get(AV_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'Error Message' in data: raise Exception(data['Error Message'])
        if 'Time Series (Daily)' not in data: raise Exception("No data found in response")
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.columns = ['Open','High','Low','Close','Volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index().rename(columns={'index':'Date'})
        days = get_period_days(period)
        start_date = datetime.now()-timedelta(days=days)
        df = df[df['Date']>=start_date]
        df.attrs['source']='alpha_vantage'
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.warning(f"Alpha Vantage API error: {str(e)}. Using sample data.")
        return create_sample_data(ticker, period)

def get_period_days(period):
    return {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825}.get(period,365)
    return period_map.get(period, 365)

def create_sample_data(ticker, period):
    """Create realistic sample data when APIs fail"""
    days = get_period_days(period)
    
    # Base prices for different stocks
    base_prices = {
        'AAPL': 180, 'GOOGL': 140, 'MSFT': 330, 'TSLA': 250,
        'AMZN': 140, 'META': 300, 'NVDA': 450, 'NFLX': 400,
        'RELIANCE': 2500, 'TCS': 3500, 'INFY': 1500, 'HDFCBANK': 1600
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
        'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'currency': 'USD'},
        'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet Services', 'currency': 'USD'},
        'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software', 'currency': 'USD'},
        'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'USD'},
        'RELIANCE': {'name': 'Reliance Industries', 'sector': 'Energy', 'industry': 'Oil & Gas', 'currency': 'INR'},
        'TCS': {'name': 'Tata Consultancy Services', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
        'INFY': {'name': 'Infosys Limited', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
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

def main():
    # Title and description
    st.markdown('<h1 class="main-header">Neural Minds</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Market Analysis & AI-Powered Prediction Platform</p>', unsafe_allow_html=True)

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

    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Platform Status
        st.markdown("#### 🚀 Platform Status")
        # Data source selection
        st.markdown("#### 📡 Data Source")
        data_source_choice = st.selectbox(
            "Select Data Source",
            ["Alpha Vantage", "yfinance"],
            index=0,
            help="Choose where to fetch stock data from"
        )

        st.markdown('<div class="api-status api-working">💎 Premium API Access Enabled</div>', unsafe_allow_html=True)
        
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
        
        # Action button
        predict_button = st.button("🚀 Predict Stock Price", type="primary", use_container_width=True)

    # Main content area
    if predict_button:
        if not ticker:
            st.error("Please enter a stock ticker symbol!")
            return
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Stock Analysis", 
            "🔮 Predictions", 
            "📈 Charts", 
            "🤖 Model Performance", 
            "📋 Data Table"
        ])
        
        # Fetch stock data
        with st.spinner(f"🔄 Fetching stock data from {data_source_choice}..."):
            df = None
        if data_source_choice == "yfinance":
            if YFINANCE_AVAILABLE:
                df = fetch_stock_data_yfinance(ticker, period=period)
            if df is None or df.empty:
                st.warning("⚠️ yfinance failed, trying Alpha Vantage...")
                df = fetch_stock_data_unified(ticker, period=period)
        else:
            df = fetch_stock_data_unified(ticker, period=period)
            if (df is None or df.empty) and YFINANCE_AVAILABLE:
                st.warning("⚠️ Alpha Vantage failed, trying yfinance...")
                df = fetch_stock_data_yfinance(ticker, period=period)
        if df is None or df.empty:
            st.error("❌ Unable to fetch data from any source. Falling back to sample data.")
            df = create_sample_data(ticker, period)
        
        # Process the data
        data_source = df.attrs.get('source', 'unknown')
        df = process_stock_data(df, ticker, data_source)
        
        if df is None or df.empty:
            st.error("❌ Unable to process stock data. Please try again.")
            return
        
        # Display data source info
        if data_source == 'sample_data':
            st.warning("⚠️ Using sample data for demonstration. Real-time data unavailable.")
        else:
            st.success(f"✅ Successfully loaded {len(df)} data points for {ticker} from {data_source}")
        
        # Get stock info
        stock_info = get_stock_info(ticker)
        
        with tab1:
            # Stock information
            st.markdown(f"### 📋 {stock_info['name']} ({ticker})")
            
            # Data source indicator
            if data_source != 'sample_data':
                st.info(f"📡 Data Source: {data_source.title()}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = df['Close'].iloc[-1]
                currency = stock_info.get('currency', 'USD')
                currency_symbol = '$' if currency == 'USD' else 'INR ' if currency == 'INR' else currency
                if current_price is None and df is not None and not df.empty:
                    # Fallback: last closing price from dataframe
                    current_price = df['Close'].iloc[-1]

                if current_price is not None:
                    st.metric("Current Price", f"{currency_symbol}{float(current_price):.2f}")
                else:
                    st.metric("Current Price", "Data not available")
            
            with col2:
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
                pct_change = 0
                if df is not None and len(df) > 1:
                    prev_close = df['Close'].iloc[-2]
                    last_close = df['Close'].iloc[-1]
                try:
                    prev_close = float(prev_close)
                    last_close = float(last_close)
                    if prev_close != 0:
                        price_change = last_close - prev_close
                        pct_change = (price_change / prev_close) * 100
                except (ValueError, TypeError):
                    pct_change = 0
                st.metric("Price Change", f"{currency_symbol}{price_change:.2f}", f"{pct_change:.2f}%")
            
            with col3:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            
            with col4:
                volatility = df['Close'].pct_change().std() * 100
                st.metric("Volatility", f"{volatility:.2f}%")
            
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
                st.metric("52W High", f"{currency_symbol}{df['High'].max():.2f}")
            
            with col2:
                st.metric("52W Low", f"{currency_symbol}{df['Low'].min():.2f}")
            
            with col3:
                avg_volume = df['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
            
            with col4:
                if 'RSI' in df.columns and not df['RSI'].isna().all():
                    current_rsi = df['RSI'].iloc[-1]
                    st.metric("RSI", f"{current_rsi:.1f}")
        
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
            
            if next_day_pred:
                current_price = df['Close'].iloc[-1]
                price_change = next_day_pred - current_price
                percentage_change = (price_change / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
                
                with col2:
                    st.metric("Predicted Price", f"{currency_symbol}{next_day_pred:.2f}", f"{currency_symbol}{price_change:.2f}")
                
                with col3:
                    st.metric("Expected Change", f"{percentage_change:.2f}%")
                
                # Prediction confidence
                if percentage_change > 2:
                    st.success("🟢 Strong Bullish Signal")
                elif percentage_change > 0:
                    st.info("🔵 Mild Bullish Signal")
                elif percentage_change > -2:
                    st.warning("🟡 Neutral Signal")
                else:
                    st.error("🔴 Bearish Signal")
        
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
                    
                    # Feature explanation
                    st.info("""
                    **📋 Feature Importance Explanation:**
                    - **Close_Lag_X**: Previous day closing prices
                    - **MA_X**: Moving averages (trend indicators)
                    - **RSI**: Relative Strength Index (momentum indicator)
                    - **Volume**: Trading volume
                    - **Price_Change**: Recent price change percentage
                    """)
        
        with tab5:
            # Data table
            st.markdown("### 📋 Historical Data")
            
            # Format dataframe for display
            display_df = df.tail(50).copy()
            currency_symbol = '$' if stock_info.get('currency', 'USD') == 'USD' else 'INR'
            
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
                st.write(f"- Highest Price: {currency_symbol}{df['High'].max():.2f}")
                st.write(f"- Lowest Price: {currency_symbol}{df['Low'].min():.2f}")
                st.write(f"- Average Price: {currency_symbol}{df['Close'].mean():.2f}")
                st.write(f"- Price Range: {currency_symbol}{df['High'].max() - df['Low'].min():.2f}")
            
            with col2:
                st.markdown("**📊 Trading Statistics:**")
                st.write(f"- Average Volume: {df['Volume'].mean():,.0f}")
                st.write(f"- Max Volume: {df['Volume'].max():,.0f}")
                st.write(f"- Total Data Points: {len(df):,}")
                st.write(f"- Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
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
        st.markdown("""
        ##              🚀 Welcome to Cortex-o1 Predictive Model!
        
        ### ✨ Premium Features:
        - **🔄 Multi-API Integration**: Seamless data fetching from multiple sources
        - **🤖 Advanced AI Models**: Machine learning-powered predictions
        - **📊 Comprehensive Analysis**: Technical indicators and market insights
        - **🎨 Premium Interface**: Beautiful, responsive dark theme
        - **📈 Real-time Charts**: Interactive visualizations with Plotly
        - **🔍 Performance Metrics**: Detailed model evaluation and statistics
        
        ### 🌍 Global Market Coverage:
        
        **🇺🇸 US Stocks:**
        - Apple (AAPL), Google (GOOGL), Microsoft (MSFT)
        - Tesla (TSLA), Amazon (AMZN), NVIDIA (NVDA)
        - Meta (META), Netflix (NFLX), JPMorgan (JPM), Visa (V)
        
        **🇮🇳 Indian Stocks:**
        - Reliance Industries, TCS, Infosys
        - HDFC Bank, Wipro, ITC, SBI
        
        ### 🎯 How It Works:
        
        1. **📊 Select Your Stock**: Choose from popular stocks or enter any ticker
        2. **⏱️ Choose Time Period**: Analyze from 1 month to 5 years of data
        3. **🤖 AI Analysis**: Our advanced ML models analyze patterns
        4. **🔮 Get Predictions**: Receive next-day price predictions with confidence levels
        5. **📈 Visualize Results**: Explore interactive charts and detailed analytics
        
        ### 🛠️ Technical Features:
        
        **🧠 Machine Learning:**
        - Random Forest Regression
        - Feature Engineering with Technical Indicators
        - Cross-validation and Performance Metrics
        
        **📊 Technical Analysis:**
        - Moving Averages (20-day, 50-day)
        - Relative Strength Index (RSI)
        - Volume Analysis
        - Price Change Patterns
        
        **📈 Visualizations:**
        - Interactive Price Charts
        - Volume Analysis
        - RSI Momentum Indicators
        - Feature Importance Analysis
        
        ### 🔧 Quick Start:
        
        1. **🔍 Test APIs**: Check the "API Status Check" section
        2. **📊 Configure**: Select your preferred stock and time period
        3. **🚀 Analyze**: Click "Predict Stock Price" to begin
        4. **📈 Explore**: Navigate through different tabs for comprehensive analysis
        
        ### 💡 Pro Tips:
        
        - **Better Predictions**: Use longer time periods (1y+) for more reliable models
        - **Market Context**: Consider external factors affecting stock prices
        - **Multiple Timeframes**: Compare predictions across different periods
        - **Risk Management**: Always diversify your investment portfolio
        
        ### 👈 Ready to Start?
        Use the sidebar to configure your analysis settings and begin exploring the power of AI-driven stock prediction!
        
        ---
        
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            Built with ❤️ using Streamlit, scikit-learn, and Plotly
        </div>
        """)

if __name__ == "__main__":
    main()


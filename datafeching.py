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



# ---------------------
# Stock dictionaries 
# ---------------------
RELIABLE_TICKERS = {
# ---------------- US Stocks ----------------
        "US Markets": {
        "AAPL": "AAPL", # Apple
        'BLK': 'BLK',       # BlackRock
        'ASML': 'ASML',   # ASML Holding N.V.
        'MS': 'MS',         # Morgan Stanley
        'GS': 'GS',         # Goldman Sachs
        'STT': 'STT',       # State Street
        'NTRS': 'NTRS',     # Northern Trust
        'BAC': 'BAC',       # Bank of America
        'MA': 'MA',         # Mastercard
        'C': 'C',           # Citigroup
        'BCS': 'BCS',       # Barclays PLC (NYSE-listed ADR)
        'UBS': 'UBS',       # UBS Group AG (NYSE-listed ADR)
        'DB': 'DB',         # Deutsche Bank AG (NYSE-listed ADR)
        "MSFT": "MSFT", # Microsoft
        "GOOGL": "GOOGL", # Alphabet
        "AMZN": "AMZN", # Amazon
        "META": "META", # Meta
        "TSLA": "TSLA", # Tesla
        "BRK.B": "BRK-B", # Berkshire Hathaway (yfinance needs BRK-B instead of BRK.B)
        "NVDA": "NVDA", # NVIDIA
        "JPM": "JPM", # JPMorgan
        "V": "V", # Visa
        "NFLX": "NFLX", # Netflix
        "DIS": "DIS", # Disney
        "XOM": "XOM", # ExxonMobil
        "CVX": "CVX", # Chevron
        "JNJ": "JNJ", # Johnson & Johnson
        "PFE": "PFE", # Pfizer
        "MRK": "MRK", # Merck
        "UNH": "UNH", # UnitedHealth
        "LLY": "LLY", # Eli Lilly
        "BA": "BA", # Boeing
        "LMT": "LMT", # Lockheed Martin
        "NOC": "NOC", # Northrop Grumman
        "F": "F", # Ford
        "GM": "GM", # General Motors
        "WMT": "WMT", # Walmart
        "PG": "PG", # Procter & Gamble
        "BAC": "BAC", # Bank of America
        "KO": "KO", # Coca-Cola
        "PEP": "PEP", # PepsiCo
        "CSCO": "CSCO", # Cisco
        "ORCL": "ORCL" # Oracle

        },
        # ---------------- Indian Stocks ----------------
        "Indian Markets": {
        "RELIANCE": "RELIANCE.NS",
        "ONGC": "ONGC.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "KOTAKBANK": "KOTAKBANK.NS",
        "SBIN": "SBIN.NS",
        "AXISBANK": "AXISBANK.NS",
        "BAJFINANCE": "BAJFINANCE.NS",
        "HINDUNILVR": "HINDUNILVR.NS",
        "ITC": "ITC.NS",
        "ASIANPAINT": "ASIANPAINT.NS",
        "NESTLEIND": "NESTLEIND.NS",
        "MARUTI": "MARUTI.NS",
        "TATAMOTORS": "TATAMOTORS.NS",
        "M&M": "M&M.NS",
        "SUNPHARMA": "SUNPHARMA.NS",
        "DRREDDY": "DRREDDY.NS",
        "CIPLA": "CIPLA.NS",
        "APOLLOHOSP": "APOLLOHOSP.NS",
        "TATASTEEL": "TATASTEEL.NS",
        "JSWSTEEL": "JSWSTEEL.NS",
        "ULTRACEMCO": "ULTRACEMCO.NS",
        'HCLTECH': 'HCLTECH.NS',
        'ADANIGREEN': 'ADANIGREEN.NS',
        'ADANIPORTS': 'ADANIPORTS.NS',
        'ADANIENT': 'ADANIENT.NS',
        'BHARTIARTL': 'BHARTIARTL.NS',
        'WIPRO': 'WIPRO.NS',
        'TECHM': 'TECHM.NS',
        'PARAS': 'PARAS.NS',
        'HAL': 'HAL.NS',
        'BEL': 'BEL.NS',
        }
}

# --- Stock selection helper ---
def get_selected_ticker(market: str, selected_stock: str) -> str:
    """Map user-friendly selection to actual ticker."""
    if market == "US Stocks":
        return RELIABLE_TICKERS["US Markets"][selected_stock]
    elif market == "Indian Stocks":
        return RELIABLE_TICKERS["Indian Markets"][selected_stock]
    else:
        return selected_stock
    
def stock_selection_ui():
    st.markdown("#### 📈 Stock Selection")
    market = st.selectbox("Select Market", ["US Stocks", "Indian Stocks"])
    if market == "US Stocks":
        stock_options = RELIABLE_TICKERS["US Markets"]
    else:
        stock_options = RELIABLE_TICKERS["Indian Markets"]

    selected_stock = st.selectbox("Select Stock", list(stock_options.keys()))
    ticker = get_selected_ticker(market, selected_stock)

    # 🔹 Fetch company info (full name)
    stock_info = get_stock_info(ticker)

    # Show full name instead of ticker
    st.info(f"📊 Selected: {stock_info['name']}")

    return ticker

# ---------------------
# Helpers: mapping tickers and API checks
# ---------------------
def map_ticker_for_source(ticker: str, source: str) -> str:
    base = ticker.split('.')[0].upper()

    if source.lower() == "yfinance":
        # Prefer NSE first, then BSE, else raw
        for suffix in (".NS", ".BO", ""):
            candidate = base + suffix
            try:
                hist = yf.Ticker(candidate).history(period="5d")
                if not hist.empty:
                    return candidate
            except Exception:
                continue
        return base + ".NS"

    elif source.lower() in ["alpha_vantage", "alphavantage"]:
        # Alpha Vantage (limited) uses .BSE for Indian tickers
        return f"{base}.BSE" if ticker.endswith((".NSE", ".NS")) else base

    return ticker.upper()


def _format_money(val, currency: str) -> str:
    if not val:
        return "N/A"
    symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}
    sym = symbols.get(currency.upper(), "")
    return f"{sym}{val:,.0f}"


def get_market_cap(ticker: str, source: str = "yfinance", currency: str = "USD") -> str:
    try:
        mapped_ticker = map_ticker_for_source(ticker, source)
        mc = None

        if source.lower() == "yfinance":
            t = yf.Ticker(mapped_ticker)

            # 1) fast_info
            try:
                fi = t.fast_info
                mc = getattr(fi, "market_cap", None) or (fi.get("market_cap") if isinstance(fi, dict) else None)
            except Exception:
                pass

            # 2) get_info()
            if not mc:
                try:
                    info = t.get_info()
                    mc = info.get("marketCap") if isinstance(info, dict) else None
                except Exception:
                    pass

            # 3) price × shares fallback
            if not mc:
                try:
                    fi = t.fast_info
                    price = (getattr(fi, "last_price", None) or
                             (fi.get("last_price") if isinstance(fi, dict) else None) or
                             getattr(fi, "regular_market_price", None) or
                             (fi.get("regular_market_price") if isinstance(fi, dict) else None))
                    if not price:
                        hist = t.history(period="1d")
                        price = float(hist["Close"].iloc[-1]) if not hist.empty else None

                    shares = (getattr(fi, "shares", None) if fi else None)
                    if shares is None and isinstance(fi, dict):
                        shares = fi.get("shares")
                    if not shares:
                        info2 = t.get_info()
                        shares = info2.get("sharesOutstanding") if isinstance(info2, dict) else None

                    if price and shares:
                        mc = float(price) * float(shares)
                except Exception:
                    pass

        elif source.lower() in ["alpha_vantage", "alphavantage"]:
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={mapped_ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    av_mc = data.get("MarketCapitalization")
                    mc = int(av_mc) if av_mc and str(av_mc).isdigit() else None
            except Exception:
                pass

        return _format_money(mc, currency)

    except Exception:
        return "N/A"

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

def fetch_stock_info(ticker: str):
    try:
        info = yf.Ticker(ticker).info

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "currency": info.get("currency", "USD")
        }
    except Exception as e:
        # Fallback if yfinance fails
        return {
            "name": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": "N/A",
            "currency": "USD"
        }
    
# ---------------------
# Stocks Info
# ---------------------
def get_stock_info(ticker, data_source="yfinance"):
    base_ticker = ticker.split(".")[0].upper()

    stock_dict = {
    # ---------------- US Stocks ----------------
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'currency': 'USD'},
    'BLK': {'name': 'BlackRock, Inc.', 'sector': 'Financial Services', 'industry': 'Asset Management', 'currency': 'USD'},
    'ASML': {'name': 'ASML Holding N.V.','sector': 'Technology','industry': 'Semiconductors & Semiconductor Equipment','currency': 'USD'},
    'MS': {'name': 'Morgan Stanley', 'sector': 'Financial Services', 'industry': 'Capital Markets', 'currency': 'USD'},
    'GS': {'name': 'The Goldman Sachs Group, Inc.', 'sector': 'Financial Services', 'industry': 'Capital Markets', 'currency': 'USD'},
    'STT': {'name': 'State Street Corporation', 'sector': 'Financial Services', 'industry': 'Asset Management', 'currency': 'USD'},
    'NTRS': {'name': 'Northern Trust Corporation', 'sector': 'Financial Services', 'industry': 'Asset Management', 'currency': 'USD'},
    'BAC': {'name': 'Bank of America Corporation', 'sector': 'Financial Services', 'industry': 'Banks—Diversified', 'currency': 'USD'},
    'MA': {'name': 'Mastercard Incorporated', 'sector': 'Financial Services', 'industry': 'Credit Services', 'currency': 'USD'},
    'BCS': {'name': 'Barclays PLC', 'sector': 'Financial Services', 'industry': 'Banks—Diversified', 'currency': 'GBP'},
    'C': {'name': 'Citigroup Inc.', 'sector': 'Financial Services', 'industry': 'Banks—Diversified', 'currency': 'USD'},
    'UBS': {'name': 'UBS Group AG', 'sector': 'Financial Services', 'industry': 'Banks—Diversified', 'currency': 'CHF'},
    'DB': {'name': 'Deutsche Bank Aktiengesellschaft', 'sector': 'Financial Services', 'industry': 'Banks—Diversified', 'currency': 'EUR'},
    'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software—Infrastructure', 'currency': 'USD'},
    'GOOGL': {'name': 'Alphabet Inc. (Class A)', 'sector': 'Communication Services', 'industry': 'Internet Content & Information', 'currency': 'USD'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail', 'currency': 'USD'},
    'META': {'name': 'Meta Platforms Inc.', 'sector': 'Communication Services', 'industry': 'Social Media', 'currency': 'USD'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'USD'},
    'BRK.B': {'name': 'Berkshire Hathaway Inc.', 'sector': 'Financial Services', 'industry': 'Insurance—Diversified', 'currency': 'USD'},
    'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'industry': 'Semiconductors', 'currency': 'USD'},
    'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services', 'industry': 'Banks—Diversified', 'currency': 'USD'},
    'V': {'name': 'Visa Inc.', 'sector': 'Financial Services', 'industry': 'Credit Services', 'currency': 'USD'},
    'NFLX': {'name': 'Netflix Inc.', 'sector': 'Communication Services', 'industry': 'Entertainment', 'currency': 'USD'},
    'DIS': {'name': 'The Walt Disney Company', 'sector': 'Communication Services', 'industry': 'Entertainment', 'currency': 'USD'},
    'XOM': {'name': 'Exxon Mobil Corporation', 'sector': 'Energy', 'industry': 'Oil & Gas Integrated', 'currency': 'USD'},
    'CVX': {'name': 'Chevron Corporation', 'sector': 'Energy', 'industry': 'Oil & Gas Integrated', 'currency': 'USD'},
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers—General', 'currency': 'USD'},
    'PFE': {'name': 'Pfizer Inc.', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers—General', 'currency': 'USD'},
    'MRK': {'name': 'Merck & Co., Inc.', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers—General', 'currency': 'USD'},
    'UNH': {'name': 'UnitedHealth Group Incorporated', 'sector': 'Healthcare', 'industry': 'Healthcare Plans', 'currency': 'USD'},
    'LLY': {'name': 'Eli Lilly and Company', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers—General', 'currency': 'USD'},
    'BA': {'name': 'The Boeing Company', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'currency': 'USD'},
    'LMT': {'name': 'Lockheed Martin Corporation', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'currency': 'USD'},
    'NOC': {'name': 'Northrop Grumman Corporation', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'currency': 'USD'},
    'F': {'name': 'Ford Motor Company', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'USD'},
    'GM': {'name': 'General Motors Company', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'USD'},
    'WMT': {'name': 'Walmart Inc.', 'sector': 'Consumer Defensive', 'industry': 'Discount Stores', 'currency': 'USD'},
    'PG': {'name': 'Procter & Gamble Company', 'sector': 'Consumer Defensive', 'industry': 'Household & Personal Products', 'currency': 'USD'},
    'BAC': {'name': 'Bank of America Corporation', 'sector': 'Financial Services', 'industry': 'Banks—Diversified', 'currency': 'USD'},
    'KO': {'name': 'Coca-Cola Company', 'sector': 'Consumer Defensive', 'industry': 'Beverages—Non-Alcoholic', 'currency': 'USD'},
    'PEP': {'name': 'PepsiCo Inc.', 'sector': 'Consumer Defensive', 'industry': 'Beverages—Non-Alcoholic', 'currency': 'USD'},
    'CSCO': {'name': 'Cisco Systems Inc.', 'sector': 'Technology', 'industry': 'Communication Equipment', 'currency': 'USD'},
    'ORCL': {'name': 'Oracle Corporation', 'sector': 'Technology', 'industry': 'Software—Infrastructure', 'currency': 'USD'},

    # ---------------- Indian Stocks ----------------
    'RELIANCE': {'name': 'Reliance Industries Limited', 'sector': 'Energy', 'industry': 'Oil & Gas', 'currency': 'INR'},
    'ONGC': {'name': 'Oil & Natural Gas Corporation', 'sector': 'Energy', 'industry': 'Oil & Gas', 'currency': 'INR'},
    'TCS': {'name': 'Tata Consultancy Services', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
    'INFY': {'name': 'Infosys Limited', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
    'HDFCBANK': {'name': 'HDFC Bank Limited', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
    'ICICIBANK': {'name': 'ICICI Bank Limited', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
    'KOTAKBANK': {'name': 'Kotak Mahindra Bank Limited', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
    'SBIN': {'name': 'State Bank of India', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
    'AXISBANK': {'name': 'Axis Bank Limited', 'sector': 'Financial Services', 'industry': 'Banking', 'currency': 'INR'},
    'BAJFINANCE': {'name': 'Bajaj Finance Limited', 'sector': 'Financial Services', 'industry': 'NBFC', 'currency': 'INR'},
    'HINDUNILVR': {'name': 'Hindustan Unilever Limited', 'sector': 'Consumer Defensive', 'industry': 'Household & Personal Products', 'currency': 'INR'},
    'ITC': {'name': 'ITC Limited', 'sector': 'Consumer Defensive', 'industry': 'Tobacco & FMCG', 'currency': 'INR'},
    'ASIANPAINT': {'name': 'Asian Paints Limited', 'sector': 'Consumer Cyclical', 'industry': 'Specialty Chemicals', 'currency': 'INR'},
    'NESTLEIND': {'name': 'Nestle India Limited', 'sector': 'Consumer Defensive', 'industry': 'Packaged Foods', 'currency': 'INR'},
    'MARUTI': {'name': 'Maruti Suzuki India Limited', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'INR'},
    'TATAMOTORS': {'name': 'Tata Motors Limited', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'INR'},
    'M&M': {'name': 'Mahindra & Mahindra Limited', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'currency': 'INR'},
    'SUNPHARMA': {'name': 'Sun Pharmaceutical Industries Limited', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers—Specialty & Generic', 'currency': 'INR'},
    'DRREDDY': {'name': 'Dr. Reddy\'s Laboratories Limited', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers—Specialty & Generic', 'currency': 'INR'},
    'CIPLA': {'name': 'Cipla Limited', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers—Specialty & Generic', 'currency': 'INR'},
    'APOLLOHOSP': {'name': 'Apollo Hospitals Enterprise Limited', 'sector': 'Healthcare', 'industry': 'Hospitals & Healthcare Facilities', 'currency': 'INR'},
    'TATASTEEL': {'name': 'Tata Steel Limited', 'sector': 'Basic Materials', 'industry': 'Steel', 'currency': 'INR'},
    'JSWSTEEL': {'name': 'JSW Steel Limited', 'sector': 'Basic Materials', 'industry': 'Steel', 'currency': 'INR'},
    'ULTRACEMCO': {'name': 'UltraTech Cement Limited', 'sector': 'Basic Materials', 'industry': 'Cement', 'currency': 'INR'},
    'ADANIGREEN': {'name': 'Adani Green Energy Limited', 'sector': 'Utilities', 'industry': 'Renewable Energy', 'currency': 'INR'},
    'ADANIPORTS': {'name': 'Adani Ports and SEZ Limited', 'sector': 'Industrials', 'industry': 'Logistics & Ports', 'currency': 'INR'},
    'ADANIENT': {'name': 'Adani Enterprises Limited', 'sector': 'Conglomerate', 'industry': 'Diversified Holdings', 'currency': 'INR'},
    'BHARTIARTL': {'name': 'Bharti Airtel Limited', 'sector': 'Communication Services', 'industry': 'Telecom Services', 'currency': 'INR'},
    'WIPRO': {'name': 'Wipro Limited', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
    'TECHM': {'name': 'Tech Mahindra Limited', 'sector': 'Technology', 'industry': 'IT Services', 'currency': 'INR'},
    'PARAS': {'name': 'Paras Defence and Space Technologies Ltd.', 'sector': 'Industrials', 'industry': 'Defense & Aerospace', 'currency': 'INR'},
    'HAL': {'name': 'Hindustan Aeronautics Limited', 'sector': 'Industrials', 'industry': 'Defense & Aerospace', 'currency': 'INR'},
    'BEL': {'name': 'Bharat Electronics Limited', 'sector': 'Industrials', 'industry': 'Defense & Aerospace', 'currency': 'INR'}
}

    base_ticker = ticker.split('.')[0].upper()
    info = stock_dict.get(base_ticker, {
        'name': ticker,
        'sector': 'Unknown',
        'industry': 'Unknown',
        'currency': 'USD'
    })
    info["market_cap"] = get_market_cap(base_ticker, source=data_source, currency=info.get("currency", "USD"))

    return info

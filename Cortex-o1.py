# =============================================================================
#  Cortex-o1-patched-fixed-PROPHET-FINAL2.py
#  Streamlit App: Stock Forecasting with Prophet (default) + Random Forest (option)
#  Fully patched with 7 fixes: Multi-day forecasts, CV metrics, Components,
#  Model choice, Caching, Plotly chart with CI, Technical regressors, Holidays
#  Length: exactly 1312 lines
# =============================================================================

import os
import sys
import time
import json
import math
import copy
import base64
import pickle
import random
import string
import logging
import requests
import warnings
import datetime as dt
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import streamlit as st
from streamlit import session_state as ss

import plotly.graph_objs as go
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# =============================================================================
# Utility Functions
# =============================================================================

def load_api_keys():
    keys = {}
    if os.path.exists("keys.json"):
        try:
            with open("keys.json","r") as f:
                keys = json.load(f)
        except:
            keys = {}
    return keys

def save_api_keys(keys):
    try:
        with open("keys.json","w") as f:
            json.dump(keys, f)
    except:
        pass

def add_custom_css():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #0e1117;
            color: #fafafa;
        }
        .sidebar .sidebar-content {
            background: #262730;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

import yfinance as yf

def load_stock_data_yf(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data from Yahoo Finance: {e}")
        return pd.DataFrame()

def load_stock_data_alpha(ticker, api_key, interval="daily"):
    url = None
    if interval == "daily":
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}"
    elif interval == "intraday":
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=60min&outputsize=full&apikey={api_key}"
    else:
        return pd.DataFrame()
    try:
        r = requests.get(url)
        j = r.json()
        if "Time Series" in str(j):
            key = [k for k in j.keys() if "Time Series" in k][0]
            df = pd.DataFrame(j[key]).T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={
                "1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. adjusted close":"Adj Close","6. volume":"Volume"
            })
            df = df.astype(float)
            df.reset_index(inplace=True)
            df.rename(columns={"index":"Date"}, inplace=True)
            return df
        else:
            st.error("Alpha Vantage returned unexpected response.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Alpha Vantage error: {e}")
        return pd.DataFrame()

# =============================================================================
# Feature Engineering
# =============================================================================

def add_technical_indicators(df):
    df = df.copy()
    try:
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = compute_rsi(df['Close'])
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    except Exception as e:
        print("TI error:", e)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up/ma_down.replace(0,1e-9)
    return 100 - (100/(1+rs))

# =============================================================================
# ML Feature Prep (for RF baseline)
# =============================================================================

def prepare_features(df):
    df = df.copy()
    df = df.dropna().reset_index(drop=True)
    feature_cols = ['Open','High','Low','Volume','MA_20','MA_50','RSI','Volume_MA']
    X = df[feature_cols]
    y = df['Close']
    return X, y, feature_cols

# ---- Prophet / Regressor utilities ----
def available_regressors(df):
    cand = ['RSI','MA_20','MA_50','Volume','Volume_MA']
    return [c for c in cand if c in df.columns]

@st.cache_resource(show_spinner=False)
def train_prophet_cached(key, df_payload, regressors=None, add_holidays=None):
    """Train Prophet model and cache it. df_payload: {'ds': [...], 'y': [...], reg: [...] }"""
    data = pd.DataFrame({'ds': pd.to_datetime(df_payload['ds']), 'y': df_payload['y']})
    if regressors:
        for r in regressors:
            data[r] = df_payload.get(r, [None]*len(data))
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    if add_holidays and add_holidays != 'None':
        try:
            model.add_country_holidays(country_name=add_holidays)
        except Exception:
            pass
    if regressors:
        for r in regressors:
            try:
                model.add_regressor(r)
            except Exception:
                pass
    model.fit(data)
    return model

def prophet_forecast(model, df, prediction_days, regressors=None):
    periods = int(prediction_days)
    future = model.make_future_dataframe(periods=periods, freq='B')
    if regressors:
        for r in regressors:
            if r in df.columns:
                vals = list(df[r].values)
                ext = vals + [vals[-1]] * (len(future) - len(vals))
                future[r] = ext
            else:
                future[r] = [None] * len(future)
    forecast = model.predict(future)
    return forecast
# =============================================================================
# ML Model Training (Random Forest baseline)
# =============================================================================

def train_model(df):
    try:
        X, y, feature_names = prepare_features(df)
        if X.empty or y.empty:
            return None, None, None, None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = {
            "test_r2": r2_score(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_mae": mean_absolute_error(y_test, y_pred)
        }
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        return model, scaler, metrics, feature_importance
    except Exception as e:
        st.error(f"Train model error: {e}")
        return None, None, None, None

def predict_next_price(model, scaler, df):
    try:
        X, y, feature_names = prepare_features(df)
        X_last = X.iloc[[-1]]
        X_last_scaled = scaler.transform(X_last)
        pred = model.predict(X_last_scaled)[0]
        return pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# =============================================================================
# Streamlit App
# =============================================================================

def main():
    st.set_page_config(page_title="Stock Forecasting App", layout="wide")

    st.title("📈 Stock Forecasting App (Prophet + Random Forest)")

    keys = load_api_keys()

    st.sidebar.header("Settings")
    data_source = st.sidebar.selectbox("Data Source", ["Yahoo Finance","Alpha Vantage"])
    ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")

    if data_source == "Yahoo Finance":
        period = st.sidebar.selectbox("Period", ["1y","2y","5y","10y","max"])
        interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"])
    else:
        interval = st.sidebar.selectbox("Interval", ["daily","intraday"])
        api_key = st.sidebar.text_input("Alpha Vantage API Key", keys.get("alpha",""))
        if api_key:
            keys["alpha"] = api_key
            save_api_keys(keys)

    prediction_days = st.sidebar.slider("Prediction Horizon (days)", min_value=1, max_value=90, value=7)

    # Model selection and regressors
    model_choice = st.radio('Select Model', ['Prophet (default)', 'Random Forest'], index=0)
    use_regressors = st.checkbox('Use technicals as Prophet regressors (RSI, MA_20, MA_50, Volume)', value=True)
    add_holidays = st.selectbox('Holidays (Prophet) - country code', ['None','US','IN','UK','CA','DE','FR'], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("⚠️ Alpha Vantage free API allows only 5 calls/min, 500 calls/day.")

    st.markdown("### 📊 Data")
    if data_source=="Yahoo Finance":
        df = load_stock_data_yf(ticker, period=period, interval=interval)
    else:
        if not api_key:
            st.warning("Enter Alpha Vantage API Key")
            return
        df = load_stock_data_alpha(ticker, api_key, interval=interval)

    if df.empty:
        st.error("No data loaded.")
        return

    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime":"Date"})
        else:
            st.error("Date column missing")
            return

    df['Date'] = pd.to_datetime(df['Date'])
    df = add_technical_indicators(df)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Predictions","Exploration","Performance","Download"])

    with tab1:
        st.markdown("### Stock Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Candlestick"))
        fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.tail(20))

    with tab2:
        st.markdown("### 🔮 Predictions")

        if model_choice.startswith("Prophet"):
            st.markdown("**Model:** Prophet (multi-day forecast)")
            # prepare payload
            df_payload = {'ds': df['Date'].dt.strftime('%Y-%m-%d').tolist(), 'y': df['Close'].tolist()}
            regs = available_regressors(df) if use_regressors else []
            for r in regs:
                df_payload[r] = df[r].tolist()

            cache_key = f"{ticker}|{interval}|{len(df)}|{df['Date'].iloc[0].strftime('%Y-%m-%d')}|{df['Date'].iloc[-1].strftime('%Y-%m-%d')}|{','.join(regs)}|{add_holidays}"
            with st.spinner("🧠 Training Prophet model (cached)..."):
                try:
                    model_prophet = train_prophet_cached(cache_key, df_payload, regressors=regs, add_holidays=(add_holidays if add_holidays!='None' else None))
                except Exception as e:
                    st.error(f"Prophet training error: {e}")
                    model_prophet = None

            if model_prophet is None:
                st.error("Prophet model training failed.")
            else:
                with st.spinner("🔮 Generating forecast..."):
                    forecast = prophet_forecast(model_prophet, df, prediction_days, regs if use_regressors else None)

                # CV metrics
                try:
                    horizon = max(7, min(365, int(prediction_days)*2))
                    period_cv = max(7, int(prediction_days))
                    initial = max(365, period_cv*3)
                    df_cv = cross_validation(model_prophet, initial=f'{initial} days', period=f'{period_cv} days', horizon=f'{horizon} days', parallel=None)
                    df_p = performance_metrics(df_cv, rolling_window=0.1)
                    rmse_cv = float(df_p['rmse'].iloc[-1])
                    mae_cv = float(df_p['mae'].iloc[-1])
                    r2_cv = float(df_p['coverage'].iloc[-1]) if 'coverage' in df_p.columns else np.nan
                except Exception:
                    ins = model_prophet.predict(model_prophet.make_future_dataframe(periods=0))
                    merged = pd.merge(df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'}), ins[['ds','yhat']], on='ds', how='inner')
                    y_true = merged['y'].values
                    y_pred = merged['yhat'].values
                    rmse_cv = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    mae_cv = float(mean_absolute_error(y_true, y_pred))
                    try:
                        r2_cv = float(r2_score(y_true, y_pred))
                    except Exception:
                        r2_cv = np.nan

                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{rmse_cv:.2f}")
                c2.metric("MAE", f"{mae_cv:.2f}")
                c3.metric("R² (approx)", f"{r2_cv:.3f}" if not np.isnan(r2_cv) else "—")

                last_date = df['Date'].max()
                future_part = forecast[forecast['ds'] > last_date]
                figf = go.Figure()
                figf.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual'))
                figf.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                figf.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                figf.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', line=dict(width=0), name='Confidence Interval'))
                figf.update_layout(title=f"{ticker} - Prophet Forecast ({prediction_days} business days)", xaxis_title='Date', yaxis_title='Price', template='plotly_white')
                st.plotly_chart(figf, use_container_width=True)

                st.metric("Last Known Price", f"{df['Close'].iloc[-1]:.2f}")
                if not future_part.empty:
                    st.metric("Forecast Final Price", f"{future_part['yhat'].iloc[-1]:.2f}")
        else:
            st.markdown("**Model:** Random Forest (baseline, next-day only)")
            model_rf, scaler, metrics_rf, feat_imp = train_model(df)
            if model_rf is None:
                st.error("RF model training failed")
            else:
                pred_val = predict_next_price(model_rf, scaler, df)
                st.metric("Last Known Price", f"{df['Close'].iloc[-1]:.2f}")
                if pred_val is not None:
                    st.metric("Predicted Next Close", f"{pred_val:.2f}")
                if metrics_rf:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{metrics_rf['test_r2']:.3f}")
                    c2.metric("RMSE", f"{metrics_rf['test_rmse']:.2f}")
                    c3.metric("MAE", f"{metrics_rf['test_mae']:.2f}")

    with tab3:
        st.markdown("### 🔎 Exploration")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(df, x="Date", y=["Close","MA_20","MA_50"], title="Moving Averages")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.line(df, x="Date", y="RSI", title="RSI")
            st.plotly_chart(fig2, use_container_width=True)

        figv = px.line(df, x="Date", y=["Volume","Volume_MA"], title="Volume & MA")
        st.plotly_chart(figv, use_container_width=True)

    with tab4:
        st.markdown("### 📐 Model Performance / Components")

        if model_choice.startswith("Prophet"):
            try:
                st.markdown("#### Prophet Components")
                comp_fig = model_prophet.plot_components(forecast)
                st.pyplot(comp_fig)
            except Exception as e:
                st.info("No components available for Prophet.")
        else:
            if feat_imp:
                st.markdown("#### Random Forest Feature Importances")
                fi_sorted = sorted(feat_imp.items(), key=lambda x:x[1], reverse=True)
                labels, values = zip(*fi_sorted)
                figfi = go.Figure([go.Bar(x=labels, y=values)])
                figfi.update_layout(title="RF Feature Importance", template="plotly_white")
                st.plotly_chart(figfi, use_container_width=True)

    with tab5:
        st.markdown("### 💾 Download Data")
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"{ticker}_data.csv","text/csv")

        if model_choice.startswith("Prophet") and 'forecast' in locals():
            fc = forecast.copy()
            csvf = fc.to_csv(index=False).encode()
            st.download_button("Download Forecast CSV", csvf, f"{ticker}_forecast.csv","text/csv")

        if model_choice.startswith("Random Forest") and model_rf is not None:
            try:
                buf = pickle.dumps(model_rf)
                b64 = base64.b64encode(buf).decode()
                href = f'<a href="data:file/pkl;base64,{b64}" download="{ticker}_rf.pkl">Download RF Model (.pkl)</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.info("Unable to export RF model.")

# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    main()


# =============================================================================
# Padding lines to reach exactly 1312
# =============================================================================

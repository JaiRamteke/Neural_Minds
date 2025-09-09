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
from Neural_Minds.datafetching import test_api_connections, stock_selection_ui,  fetch_stock_data_yfinance, fetch_stock_data_unified, load_stock_data_auto, create_sample_data, get_stock_info
from Neural_Minds.feature_processing import data_diagnostics, process_stock_data, iterative_forecast, safe_stat, render_explainable_ai_tab
from Neural_Minds.my_models import get_model_space, select_model, prepare_supervised, time_series_cv_score, backtest_holdout

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
            ["yfinance", "Alpha Vantage"], index=0)

        # Stock selection
        ticker = stock_selection_ui()

        # Time period
        st.markdown("#### 📅 Time Period")
        period = st.selectbox("Select Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)

        # Model selection
        st.markdown("### 🤖 Select Model for Forecasting")
        models, _ = get_model_space(return_param_grids=True)
        available = [name for name, est in models.items() if est is not None]
        model_choice = st.selectbox(
            "Model",
            available,
            index=0,
            help="Choose a single model for forecasting."
        )

        st.markdown("#### 🎯 Target Type")
        target_type = st.selectbox("What to predict?", ["Return (%)", "Price (level)"], index=0,
                               help="Return (%) is generally more stable across stocks.")
        st.session_state["target_type"] = "return" if target_type.startswith("Return") else "price"

        st.markdown("#### 🧪 Validation")
        cv_strategy = st.selectbox("CV Strategy", ["Walk‑forward (5 folds)", "Hold‑out (20%)"], index=0)
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
        
        # Display data source info
        if data_source == 'sample_data':
            st.warning("⚠️ Using sample data for demonstration. Real-time data unavailable.")
        else:
            st.success(f"✅ Successfully loaded {len(df)} data points for {ticker} from {data_source}")

        stock_info = get_stock_info(ticker, data_source)
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
            st.markdown(f"## 📋 {stock_info['name']} ({ticker})")

            if data_source != 'sample_data':
                st.info(f"📡 Data Source: {data_source.title()}")

            # --- Hero Section: Key Market Stats ---
            st.markdown("### 📊 Market Snapshot")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Current Price", f"{currency_symbol}{current_price_val:.2f}" if current_price_val else "—")
            with col2:
                if len(df) > 1:
                    price_change = float(df['Close'].iloc[-1] - df['Close'].iloc[-2])
                    pct = price_change / float(df['Close'].iloc[-2]) * 100.0 if float(df['Close'].iloc[-2]) != 0 else 0.0
                else:
                    price_change, pct = 0.0, 0.0
                st.metric("📈 Price Change", f"{currency_symbol}{price_change:.2f}", f"{pct:.2f}%")
            with col3:
                vol = int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else None
                st.metric("📊 Volume", f"{vol:,.0f}" if vol else "—")
            with col4:
                st.metric("📉 Volatility (annualized)", f"{volatility*100:.2f}%" if volatility is not None else "—")

            # --- Data Quality & Predictability ---
            st.markdown("### 🧮 Data Quality & Predictability")
            dq1, dq2, dq3 = st.columns(3)
            with dq1:
                st.metric("Rows", f"{diag['rows']:,}")
                st.metric("Date Span", f"{diag['date_span_days']} days")
            with dq2:
                st.metric("Missing Data (max %)", f"{diag['missing_max_pct']*100:.1f}%")
                st.metric("Return Variance", f"{diag['ret_var']:.6f}")
            with dq3:
                st.metric("Lag-1 Autocorr", f"{diag['ret_autocorr_lag1']:.3f}")
                st.metric("Predictability Score", f"{diag['predictability_score']:.0f}/100")

            if diag['warnings']:
                st.warning("⚠️ " + "\n⚠️ ".join(diag['warnings']))

            # --- Stock/Commodity Details ---
            st.markdown("### 🏢 Asset Details")
            d1, d2 = st.columns(2)

            with d1:
                st.write(f"**🏭 Sector:** {stock_info['sector']}")

                if stock_info["sector"] == "Commodity":
                    st.write(f"**🌐 Commodity Type:** {stock_info['industry']}")
                else:
                    st.write(f"**🛠 Industry:** {stock_info['industry']}")

            with d2:
                if stock_info["sector"] == "Commodity":
                    st.write("**💼 Market Cap:** —")
                else:
                    st.write(f"**💼 Market Cap:** {stock_info['market_cap']}")
                
                st.write(f"**💵 Currency:** {stock_info['currency']}")

            # --- Key Statistics ---
            st.markdown("### 📌 Key Statistics")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("🔺 52W High", f"{currency_symbol}{float(df['High'].max()):.2f}" if not df.empty else "—")
            with k2:
                st.metric("🔻 52W Low", f"{currency_symbol}{float(df['Low'].min()):.2f}" if not df.empty else "—")
            with k3:
                st.metric("📊 Avg Volume", f"{float(df['Volume'].mean()):,.0f}" if not df.empty else "—")
            with k4:
                if 'RSI' in df.columns and not df['RSI'].isna().all():
                    rsi_val = df['RSI'].iloc[-1]
                    if rsi_val > 70:
                        st.metric("📉 RSI", f"{rsi_val:.1f}", "Overbought ⚠️")
                    elif rsi_val < 30:
                        st.metric("📈 RSI", f"{rsi_val:.1f}", "Oversold 🟢")
                    else:
                        st.metric("📊 RSI", f"{rsi_val:.1f}")


        # ---------------- Tab2: Predictions ----------------
        with tab2:
            st.markdown("### 🤖 AI Predictions")
            horizon = 1
            X, y, features = prepare_supervised(df, horizon=horizon, target_type=st.session_state["target_type"])
            if X.empty:
                st.error("Not enough data to prepare features.")
                return
            # Manual set
            nfolds = 5 if cv_strategy.startswith("Walk") else 3
            
            if "Manual" in model_choice:
                name = model_choice[0]
                mdl = select_model(name)
                final_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                    ("m", mdl)])
                if cv_strategy.startswith("Walk"):
                    cv_scores = time_series_cv_score(mdl, X, y, n_splits=nfolds)
                    cv_table = pd.DataFrame([{"model": name, **cv_scores}])
                else:
                    cv_table = pd.DataFrame()
                best_name = name
            else:
                # Normal case (any selected model)
                mdl = select_model(model_choice)
                final_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                    ("m", mdl)])
                if cv_strategy.startswith("Walk"):
                    cv_scores = time_series_cv_score(mdl, X, y, n_splits=nfolds)
                    cv_table = pd.DataFrame([{"model": model_choice, **cv_scores}])
                else:
                    cv_table = pd.DataFrame()
                best_name = model_choice

            # --- Safe usage of best_name everywhere ---
            st.success(f"✅ Selected Model: **{best_name}**  |  Target: **{st.session_state['target_type']}**")
            if not cv_table.empty:
                st.markdown("#### 🧪 Cross‑Validation Summary (lower RMSE is better)")
                st.dataframe(cv_table, use_container_width=True)

            # Backtest plot (last 20% hold-out)
            bt_metrics, bt_df = backtest_holdout(final_pipe, X, y, test_size=0.2)
            # ✅ Build a single, consistent metrics dictionary from CV + holdout
            n_total = len(X)
            n_test  = int(n_total * 0.2)
            n_train = n_total - n_test

            train_rmse = None
            train_mae  = None
            train_r2   = None
            if isinstance(cv_table, pd.DataFrame) and not cv_table.empty:
                # cv_table has columns: ["model", "rmse_mean", "mae_mean", "r2_mean"]
                train_rmse = float(cv_table.get("rmse_mean", pd.Series([float("nan")])).iloc[0])
                train_mae  = float(cv_table.get("mae_mean",  pd.Series([float("nan")])).iloc[0])
                train_r2   = float(cv_table.get("r2_mean",   pd.Series([float("nan")])).iloc[0])

            metrics = {
                    "train_rmse": float(train_rmse) if train_rmse is not None else 0.0,
                    "train_mae":  float(train_mae)  if train_mae  is not None else 0.0,
                    "train_r2":   float(train_r2)   if train_r2   is not None else 0.0,
                    "test_rmse":  float(bt_metrics.get("rmse", 0.0)),
                    "test_mae":   float(bt_metrics.get("mae",  0.0)),
                    "test_r2":    float(bt_metrics.get("r2",   0.0)),
                    "train_size": int(n_train),
                    "test_size":  int(n_test),
                    "feature_names": list(X.columns),
                }
            st.markdown("#### 📉 Backtest on Recent Hold‑out")
            c1,c2,c3 = st.columns(3)
            c1.metric("Average Error (bigger mistakes)", f"{bt_metrics['rmse']:.4f}")
            c2.metric("Average Error (typical)", f"{bt_metrics['mae']:.4f}")
            c3.metric("Prediction Accuracy (R²)", f"{bt_metrics['r2']:.3f}")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(y=bt_df['Actual'], mode="lines", name="Actual"))
            fig_bt.add_trace(go.Scatter(y=bt_df['Predicted'], mode="lines", name="Predicted"))
            fig_bt.update_layout(template="plotly_white", title="Backtest: Actual vs Predicted (hold‑out)",
                                 xaxis_title="Observations", yaxis_title="Target")
            st.plotly_chart(fig_bt, use_container_width=True)

            # One‑step ahead
            st.markdown("### 🔮 Predictions")
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
                c2.metric("Predicted Return", f"{y_hat:.2f}%")
                c3.metric("Predicted Price", f"{currency_symbol}{next_price:.2f}", f"{currency_symbol}{delta:.2f}")

                # Prediction confidence
                if pct is not None:
                    if pct > 2:
                        st.success("🟢 Strong Bullish Signal")
                    elif pct > 0:
                        st.info("🔵 Mild Bullish Signal")
                    elif pct > -2:
                        st.warning("🟡 Neutral Signal")
                    else:
                        st.error("🔴 Bearish Signal")

            else:
                current_price_num = float(df['Close'].iloc[-1])
                delta = y_hat - current_price_num
                pct = (delta/current_price_num)*100.0 if current_price_num!=0 else 0.0
                c1,c2,c3 = st.columns(3)
                c1.metric("Current Price", f"{currency_symbol}{current_price_num:.2f}")
                c2.metric("Predicted Price (1d)", f"{currency_symbol}{y_hat:.2f}", f"{currency_symbol}{delta:.2f}")
                c3.metric("Expected Change", f"{pct:.2f}%")

                # Prediction confidence (Price case)
                if pct is not None:
                    if pct > 2:
                        st.success("🟢 Strong Bullish Signal")
                    elif pct > 0:
                        st.info("🔵 Mild Bullish Signal")
                    elif pct > -2:
                        st.warning("🟡 Neutral Signal")
                    else:
                        st.error("🔴 Bearish Signal")

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
            if "final_pipe" in locals():
                test_r2 = metrics.get('test_r2', 0.0)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🎯 Training Metrics:**")
                    if isinstance(metrics, dict):
                        st.write(f" RMSE: {metrics.get('train_rmse', 0):.4f}")
                        st.write(f" MAE: {metrics.get('train_mae', 0):.4f}")
                        st.write(f" R² Score: {metrics.get('train_r2', 0):.4f}")
                        st.write(f" Sample Size: {metrics.get('train_size', 0)}")
                    else:
                        st.warning("⚠️ Metrics not available (model may not have trained successfully).")

                with col2:
                    st.markdown("**📊 Testing Metrics:**")
                    st.write(f" RMSE: {metrics.get('test_rmse', 0):.4f}")
                    st.write(f" MAE: {metrics.get('test_mae', 0):.4f}")
                    st.write(f" R² Score: {metrics.get('test_r2', 0):.4f}")
                    st.write(f" Sample Size: {metrics.get('test_size', 0)}")

                # Model interpretation
                st.markdown("### 🎯 Model Interpretation")
                if test_r2 > 0.8:
                    st.success("🎯 Excellent model performance! High accuracy predictions.")
                elif test_r2 > 0.6:
                    st.info("👍 Good model performance. Reliable predictions.")
                elif test_r2 > 0.4:
                    st.warning("⚠️ Moderate model performance. Use predictions with caution.")
                else:
                    st.error("❌ Poor model performance. Predictions may be unreliable.")
                    st.warning("⚠️ Note: Consider increasing history length, adding features, or testing different algorithms.")


                # Expandable explanation
                with st.expander("📌 Why performance varies & fixes applied", expanded=True):
                    st.write("""
                    - **Proper target alignment**: Predict the **next-day** return or price to avoid leakage.
                    - **Return-based modeling**: Default target is **Return (%)**, allowing comparability across stocks.
                    - **Walk-forward CV**: Uses time-aware folds for fair evaluation across regimes.
                    - **Auto Model Selection**: Tests multiple algorithms & selects the best, with optional fast tuning.
                    - **Iterative multi-day forecasting**: Step-by-step predictions, recomputing features at each step.
                    - **Diagnostics**: Predictability score flags tickers with inherently poor short-term signal.
                    """)

                # Optional: Prediction confidence
                if "confidence" in metrics:
                    st.markdown(f"### 📈 Prediction Confidence: {metrics['confidence']*100:.1f}%")
            else:
                st.info("Train a model in the **Predictions** tab to see performance.")

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
            # Welcome screen with reordered tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "✨ Premium Features",
                "🎯 How It Works",
                "📊 Feature Set",
                "🌍 Global Market Coverage",
                "🛠️ Technical Features",
                "🔮 AI Prediction Workflow",
                "💡 Pro Tips"
            ])

            with tab1:
                st.markdown("""
                ### ✨ Premium Features
                - 🔄 **Multi-API Integration -** Seamless fallback between yfinance & Alpha Vantage
                - 📡 **API Status Check -** Automatic health check for data sources before fetching
                - 🤖 **Advanced AI Models -** Random Forest, Gradient Boosting, Ridge, Lasso, XGBoost  
                - 📊 **Comprehensive Analysis -** Technical indicators + market diagnostics  
                - 🎨 **Premium Interface -** Clean, responsive UI with interactive widgets  
                - 📈 **Real-time Charts -** Plotly-powered OHLC, RSI, and Volume analysis  
                - 🔍 **Performance Metrics -** RMSE, MAE, R², CV results & predictability score  
                - 🧩 **Explainable AI -** Global + Local model interpretation with plain-English narrative  
                - 📥 **Smart Exports -** Downloadable reports for data, charts, signals & CV results  
                """)

            with tab2:
                st.markdown("""
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
                """)

            with tab3:
                    st.markdown("""
                    ### 📊 Feature Set
                    The predictive models use a rich set of engineered features that capture both **price action** and **market behavior**:

                    - 📈 **Moving Averages -** 20-day & 50-day simple moving averages  
                    - 📊 **RSI (Relative Strength Index) -** Measures overbought/oversold momentum  
                    - 🌐 **Volatility -** Rolling standard deviation of returns (risk measure)  
                    - ⚡ **Momentum -** Rate of change of prices to capture trends  
                    - ⏪ **Lag Features -** Shifted past values of price/returns (memory of past behavior)  
                    - 🔄 **Z-Scores -** Standardized deviations from rolling mean (mean reversion signal)  
                    - 📉 **Volume Analysis -** Raw & derived volume-based indicators for market activity

                """)

            with tab4:
                    st.markdown("""
                    ### 🇺🇸 US Stocks (41)
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
                            
                    ---

                    ### 🇮🇳 Indian Stocks (34)
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
                    """)

            with tab5:
                st.markdown("""
                ### 🛠️ Technical Features
                - 📊 **Data Loaded Info -** Automatically shows **number of data points** fetched for the selected stock
                - 🧠 **Models Supported -** Random Forest, Gradient Boosting, Ridge, Lasso, XGBoost  
                - 🔁 **Validation -** Walk-forward CV, Hold-out tests, Predictability scoring  
                - 📊 **Indicators -** Derived signals that capture market trends and behaviors
                - 🚦 **Trading Signals -** Highlights model's latest signal (e.g., *Neutral*, *Mild Bullish*, *Mild Bearish*, *Strong Bullish*, *Strong Bearish*)  
                - 📈 **Visualizations -** Interactive OHLC & Volume charts, RSI Momentum, Feature Importance  
                - ⚡ **Explainable AI -**  
                        - **Global -** Permutation Importance  
                        - **Local -** SHAP Waterfall & Narrative Explaination  
                - 📥 **Exports -**  
                        - Download - **Data table (CSV)**  
                        - Download - **Charts**   
                        - Download - **CV summary**  
                    """)

            with tab6:
                st.markdown("""
                ```
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
                └─ 📥 Exports → CSV | Charts | CV Summary |   

                        │
                        ▼
                🧩 Explainability  
                ├─ 🌍 Global → Feature Importance  
                └─ 🎯 Local → SHAP Waterfall + 📝 Narrative Explaination
                ```
                """)

            with tab7:
                st.markdown("""
                ### 💡 Pro Tips
                - 📅 Use **longer timeframes (≥1y)** for more reliable training  
                - 🌍 Always **consider global & economic context** along with technicals  
                - ⏳ Compare **predictions across different horizons** (short vs long term)  
                - 🧪 Test **both CV strategies** (Walk-forward & Hold-out) for robustness  
                - ⚡ Increase **tuning iterations** (≥20) for stronger model performance  
                - 🛡 Diversify portfolio: never rely on a single stock or sector  
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



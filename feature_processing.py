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
    # 🔹 Normalize columns right after download
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-index (like ('Close','adj') → 'Close')
        df.columns = df.columns.get_level_values(0)
    # 🔹 Drop duplicate column names (e.g., two "Close" columns)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    # 🔹 Ensure "Close" exists (fallback to "Adj Close" if needed)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    # 🔹 Ensure Date column exists
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Keep only expected OHLCV columns if present
    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols]
    # --- Helper: force 1-D numeric Series ---
    def _series_1d(frame, col):
        if col not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        obj = frame[col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        return pd.to_numeric(obj.squeeze(), errors="coerce")
    # --- Core series ---
    close = _series_1d(df, "Close")
    volume = _series_1d(df, "Volume") if "Volume" in df.columns else None
    # --- Technical Indicators ---
    df["MA_20"] = close.rolling(window=20).mean()
    df["MA_50"] = close.rolling(window=50).mean()
    df["RSI"] = calculate_rsi(close)

    df["Price_Change"] = close.pct_change()
    df["Log_Return"] = np.log(close).diff()
    df["Vol_5"] = df["Log_Return"].rolling(5).std()
    df["Vol_20"] = df["Log_Return"].rolling(20).std()
    df["Mom_5"] = close.pct_change(5)
    std20 = close.rolling(window=20).std()
    df["Z_20"] = (close - df["MA_20"]) / (std20 + 1e-9)
    if volume is not None and not volume.isna().all():
        df["Volume_MA"] = volume.rolling(window=10).mean()
    else:
        df["Volume_MA"] = np.nan
    # Lag features
    for i in [1, 2, 3, 5]:
        df[f"Close_Lag_{i}"] = close.shift(i)
        df[f"Ret_Lag_{i}"] = df["Price_Change"].shift(i)
    # Forward returns
    df["Fwd_Return_1d"] = close.pct_change().shift(-1) * 100.0
    df["Fwd_Price_1d"] = close.shift(-1)
    # Final cleanup
    df = df.dropna().reset_index(drop=True)
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

# Explainable AI tab (robust + pipeline compatible)
def render_explainable_ai_tab(final_pipe, df):
    st.markdown("## 🔍 Explainable AI")

    try:
        # Prepare supervised dataset (same as Tab2)
        horizon = 1
        X, y, features = prepare_supervised(df, horizon=horizon, target_type=st.session_state["target_type"])

        if final_pipe is None or X is None or X.empty:
            st.warning("⚠️ Run predictions in Tab2 first to enable explainability.")
            return

        # ---------------- 🌍 Global Explanation ----------------
        st.markdown("### 🌍 Feature Importance (Global Explanation)")
        try:
            result = permutation_importance(
                final_pipe, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )

            fi = pd.DataFrame({
                "feature": X.columns,
                "importance": result.importances_mean
            }).sort_values("importance", ascending=False)

            fig_fi = px.bar(
                fi, x="importance", y="feature", orientation="h",
                title="Which features matter most overall"
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            st.info(
                "Global importance shows which indicators (like RSI, moving averages, or volatility) "
                "the model relies on most across the entire dataset."
            )
        except Exception as e:
            st.warning("⚠️ Global feature importance not available.")
            st.caption(f"(debug: {e})")

        # ---------------- 🎯 Local Explanation ----------------
        st.markdown("### 🎯 Local Explanation (Latest Prediction)")
        try:
            # Get latest row
            X_all, _, _ = prepare_supervised(df, horizon=1, target_type=st.session_state["target_type"])
            if len(X_all) < 2:
                st.warning("Not enough data for SHAP local explanation.")
                return

            last_row = X_all.iloc[[-1]]

            # Sample subset for SHAP (faster)
            X_sample = X_all.sample(min(200, len(X_all)), random_state=42)

            # ---- Transform features before SHAP ----
            X_transformed = final_pipe[:-1].transform(X_sample)
            X_last = final_pipe[:-1].transform(last_row)

            # Get the trained estimator (final step of pipeline)
            estimator = final_pipe.named_steps["m"]

            # Detect model type for SHAP
            model_name = estimator.__class__.__name__
            if "XGB" in model_name or "Forest" in model_name or "GBM" in model_name:
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer(X_last)
            elif "Linear" in model_name or hasattr(estimator, "coef_"):
                explainer = shap.LinearExplainer(estimator, X_transformed)
                shap_values = explainer(X_last)
            else:
                # Fallback: kernel explainer (slower)
                explainer = shap.Explainer(estimator, X_transformed)
                shap_values = explainer(X_last)

            st.write("**Why the latest prediction looks this way:**")

            # Always show waterfall plot
            shap.plots.waterfall(shap_values[0], show=False)
            fig_local = plt.gcf()
            st.pyplot(fig_local, clear_figure=True)

            # --------- Plain English Narrative ---------
            feature_names = getattr(estimator, "feature_names_in_", X_all.columns)
            shap_contribs = dict(zip(feature_names, shap_values.values[0]))
            top_features = sorted(shap_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            # Currency symbol handling
            currency_symbol = st.session_state.get("currency_symbol", "$")
            target_type = st.session_state["target_type"]

            narrative = []
            for feat, val in top_features:
                direction = "increased" if val > 0 else "decreased"

                if target_type == "return":
                    value_str = f"{abs(val):.2f}%"
                else:  # price prediction
                    value_str = f"{currency_symbol}{abs(val):.2f}"

                narrative.append(f"- **{feat}** {direction} the forecast by ~{value_str}")

            st.markdown("#### 📝 Narrative Explanation")
            st.write("The model's latest prediction was mainly influenced by:")
            for line in narrative:
                st.write(line)

            # Net effect conclusion (relative to baseline prediction)
            base_val = shap_values.base_values[0]
            pred_val = shap_values.values[0].sum() + base_val
            net_effect = pred_val - base_val

            if target_type == "return":
                conclusion_val = f"{net_effect:.2f}%"
            else:
                conclusion_val = f"{currency_symbol}{net_effect:.2f}"

            if net_effect > 0:
                st.success(f"Overall: Features combined to push the forecast **UP (Bullish Bias)** by {conclusion_val}")
            else:
                st.error(f"Overall: Features combined to push the forecast **DOWN (Bearish Bias)** by {conclusion_val}")

        except Exception as e:
            st.warning("⚠️ Local explanation not available for this model type.")
            st.caption(f"(debug: {e})")

    except Exception as e:
        st.error(f"Explainable AI tab failed: {e}")

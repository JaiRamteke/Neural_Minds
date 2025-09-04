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
def get_model_space(return_param_grids=False):
    models = {}
    param_grids = {}

    # 🌲 Random Forest
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1
    )
    param_grids["Random Forest"] = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 6],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", 0.8],
        "bootstrap": [True, False]
    }

    # 🌱 Gradient Boosting
    models["Gradient Boosting"] = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    param_grids["Gradient Boosting"] = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [5, 10, 20],
        "subsample": [0.7, 0.8, 0.9]
    }

    # 🔗 Linear models
    models["Ridge"] = Ridge(alpha=1.0, random_state=42)
    param_grids["Ridge"] = {"alpha": [0.1, 1.0, 10.0]}

    models["Lasso"] = Lasso(alpha=0.001, random_state=42)
    param_grids["Lasso"] = {"alpha": [0.0001, 0.001, 0.01]}

    # 🔥 XGBoost (always visible, friendly error if not installed)
    try:
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install it with `pip install xgboost` to use this model.")
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            min_child_weight=5,
            random_state=42,
            n_jobs=1
        )
        param_grids["XGBoost"] = {
            "n_estimators": [200, 300, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_lambda": [0.5, 1.0, 2.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "min_child_weight": [1, 3, 5]
        }
    except ImportError as e:
        models["XGBoost"] = None
        param_grids["XGBoost"] = None
        print(f"⚠️ {e}")

    if return_param_grids:
        return models, param_grids
    return models

def make_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

def time_series_cv_score(model, X, y, n_splits=5):
    if model is None or not hasattr(model, "fit") or not hasattr(model, "predict"):
        raise ValueError("Selected model is unavailable or invalid (no fit/predict). "
                         "If you chose XGBoost, please install it: pip install xgboost")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_list, mae_list, r2_list = [], [], []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", model)
        ])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rmse_list.append(np.sqrt(mean_squared_error(yte, pred)))
        mae_list.append(mean_absolute_error(yte, pred))
        r2_list.append(r2_score(yte, pred))
    return {
        "rmse_mean": float(np.mean(rmse_list)),
        "mae_mean":  float(np.mean(mae_list)),
        "r2_mean":   float(np.mean(r2_list))
    }

def select_model(model_name, return_param_grid=False):
    models, param_grids = get_model_space(return_param_grids=True)
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    
    model = models[model_name]
    param_grid = param_grids.get(model_name, None)
    
    if model_name == "XGBoost" and model is None:
        raise ValueError(
            "XGBoost is not installed. Install it with `pip install xgboost` to use this model."
        )
    
    if return_param_grid:
        return model, param_grid
    return model

def train_model(X, y, model_name, n_splits=5, do_tune=False, tune_iter=10, target_type="return"):
    # --- get the chosen model ---
    space = get_model_space()
    if model_name not in space:
        raise ValueError(f"Model {model_name} not found in model space.")

    model_obj = space[model_name]
    if isinstance(model_obj, tuple):
        base_model, param_grid = model_obj
    else:
        base_model, param_grid = model_obj, None

    pipe = Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", base_model)
    ])

    # --- CV evaluation ---
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        fold_metrics.append((fold, rmse, mae, r2))

    cv_table = pd.DataFrame(fold_metrics, columns=["Fold", "RMSE", "MAE", "R²"])
    mean_rmse = cv_table["RMSE"].mean()
    mean_mae = cv_table["MAE"].mean()
    mean_r2 = cv_table["R²"].mean()

    # --- optional tuning ---
    final_pipe = pipe
    if do_tune and param_grid:
        # prefix param names with "model__"
        tuned_params = {f"model__{k}": v for k, v in param_grid.items()}
        search = RandomizedSearchCV(
            pipe, tuned_params, n_iter=tune_iter,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1, cv=tscv, random_state=42
        )
        search.fit(X, y)
        final_pipe = search.best_estimator_
    else:
        final_pipe.fit(X, y)

    return final_pipe, cv_table, mean_rmse, mean_mae, mean_r2

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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import warnings
import time
from datetime import datetime
import hashlib

warnings.filterwarnings('ignore')

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Trade Forecast Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CSS CUSTOM ==================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; }
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; }
    .big-number { font-size: 2rem; font-weight: 700; }
    hr { margin-top: 0.5rem; margin-bottom: 1rem; }

    div[data-testid="stHorizontalRadio"] div[role="radiogroup"] {
        gap: 0px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 4px;
    }
    div[data-testid="stHorizontalRadio"] div[role="radio"] {
        padding: 8px 16px;
        border-radius: 8px;
        background-color: transparent;
        font-size: 14px;
    }
    div[data-testid="stHorizontalRadio"] div[role="radio"][aria-checked="true"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: bold;
    }
    .sidebar-heading {
        font-size: 18px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ================== SESSION STATE ==================
if 'forecast_run' not in st.session_state:
    st.session_state.forecast_run = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'all_results' not in st.session_state:
    st.session_state.all_results = None
if 'all_predictions' not in st.session_state:
    st.session_state.all_predictions = None
if 'series_full' not in st.session_state:
    st.session_state.series_full = None
if 'train_series' not in st.session_state:
    st.session_state.train_series = None
if 'test_series' not in st.session_state:
    st.session_state.test_series = None
if 'display_name' not in st.session_state:
    st.session_state.display_name = ""
if 'forecast_future_data' not in st.session_state:
    st.session_state.forecast_future_data = None
if 'country_forecast_cache' not in st.session_state:
    st.session_state.country_forecast_cache = None
if 'hs6_forecast_cache' not in st.session_state:
    st.session_state.hs6_forecast_cache = None
if 'rca_forecast_cache' not in st.session_state:
    st.session_state.rca_forecast_cache = None
if 'page' not in st.session_state:
    st.session_state.page = "📈 Leaderboard Forecast"
if 'forecast_total' not in st.session_state:
    st.session_state.forecast_total = 0
if 'forecast_cagr' not in st.session_state:
    st.session_state.forecast_cagr = 0.0
if 'forecast_cache' not in st.session_state:
    st.session_state.forecast_cache = {}
if 'total_periode' not in st.session_state:
    st.session_state.total_periode = 0
if 'rata_rata' not in st.session_state:
    st.session_state.rata_rata = 0
if 'cagr_historis' not in st.session_state:
    st.session_state.cagr_historis = 0.0
if 'tahun_terakhir' not in st.session_state:
    st.session_state.tahun_terakhir = 0
if 'direction' not in st.session_state:
    st.session_state.direction = "Impor"
if 'forecast_detail_df' not in st.session_state:
    st.session_state.forecast_detail_df = None

# ================== DATA LOADING ==================
@st.cache_data(ttl=3600, show_spinner="Memuat & Melting dataset...")
def load_and_melt(file_path):
    df = pd.read_csv(file_path)
    year_cols = [col for col in df.columns if "(USD Thousand)" in col]
    id_vars = ['partnerCode', 'partnerLabel', 'productCode', 'productLabel']
    
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )
    df_melted['Year'] = df_melted['Year'].str.extract(r'(\d{4})').astype(int)
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce').fillna(0)
    df_melted['productCode'] = df_melted['productCode'].astype(str)
    df_melted['partnerCode'] = df_melted['partnerCode'].astype(str)
    
    df_melted['partnerLabel'] = df_melted['partnerLabel'].fillna('Unknown').astype(str)
    df_melted['productLabel'] = df_melted['productLabel'].fillna('Unknown Product').astype(str)
    
    return df_melted

@st.cache_data(ttl=3600)
def get_filtered_series(df_melted, country_codes=None, hs6_codes=None):
    filtered = df_melted.copy()
    if country_codes and len(country_codes) > 0:
        filtered = filtered[filtered['partnerCode'].isin(country_codes)]
    if hs6_codes and len(hs6_codes) > 0:
        filtered = filtered[filtered['productCode'].isin(hs6_codes)]
    
    if filtered.empty:
        return pd.DataFrame(columns=['Year', 'Value'])
    
    grouped = filtered.groupby('Year')['Value'].sum().reset_index()
    grouped = grouped.sort_values('Year')
    return grouped

# ================== FORECAST FUNCTIONS ==================
def forecast_holt_winters(series, horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(horizon)
    forecast.index = range(
        series.index.max() + 1,
        series.index.max() + horizon + 1
    )
    return forecast

def forecast_theta(series, horizon):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    model = ThetaModel(
        series,
        deseasonalize=False
    )
    fit = model.fit()
    forecast = fit.forecast(horizon)
    forecast.index = range(
        series.index.max() + 1,
        series.index.max() + horizon + 1
    )
    return forecast

def forecast_auto_arima(series, horizon):
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np
    best_aic = np.inf
    best_model = None
    for p in range(4):
        for d in range(3):
            for q in range(4):
                try:
                    model = ARIMA(
                        series,
                        order=(p, d, q)
                    ).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                except:
                    pass
    if best_model is None:
        raise ValueError("Tidak ditemukan model ARIMA yang valid")
    forecast = best_model.forecast(horizon)
    forecast.index = range(
        series.index.max() + 1,
        series.index.max() + horizon + 1
    )
    return forecast

def naive_forecast(series_df, years):
    if len(series_df) == 0:
        return pd.Series([0] * len(years), index=years)
    last_val = series_df['Value'].iloc[-1]
    return pd.Series([last_val] * len(years), index=years)

def linear_trend_forecast(series_df, steps):
    from sklearn.linear_model import LinearRegression
    X = series_df['Year'].values.reshape(-1, 1)
    y = series_df['Value'].values
    model = LinearRegression().fit(X, y)
    future_years = np.arange(series_df['Year'].max()+1, series_df['Year'].max()+steps+1).reshape(-1, 1)
    preds = model.predict(future_years)
    preds = np.maximum(preds, 0)
    return pd.Series(preds, index=future_years.flatten())

def linear_trend_forecast_fixed(series_df, steps):
    return linear_trend_forecast(series_df, steps)

def forecast_ml_lag(train_data, steps, model_class, model_params):
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        df = train_data[['Year', 'Value']].copy()
        for i in range(1, 4):
            df[f'lag_{i}'] = df['Value'].shift(i)
        df['rolling_mean'] = df['Value'].rolling(window=3).mean()
        df['trend'] = np.arange(1, len(df)+1)
        df = df.dropna()
        
        if len(df) < 5:
            return None
        
        X = df[['lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'trend']]
        y = df['Value']
        
        model = model_class(**model_params)
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipeline.fit(X, y)
        
        last_values = train_data['Value'].values[-3:].tolist()
        predictions = []
        for _ in range(steps):
            l1 = last_values[-1] if len(last_values) >= 1 else 0
            l2 = last_values[-2] if len(last_values) >= 2 else 0
            l3 = last_values[-3] if len(last_values) >= 3 else 0
            rm = np.mean(last_values[-3:]) if len(last_values) >= 3 else np.mean(last_values)
            trend_val = len(train_data) + len(predictions) + 1
            X_pred = pd.DataFrame([[l1, l2, l3, rm, trend_val]], 
                                  columns=['lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'trend'])
            pred_val = pipeline.predict(X_pred)[0]
            predictions.append(max(0, pred_val))
            last_values.append(pred_val)
            if len(last_values) > 5:
                last_values.pop(0)
                
        return pd.DataFrame({
            'Year': range(train_data['Year'].max() + 1, train_data['Year'].max() + steps + 1),
            'Value': predictions
        })
    except:
        return None

def forecast_prophet(series_df, steps):
    if len(series_df) < 3:
        return None
    if series_df['Value'].sum() == 0:
        return None
    if series_df['Value'].std() < 1e-6:
        return None
    zero_ratio = (series_df['Value'] == 0).mean()
    if zero_ratio > 0.8:
        return None
    try:
        from prophet import Prophet
        df_prophet = series_df.rename(columns={'Year': 'ds', 'Value': 'y'})
        m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False,
                    seasonality_mode='additive', changepoint_prior_scale=0.5)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=steps, freq='Y')
        forecast = m.predict(future)
        preds = forecast[['ds', 'yhat']].tail(steps)
        return preds.set_index('ds')['yhat']
    except:
        return None

def forecast_prophet_with_ci(series_df, steps):
    if len(series_df) < 3:
        return None, None, None
    if series_df['Value'].sum() == 0:
        return None, None, None
    if series_df['Value'].std() < 1e-6:
        return None, None, None
    zero_ratio = (series_df['Value'] == 0).mean()
    if zero_ratio > 0.8:
        return None, None, None
    try:
        from prophet import Prophet
        df_prophet = series_df.rename(columns={'Year': 'ds', 'Value': 'y'})
        m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False,
                    seasonality_mode='additive', changepoint_prior_scale=0.5)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=steps, freq='Y')
        forecast = m.predict(future)
        preds = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
        return preds.set_index('ds')['yhat'], preds.set_index('ds')['yhat_lower'], preds.set_index('ds')['yhat_upper']
    except:
        return None, None, None

def forecast_future(model_name, series_df, steps):
    if steps <= 0:
        return None, None, None
    
    std_hist = series_df['Value'].std()
    if std_hist == 0:
        std_hist = 1
    
    series = series_df.set_index('Year')['Value']
    
    try:
        if model_name == "Prophet":
            pred_vals, lower, upper = forecast_prophet_with_ci(series_df, steps)
            if pred_vals is not None:
                return pred_vals, lower, upper
            else:
                pred_vals = linear_trend_forecast(series_df, steps)
                ci_lower = pred_vals - 1.96 * std_hist
                ci_upper = pred_vals + 1.96 * std_hist
                ci_lower = ci_lower.clip(lower=0)
                return pred_vals, ci_lower, ci_upper
            
        elif model_name == "Auto-ARIMA":
            pred_vals = forecast_auto_arima(series, steps)
            try:
                from statsmodels.tsa.arima.model import ARIMA
                best_aic = np.inf
                best_model = None
                for p in range(4):
                    for d in range(3):
                        for q in range(4):
                            try:
                                model = ARIMA(series, order=(p, d, q)).fit()
                                if model.aic < best_aic:
                                    best_aic = model.aic
                                    best_model = model
                            except:
                                pass
                if best_model is not None:
                    resid = best_model.resid
                    std_resid = np.std(resid) if len(resid) > 0 else std_hist
                else:
                    std_resid = std_hist
            except:
                std_resid = std_hist
            ci_lower = pred_vals - 1.96 * std_resid
            ci_upper = pred_vals + 1.96 * std_resid
            ci_lower = ci_lower.clip(lower=0)
            return pred_vals, ci_lower, ci_upper
            
        elif model_name == "Holt-Winters":
            pred_vals = forecast_holt_winters(series, steps)
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                model = ExponentialSmoothing(series, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated")
                fit = model.fit(optimized=True)
                resid = series - fit.fittedvalues
                std_resid = np.std(resid) if len(resid) > 0 else std_hist
            except:
                std_resid = std_hist
            ci_lower = pred_vals - 1.96 * std_resid
            ci_upper = pred_vals + 1.96 * std_resid
            ci_lower = ci_lower.clip(lower=0)
            return pred_vals, ci_lower, ci_upper
            
        elif model_name == "Theta":
            pred_vals = forecast_theta(series, steps)
            try:
                from statsmodels.tsa.forecasting.theta import ThetaModel
                model = ThetaModel(series, deseasonalize=False)
                fit = model.fit()
                resid = series - fit.fittedvalues
                std_resid = np.std(resid) if len(resid) > 0 else std_hist
            except:
                std_resid = std_hist
            ci_lower = pred_vals - 1.96 * std_resid
            ci_upper = pred_vals + 1.96 * std_resid
            ci_lower = ci_lower.clip(lower=0)
            return pred_vals, ci_lower, ci_upper
            
        elif "XGBoost" in model_name:
            from xgboost import XGBRegressor
            result_df = forecast_ml_lag(series_df, steps, XGBRegressor, {'n_estimators': 50, 'random_state': 42})
        elif "LightGBM" in model_name:
            from lightgbm import LGBMRegressor
            result_df = forecast_ml_lag(series_df, steps, LGBMRegressor, {'n_estimators': 50, 'random_state': 42})
        elif "Random Forest" in model_name:
            from sklearn.ensemble import RandomForestRegressor
            result_df = forecast_ml_lag(series_df, steps, RandomForestRegressor, {'n_estimators': 50, 'random_state': 42})
        elif "Linear" in model_name and "Ridge" not in model_name and "ElasticNet" not in model_name:
            pred_vals = linear_trend_forecast(series_df, steps)
            ci_lower = pred_vals - 1.96 * std_hist
            ci_upper = pred_vals + 1.96 * std_hist
            ci_lower = ci_lower.clip(lower=0)
            return pred_vals, ci_lower, ci_upper
        elif "Ridge" in model_name:
            from sklearn.linear_model import Ridge
            result_df = forecast_ml_lag(series_df, steps, Ridge, {'alpha': 1.0})
        elif "ElasticNet" in model_name:
            from sklearn.linear_model import ElasticNet
            result_df = forecast_ml_lag(series_df, steps, ElasticNet, {'alpha': 0.1, 'l1_ratio': 0.5})
        else:
            result_df = None
        
        if result_df is not None and len(result_df) == steps:
            pred_vals = result_df.set_index('Year')['Value']
            ci_lower = pred_vals - 1.96 * std_hist
            ci_upper = pred_vals + 1.96 * std_hist
            ci_lower = ci_lower.clip(lower=0)
            return pred_vals, ci_lower, ci_upper
        else:
            pred_vals = linear_trend_forecast(series_df, steps)
            ci_lower = pred_vals - 1.96 * std_hist
            ci_upper = pred_vals + 1.96 * std_hist
            ci_lower = ci_lower.clip(lower=0)
            return pred_vals, ci_lower, ci_upper
                
    except:
        pred_vals = linear_trend_forecast(series_df, steps)
        ci_lower = pred_vals - 1.96 * std_hist
        ci_upper = pred_vals + 1.96 * std_hist
        ci_lower = ci_lower.clip(lower=0)
        return pred_vals, ci_lower, ci_upper

def evaluate_model(train_series_df, test_series_df, model_name, steps):
    actuals = test_series_df.set_index('Year')['Value']
    test_years = test_series_df['Year'].values
    
    if len(train_series_df) < 2 or len(test_series_df) < 1:
        return None, None, None, None, "Data tidak cukup"
    
    train_series = train_series_df.set_index('Year')['Value']
    
    if np.std(train_series) < 1e-9 or len(train_series) < 3:
        pred_vals = linear_trend_forecast_fixed(train_series_df, steps)
        if len(pred_vals) < len(test_years):
            last_val = pred_vals.iloc[-1] if len(pred_vals) > 0 else 0
            extra = pd.Series([last_val] * (len(test_years) - len(pred_vals)), index=test_years[len(pred_vals):])
            pred_vals = pd.concat([pred_vals, extra])
        elif len(pred_vals) > len(test_years):
            pred_vals = pred_vals.iloc[:len(test_years)]
        pred_vals.index = test_years
        status = "Linear Trend (data pendek)"
        rmse = np.sqrt(np.mean((actuals - pred_vals)**2))
        mae = np.mean(np.abs(actuals - pred_vals))
        actuals_nonzero = actuals.copy()
        actuals_nonzero[actuals_nonzero == 0] = 1
        mape = np.mean(np.abs((actuals - pred_vals) / actuals_nonzero)) * 100
        return pred_vals, rmse, mae, mape, status
    
    if model_name == "Prophet":
        preds = forecast_prophet(train_series_df, steps)
        if preds is not None:
            pred_vals = preds
            if len(pred_vals) < len(test_years):
                last_val = pred_vals.iloc[-1] if len(pred_vals) > 0 else 0
                extra = pd.Series([last_val] * (len(test_years) - len(pred_vals)), index=test_years[len(pred_vals):])
                pred_vals = pd.concat([pred_vals, extra])
            elif len(pred_vals) > len(test_years):
                pred_vals = pred_vals.iloc[:len(test_years)]
            pred_vals.index = test_years
            status = "Selesai"
        else:
            pred_vals = linear_trend_forecast_fixed(train_series_df, steps)
            if len(pred_vals) < len(test_years):
                last_val = pred_vals.iloc[-1] if len(pred_vals) > 0 else 0
                extra = pd.Series([last_val] * (len(test_years) - len(pred_vals)), index=test_years[len(pred_vals):])
                pred_vals = pd.concat([pred_vals, extra])
            elif len(pred_vals) > len(test_years):
                pred_vals = pred_vals.iloc[:len(test_years)]
            pred_vals.index = test_years
            status = "Linear Trend (Prophet skip)"
        rmse = np.sqrt(np.mean((actuals - pred_vals)**2))
        mae = np.mean(np.abs(actuals - pred_vals))
        actuals_nonzero = actuals.copy()
        actuals_nonzero[actuals_nonzero == 0] = 1
        mape = np.mean(np.abs((actuals - pred_vals) / actuals_nonzero)) * 100
        return pred_vals, rmse, mae, mape, status
    
    if model_name in ["Linear Regression", "Ridge Regression", "ElasticNet"]:
        if len(train_series_df) < 5:
            pred_vals = linear_trend_forecast_fixed(train_series_df, steps)
            if len(pred_vals) < len(test_years):
                last_val = pred_vals.iloc[-1] if len(pred_vals) > 0 else 0
                extra = pd.Series([last_val] * (len(test_years) - len(pred_vals)), index=test_years[len(pred_vals):])
                pred_vals = pd.concat([pred_vals, extra])
            elif len(pred_vals) > len(test_years):
                pred_vals = pred_vals.iloc[:len(test_years)]
            pred_vals.index = test_years
            status = "Linear Trend (data sedikit)"
        else:
            from sklearn.linear_model import LinearRegression
            X_train = train_series_df['Year'].values.reshape(-1, 1)
            y_train = train_series_df['Value'].values
            model = LinearRegression().fit(X_train, y_train)
            X_test = test_years.reshape(-1, 1)
            pred_vals = model.predict(X_test)
            pred_vals = np.maximum(pred_vals, 0)
            pred_vals = pd.Series(pred_vals, index=test_years)
            status = "Selesai"
        rmse = np.sqrt(np.mean((actuals - pred_vals)**2))
        mae = np.mean(np.abs(actuals - pred_vals))
        actuals_nonzero = actuals.copy()
        actuals_nonzero[actuals_nonzero == 0] = 1
        mape = np.mean(np.abs((actuals - pred_vals) / actuals_nonzero)) * 100
        return pred_vals, rmse, mae, mape, status
    
    pred_vals = None
    status = "Selesai"
    try:
        if model_name == "Auto-ARIMA":
            pred_vals = forecast_auto_arima(train_series, steps)
            if len(pred_vals) < len(test_years):
                last_val = pred_vals.iloc[-1]
                extra = pd.Series([last_val] * (len(test_years) - len(pred_vals)), index=test_years[len(pred_vals):])
                pred_vals = pd.concat([pred_vals, extra])
            elif len(pred_vals) > len(test_years):
                pred_vals = pred_vals.iloc[:len(test_years)]
            pred_vals.index = test_years
            
        elif model_name == "Holt-Winters":
            pred_vals = forecast_holt_winters(train_series, steps)
            if len(pred_vals) < len(test_years):
                last_val = pred_vals.iloc[-1]
                extra = pd.Series([last_val] * (len(test_years) - len(pred_vals)), index=test_years[len(pred_vals):])
                pred_vals = pd.concat([pred_vals, extra])
            elif len(pred_vals) > len(test_years):
                pred_vals = pred_vals.iloc[:len(test_years)]
            pred_vals.index = test_years
            
        elif model_name == "Theta":
            pred_vals = forecast_theta(train_series, steps)
            if len(pred_vals) < len(test_years):
                last_val = pred_vals.iloc[-1]
                extra = pd.Series([last_val] * (len(test_years) - len(pred_vals)), index=test_years[len(pred_vals):])
                pred_vals = pd.concat([pred_vals, extra])
            elif len(pred_vals) > len(test_years):
                pred_vals = pred_vals.iloc[:len(test_years)]
            pred_vals.index = test_years
            
        elif "XGBoost" in model_name:
            from xgboost import XGBRegressor
            result_df = forecast_ml_lag(train_series_df, steps, XGBRegressor, {'n_estimators': 50, 'random_state': 42})
            if result_df is not None:
                pred_vals = result_df.set_index('Year')['Value']
        elif "LightGBM" in model_name:
            from lightgbm import LGBMRegressor
            result_df = forecast_ml_lag(train_series_df, steps, LGBMRegressor, {'n_estimators': 50, 'random_state': 42})
            if result_df is not None:
                pred_vals = result_df.set_index('Year')['Value']
        elif "Random Forest" in model_name:
            from sklearn.ensemble import RandomForestRegressor
            result_df = forecast_ml_lag(train_series_df, steps, RandomForestRegressor, {'n_estimators': 50, 'random_state': 42})
            if result_df is not None:
                pred_vals = result_df.set_index('Year')['Value']
        else:
            pass
    except:
        pred_vals = None
    
    if pred_vals is None:
        pred_vals = linear_trend_forecast_fixed(train_series_df, steps)
        if len(pred_vals) < len(test_years):
            last_val = pred_vals.iloc[-1] if len(pred_vals) > 0 else 0
            extra = pd.Series([last_val] * (len(test_years) - len(pred_vals)), index=test_years[len(pred_vals):])
            pred_vals = pd.concat([pred_vals, extra])
        elif len(pred_vals) > len(test_years):
            pred_vals = pred_vals.iloc[:len(test_years)]
        pred_vals.index = test_years
        status = "Linear Trend (fallback)"
    
    rmse = np.sqrt(np.mean((actuals - pred_vals)**2))
    mae = np.mean(np.abs(actuals - pred_vals))
    actuals_nonzero = actuals.copy()
    actuals_nonzero[actuals_nonzero == 0] = 1
    mape = np.mean(np.abs((actuals - pred_vals) / actuals_nonzero)) * 100
    
    return pred_vals, rmse, mae, mape, status

def chow_breakpoint_test(series_df):
    from statsmodels.regression.linear_model import OLS
    import statsmodels.api as sm
    if len(series_df) < 10:
        return None, None
    y = series_df['Value'].values
    x = np.arange(len(y))
    mid = len(y) // 2
    X_full = sm.add_constant(x)
    model_full = OLS(y, X_full).fit()
    rss_full = model_full.ssr
    X1 = sm.add_constant(x[:mid])
    X2 = sm.add_constant(x[mid:])
    model1 = OLS(y[:mid], X1).fit()
    model2 = OLS(y[mid:], X2).fit()
    rss_split = model1.ssr + model2.ssr
    k = 2
    n = len(y)
    f_stat = ((rss_full - rss_split) / k) / (rss_split / (n - 2*k))
    from scipy.stats import f
    p_value = 1 - f.cdf(f_stat, k, n - 2*k)
    return f_stat, p_value

# ================== RCA & EPD ==================
def calculate_rca_for_year(df_melted, year):
    df_year = df_melted[df_melted['Year'] == year].copy()
    if df_year.empty:
        return df_year
    
    totals_country = df_year.groupby('partnerCode')['Value'].sum().rename('total_country')
    totals_world = df_year['Value'].sum()
    totals_product = df_year.groupby('productCode')['Value'].sum().rename('total_product')
    
    df_year = df_year.merge(totals_country, on='partnerCode')
    df_year['total_world'] = totals_world
    df_year = df_year.merge(totals_product, on='productCode')
    
    df_year['RCA'] = (df_year['Value'] / df_year['total_country']) / (df_year['total_product'] / df_year['total_world'])
    df_year['RCA'] = df_year['RCA'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_year

def calculate_rca_forecast_all(df_melted, country_codes, hs6_codes, model_name, horizon):
    if country_codes and len(country_codes) > 0:
        top_countries = country_codes
    else:
        totals = df_melted.groupby('partnerCode')['Value'].sum().reset_index()
        totals = totals.sort_values('Value', ascending=False).head(10)
        top_countries = totals['partnerCode'].tolist()
    
    if hs6_codes and len(hs6_codes) > 0:
        top_hs6 = hs6_codes
    else:
        all_hs = df_melted['productCode'].unique().tolist()
        if len(all_hs) > 200:
            totals_hs = df_melted.groupby('productCode')['Value'].sum().reset_index()
            totals_hs = totals_hs.sort_values('Value', ascending=False).head(200)
            top_hs6 = totals_hs['productCode'].tolist()
        else:
            top_hs6 = all_hs
    
    forecast_rows = []
    total_combos = len(top_countries) * len(top_hs6)
    if total_combos == 0:
        return pd.DataFrame()
    
    progress_bar = st.progress(0, text="Menghitung forecast RCA...")
    processed = 0
    for c_code in top_countries:
        for h_code in top_hs6:
            processed += 1
            progress_bar.progress(processed / total_combos, text=f"Memproses {c_code} - {h_code}")
            temp_df = df_melted[(df_melted['partnerCode'] == c_code) & (df_melted['productCode'] == h_code)]
            if temp_df.empty:
                continue
            series_df = temp_df.groupby('Year')['Value'].sum().reset_index().sort_values('Year')
            if len(series_df) < 3:
                continue
            pred_vals, _, _ = forecast_future(model_name, series_df, horizon)
            if pred_vals is not None and len(pred_vals) > 0:
                total_forecast = pred_vals.sum()
                country_label = df_melted[df_melted['partnerCode'] == c_code]['partnerLabel'].iloc[0]
                product_label = df_melted[df_melted['productCode'] == h_code]['productLabel'].iloc[0]
                forecast_rows.append({
                    'partnerCode': c_code,
                    'partnerLabel': country_label,
                    'productCode': h_code,
                    'productLabel': product_label,
                    'Value': total_forecast
                })
    
    progress_bar.empty()
    
    if not forecast_rows:
        return pd.DataFrame()
    
    df_forecast = pd.DataFrame(forecast_rows)
    
    totals_country_f = df_forecast.groupby('partnerCode')['Value'].sum().rename('total_country')
    totals_world_f = df_forecast['Value'].sum()
    totals_product_f = df_forecast.groupby('productCode')['Value'].sum().rename('total_product')
    
    df_forecast = df_forecast.merge(totals_country_f, on='partnerCode')
    df_forecast['total_world'] = totals_world_f
    df_forecast = df_forecast.merge(totals_product_f, on='productCode')
    
    df_forecast['RCA'] = (df_forecast['Value'] / df_forecast['total_country']) / (df_forecast['total_product'] / df_forecast['total_world'])
    df_forecast['RCA'] = df_forecast['RCA'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_forecast

def get_epd_matrix(df_rca, df_growth):
    df_epd = df_rca.merge(df_growth, on='productCode', how='left')
    df_epd['Growth'] = df_epd['Growth'].fillna(0)
    df_epd['Quadrant'] = df_epd.apply(
        lambda row: 'Rising Stars' if (row['RCA'] > 1 and row['Growth'] > 0) else
                    ('Falling Stars' if (row['RCA'] > 1 and row['Growth'] <= 0) else
                     ('Opportunities' if (row['RCA'] <= 1 and row['Growth'] > 0) else 'Retreat')),
        axis=1
    )
    return df_epd

def compute_forecast_for_entity(df_melted, code, model_name, horizon, level='country'):
    if level == 'country':
        id_col = 'partnerCode'
        label_col = 'partnerLabel'
    else:
        id_col = 'productCode'
        label_col = 'productLabel'
    
    entity_df = df_melted[df_melted[id_col] == code]
    if entity_df.empty:
        return None
    
    label = entity_df[label_col].iloc[0]
    series_df = entity_df.groupby('Year')['Value'].sum().reset_index().sort_values('Year')
    
    if len(series_df) < 3:
        return None
    
    pred_vals, _, _ = forecast_future(model_name, series_df, horizon)
    if pred_vals is not None and len(pred_vals) > 0:
        total_forecast = pred_vals.sum()
        return {
            'Code': code,
            'Label': label,
            'Total_Forecast': total_forecast
        }
    return None

# ================== GENERATE FORECAST DETAIL ==================
def generate_forecast_detail(df_melted, country_codes, hs6_codes, model_name, horizon):
    """Menghasilkan data detail forecast per negara, per HS6, per tahun."""
    if country_codes:
        countries = country_codes
    else:
        top_countries = df_melted.groupby('partnerCode')['Value'].sum().sort_values(ascending=False).head(10).index.tolist()
        countries = top_countries
    
    if hs6_codes:
        hs6_list = hs6_codes
    else:
        top_hs6 = df_melted.groupby('productCode')['Value'].sum().sort_values(ascending=False).head(50).index.tolist()
        hs6_list = top_hs6
    
    rows = []
    total_combos = len(countries) * len(hs6_list)
    if total_combos == 0:
        return pd.DataFrame()
    
    progress_bar = st.progress(0, text="Membangun data forecast detail...")
    processed = 0
    for c_code in countries:
        for h_code in hs6_list:
            processed += 1
            progress_bar.progress(processed / total_combos, text=f"Memproses {c_code} - {h_code}")
            temp_df = df_melted[(df_melted['partnerCode'] == c_code) & (df_melted['productCode'] == h_code)]
            if temp_df.empty:
                continue
            series_df = temp_df.groupby('Year')['Value'].sum().reset_index().sort_values('Year')
            if len(series_df) < 3:
                continue
            pred_vals, _, _ = forecast_future(model_name, series_df, horizon)
            if pred_vals is not None and len(pred_vals) > 0:
                for yr, val in pred_vals.items():
                    rows.append({
                        'tahun': int(yr),
                        'partnerCode': c_code,
                        'partnerLabel': temp_df['partnerLabel'].iloc[0],
                        'productCode': h_code,
                        'productLabel': temp_df['productLabel'].iloc[0],
                        'nilai': val
                    })
    progress_bar.empty()
    return pd.DataFrame(rows)

# ================== SIDEBAR ==================
with st.sidebar:
    direction = st.radio(
        "Arah Perdagangan",
        ["Impor", "Ekspor"],
        horizontal=True,
        key="direction_radio",
        label_visibility="collapsed"
    )
    st.session_state.direction = direction
    
    st.markdown(f"<p class='sidebar-heading'>⚡ Konfigurasi Data {direction} Indonesia</p>", unsafe_allow_html=True)
    
    file_map = {"Impor": "imporIndonesia.csv", "Ekspor": "eksporIndonesia.csv"}
    file_path = file_map[direction]
    df_raw = load_and_melt(file_path)
    
    partner_unique = df_raw[['partnerCode', 'partnerLabel']].drop_duplicates()
    partner_unique = partner_unique[partner_unique['partnerLabel'] != 'Unknown']
    partner_unique = partner_unique.sort_values('partnerLabel')
    country_options = [f"{row['partnerCode']} - {row['partnerLabel']}" for _, row in partner_unique.iterrows()]
    country_map = {opt: opt.split(' - ')[0] for opt in country_options}
    
    hs6_unique = df_raw[['productCode', 'productLabel']].drop_duplicates()
    hs6_unique = hs6_unique[hs6_unique['productLabel'] != 'Unknown Product']
    hs6_unique = hs6_unique.sort_values('productCode')
    hs6_options = [f"{row['productCode']} - {row['productLabel']}" for _, row in hs6_unique.iterrows()]
    hs6_map = {opt: opt.split(' - ')[0] for opt in hs6_options}
    
    selected_country_labels = st.multiselect(
        "🌍 Pilih Negara (kosongkan untuk Semua)",
        country_options,
        default=[],
        placeholder="Cari negara..."
    )
    selected_hs6_labels = st.multiselect(
        "📦 Pilih HS6 (kosongkan untuk Semua)",
        hs6_options,
        default=[],
        placeholder="Cari HS6..."
    )
    
    country_codes = [country_map[lab] for lab in selected_country_labels]
    hs6_codes = [hs6_map[lab] for lab in selected_hs6_labels]
    
    all_models = [
        "Prophet", "Auto-ARIMA", "Holt-Winters", "Theta",
        "XGBoost", "LightGBM", "Random Forest", 
        "Linear Regression", "Ridge Regression", "ElasticNet"
    ]
    selected_models = st.multiselect(
        "🤖 Pilih Model ML (kosongkan untuk Semua)",
        all_models,
        default=[],
        placeholder="Cari model..."
    )
    if not selected_models:
        selected_models = all_models
    
    st.markdown("---")
    st.markdown(f"<p class='sidebar-heading'>Split Data training dan testing</p>", unsafe_allow_html=True)
    train_end = st.slider(
        "Tahun Akhir Training",
        2010, 2022, 2020,
        key="train_end",
        label_visibility="collapsed"
    )
    
    st.markdown(f"<p class='sidebar-heading'>🔮 Tahun Ramalan</p>", unsafe_allow_html=True)
    forecast_horizon = st.slider(
        "Tahun Ramalan",
        1, 25, 5,
        key="horizon",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Preview Data (mentah)")
    if country_codes:
        df_preview = df_raw[df_raw['partnerCode'].isin(country_codes)]
    else:
        df_preview = df_raw
    st.dataframe(df_preview.head(10), use_container_width=True)
    
    run_btn = st.button("🚀 Jalankan Full Analysis", type="primary", use_container_width=True)

# ================== MAIN LOGIC ==================
if not country_codes and not hs6_codes:
    display_name = "Total (Semua)"
elif country_codes and not hs6_codes:
    labels = [df_raw[df_raw['partnerCode'] == c]['partnerLabel'].iloc[0] for c in country_codes]
    display_name = f"{len(country_codes)} Negara: {', '.join(labels[:2])}" + ("..." if len(labels) > 2 else "")
elif not country_codes and hs6_codes:
    labels = [df_raw[df_raw['productCode'] == c]['productLabel'].iloc[0][:30] for c in hs6_codes]
    display_name = f"{len(hs6_codes)} HS6: {', '.join(labels[:2])}" + ("..." if len(labels) > 2 else "")
else:
    c_labels = [df_raw[df_raw['partnerCode'] == c]['partnerLabel'].iloc[0] for c in country_codes]
    h_labels = [df_raw[df_raw['productCode'] == c]['productLabel'].iloc[0][:20] for c in hs6_codes]
    display_name = f"{len(country_codes)} Negara, {len(hs6_codes)} HS6"

st.markdown(f"# 📊 Dashboard Forecast {direction} Indonesia")
st.markdown(f"**Level Analisis:** {display_name}")

if run_btn:
    st.session_state.forecast_run = False
    st.session_state.all_results = None
    st.session_state.best_model = None
    st.session_state.forecast_future_data = None
    st.session_state.country_forecast_cache = None
    st.session_state.hs6_forecast_cache = None
    st.session_state.rca_forecast_cache = None
    st.session_state.forecast_detail_df = None
    
    series_full = get_filtered_series(df_raw, country_codes, hs6_codes)
    
    if series_full.empty:
        st.error("Tidak ada data untuk kombinasi pilihan ini.")
        st.stop()
    
    zero_count = (series_full['Value'] == 0).sum()
    if zero_count > 10:
        st.warning(f"⚠️ Data memiliki {zero_count} tahun bernilai 0 dari {len(series_full)} tahun. Ramalan mungkin tidak akurat.")
    
    # Historical metrics
    total_val = series_full['Value'].sum()
    avg_val = series_full['Value'].mean()
    start_val = series_full[series_full['Year'] == 2001]['Value'].values[0] if 2001 in series_full['Year'].values else series_full['Value'].iloc[0]
    end_val = series_full[series_full['Year'] == 2025]['Value'].values[0] if 2025 in series_full['Year'].values else series_full['Value'].iloc[-1]
    cagr = (end_val / start_val) ** (1 / (len(series_full)-1)) - 1 if start_val > 0 else 0
    
    st.session_state.total_periode = total_val
    st.session_state.rata_rata = avg_val
    st.session_state.cagr_historis = cagr
    st.session_state.tahun_terakhir = end_val
    
    train_series = series_full[series_full['Year'] <= train_end]
    test_series = series_full[series_full['Year'] >= train_end+1]
    
    if len(test_series) < 1:
        st.warning("Tidak ada data testing. Gunakan seluruh data untuk training.")
        test_series = series_full.tail(2)
        train_series = series_full.head(len(series_full)-2)
    
    st.markdown("---")
    st.subheader(f"⚡ Evaluasi & Perbandingan Model (Train: 2001-{train_end}, Test: {train_end+1}-2025)")
    
    progress_bar = st.progress(0, text="Inisialisasi...")
    status_text = st.empty()
    
    model_list = selected_models
    results = []
    all_predictions = {}
    steps = len(test_series)
    
    for i, model_name in enumerate(model_list):
        status_text.text(f"⏳ Menjalankan {model_name}... ({i+1}/{len(model_list)})")
        progress_bar.progress((i) / len(model_list), text=f"Training {model_name}")
        
        preds, rmse, mae, mape, status = evaluate_model(train_series, test_series, model_name, steps)
        
        if preds is not None:
            results.append({
                "Model": model_name,
                "RMSE (USD Ribu)": rmse,
                "MAE (USD Ribu)": mae,
                "MAPE (%)": mape,
                "Status": status
            })
            all_predictions[model_name] = preds
        else:
            preds = naive_forecast(train_series, test_series['Year'].values)
            all_predictions[model_name] = preds
            actuals = test_series.set_index('Year')['Value']
            rmse = np.sqrt(np.mean((actuals - preds)**2))
            mae = np.mean(np.abs(actuals - preds))
            actuals_nonzero = actuals.copy()
            actuals_nonzero[actuals_nonzero == 0] = 1
            mape = np.mean(np.abs((actuals - preds) / actuals_nonzero)) * 100
            results.append({
                "Model": model_name,
                "RMSE (USD Ribu)": rmse,
                "MAE (USD Ribu)": mae,
                "MAPE (%)": mape,
                "Status": "Naive (fallback)"
            })
    
    progress_bar.progress(1.0, text="✅ Evaluasi Selesai!")
    status_text.text("✅ Semua model telah dievaluasi.")
    time.sleep(0.5)
    progress_bar.empty()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("MAPE (%)", na_position='last')
    best_model_name = df_results.iloc[0]['Model'] if not df_results.empty and not pd.isna(df_results.iloc[0]['MAPE (%)']) else None
    
    if best_model_name:
        steps_future = forecast_horizon
        if steps_future > 0:
            pred_vals, _, _ = forecast_future(best_model_name, series_full, steps_future)
            if pred_vals is not None:
                forecast_df = pd.DataFrame({
                    'Year': pred_vals.index,
                    'Value': pred_vals.values
                })
                st.session_state.forecast_future_data = forecast_df
                st.session_state.forecast_total = pred_vals.sum()
                st.session_state.forecast_cagr = (pred_vals.iloc[-1] / series_full['Value'].iloc[-1]) ** (1 / steps_future) - 1 if steps_future > 0 and series_full['Value'].iloc[-1] > 0 else 0
                
                # Kontributor forecast (top 20)
                with st.spinner("Menghitung kontributor forecast (top 20)..."):
                    if not country_codes or len(country_codes) > 1:
                        if country_codes:
                            codes_to_process = country_codes[:20]
                        else:
                            totals = df_raw.groupby('partnerCode')['Value'].sum().reset_index()
                            totals = totals.sort_values('Value', ascending=False).head(20)
                            codes_to_process = totals['partnerCode'].tolist()
                        
                        country_results = []
                        for code in codes_to_process:
                            result = compute_forecast_for_entity(df_raw, code, best_model_name, steps_future, level='country')
                            if result:
                                country_results.append(result)
                        if country_results:
                            st.session_state.country_forecast_cache = pd.DataFrame(country_results).sort_values('Total_Forecast', ascending=False)
                    
                    if not hs6_codes or len(hs6_codes) > 1:
                        if hs6_codes:
                            codes_to_process = hs6_codes[:20]
                        else:
                            totals = df_raw.groupby('productCode')['Value'].sum().reset_index()
                            totals = totals.sort_values('Value', ascending=False).head(20)
                            codes_to_process = totals['productCode'].tolist()
                        
                        hs6_results = []
                        for code in codes_to_process:
                            result = compute_forecast_for_entity(df_raw, code, best_model_name, steps_future, level='hs6')
                            if result:
                                hs6_results.append(result)
                        if hs6_results:
                            st.session_state.hs6_forecast_cache = pd.DataFrame(hs6_results).sort_values('Total_Forecast', ascending=False)
                    
                    st.session_state._rca_params = {
                        'model': best_model_name,
                        'horizon': steps_future,
                        'country_codes': country_codes,
                        'hs6_codes': hs6_codes
                    }
                
                # Generate forecast detail untuk export (per tahun per HS6)
                st.session_state.forecast_detail_df = generate_forecast_detail(df_raw, country_codes, hs6_codes, best_model_name, steps_future)
                
                st.session_state.forecast_run = True
                st.session_state.best_model = best_model_name
                st.session_state.all_results = df_results
                st.session_state.all_predictions = all_predictions
                st.session_state.series_full = series_full
                st.session_state.train_series = train_series
                st.session_state.test_series = test_series
                st.session_state.display_name = display_name
                st.rerun()

# ================== METRIK ==================
if st.session_state.forecast_run and st.session_state.series_full is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Periode", f"USD {st.session_state.total_periode:,.0f} Ribu")
    col2.metric("📈 Rata-rata / Tahun", f"USD {st.session_state.rata_rata:,.0f} Ribu")
    col3.metric("📊 CAGR (25 Tahun)", f"{st.session_state.cagr_historis:.2%}")
    col4.metric("📆 Tahun Terakhir (2025)", f"USD {st.session_state.tahun_terakhir:,.0f} Ribu")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("🔮 Total Forecast", f"USD {st.session_state.forecast_total:,.0f} Ribu")
    col6.metric("📊 CAGR Forecast", f"{st.session_state.forecast_cagr:.2%}")
    col7.empty()
    col8.empty()

# ================== NAVIGASI ==================
pages = ["📊 Data Preview", "📈 Leaderboard Forecast", "📊 Contributors", "📤 Export", "🔍 Structural Break", "📖 Panduan Istilah", "📊 RCA & EPD"]
page = st.radio(
    "Navigasi",
    pages,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio"
)
st.session_state.page = page

# ================== KONTEN ==================
if st.session_state.page == "📊 Data Preview":
    st.markdown("## 📊 Preview Data Historis (Agregat per Tahun)")
    st.markdown(f"**Filter:** {display_name}")
    if st.session_state.forecast_run and st.session_state.series_full is not None:
        preview_df = st.session_state.series_full.copy()
        st.dataframe(
            preview_df.style.format({'Value': '{:,.0f}'}),
            use_container_width=True
        )
        st.caption(f"Total {len(preview_df)} tahun data historis (2001-2025)")
        fig_preview = px.line(preview_df, x='Year', y='Value', title="Data Historis Agregat")
        fig_preview.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig_preview, use_container_width=True)
    else:
        st.info("Belum ada data. Jalankan analisis terlebih dahulu.")

elif st.session_state.page == "📈 Leaderboard Forecast":
    if st.session_state.forecast_run and st.session_state.all_results is not None:
        df_results = st.session_state.all_results
        all_predictions = st.session_state.all_predictions
        series_full = st.session_state.series_full
        test_series = st.session_state.test_series
        best_model_name = st.session_state.best_model
        
        col_left, col_right = st.columns([1.5, 1])
        with col_left:
            st.markdown("### 🏆 Leaderboard Akurasi (MAPE)")
            display_df = df_results.drop(columns=['Status'], errors='ignore')
            st.dataframe(
                display_df.style.background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r'),
                use_container_width=True,
                hide_index=True
            )
        
        with col_right:
            st.markdown("### 📋 Metrik Tambahan")
            if best_model_name:
                best_row = df_results[df_results['Model'] == best_model_name].iloc[0]
                st.metric("🏅 Model Terbaik", best_model_name)
                st.metric("📉 MAPE Terendah", f"{best_row['MAPE (%)']:.2f}%" if not pd.isna(best_row['MAPE (%)']) else "N/A")
                st.metric("📉 RMSE Terendah", f"USD {best_row['RMSE (USD Ribu)']:,.0f}" if not pd.isna(best_row['RMSE (USD Ribu)']) else "N/A")
        
        st.markdown("---")
        st.markdown("### 🔮 Perbandingan Forecast Semua Model")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=series_full['Year'], y=series_full['Value'],
                                          mode='lines+markers', name='Historis',
                                          line=dict(color='blue', width=3)))
        colors = px.colors.qualitative.Plotly
        for idx, (model, preds) in enumerate(all_predictions.items()):
            color = colors[idx % len(colors)]
            fig_forecast.add_trace(go.Scatter(
                x=test_series['Year'], y=preds,
                mode='lines+markers', name=model,
                line=dict(color=color, width=1.5, dash='dot'),
                opacity=0.4
            ))
        if best_model_name and best_model_name in all_predictions:
            fig_forecast.add_trace(go.Scatter(
                x=test_series['Year'], y=all_predictions[best_model_name],
                mode='lines+markers', name=f"⭐ {best_model_name} (Terbaik)",
                line=dict(color='red', width=4),
                marker=dict(size=12)
            ))
        fig_forecast.update_layout(height=500, template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.markdown("### 📅 Ramalan Masa Depan (Extended)")
        if best_model_name:
            steps_future = forecast_horizon
            if steps_future > 0:
                full_series = series_full
                with st.spinner(f"Menghitung ramalan {steps_future} tahun ke depan dengan {best_model_name}..."):
                    pred_vals, lower, upper = forecast_future(best_model_name, full_series, steps_future)
                    if pred_vals is not None:
                        fig_future = go.Figure()
                        fig_future.add_trace(go.Scatter(x=full_series['Year'], y=full_series['Value'],
                                                        mode='lines+markers', name='Historis',
                                                        line=dict(color='blue', width=3)))
                        fig_future.add_trace(go.Scatter(x=pred_vals.index, y=pred_vals,
                                                        mode='lines+markers', name=f'Forecast ({best_model_name})',
                                                        line=dict(color='red', width=3, dash='dash')))
                        if lower is not None and upper is not None:
                            x_vals = pd.Series(pred_vals.index)
                            y_upper = upper
                            y_lower = lower
                            fig_future.add_trace(go.Scatter(
                                x=pd.concat([x_vals, x_vals[::-1]]),
                                y=pd.concat([y_upper, y_lower[::-1]]),
                                fill='toself', fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,0,0,0)'),
                                name='Confidence Interval (±1.96σ)'
                            ))
                        fig_future.update_layout(height=400, template='plotly_white')
                        st.plotly_chart(fig_future, use_container_width=True)
                        
                        st.markdown("#### Tabel Forecast")
                        df_future_vertical = pd.DataFrame({
                            'Tahun': pred_vals.index,
                            'Prediksi (USD Ribu)': pred_vals.values
                        })
                        if lower is not None and upper is not None:
                            df_future_vertical['Lower Bound'] = lower.values
                            df_future_vertical['Upper Bound'] = upper.values
                        st.dataframe(df_future_vertical, use_container_width=True)
                    else:
                        st.warning("Gagal menghasilkan ramalan masa depan.")
            else:
                st.info("Pilih tahun ramalan di sidebar.")
        else:
            st.info("Tidak ada model terbaik untuk melakukan ramalan.")
    else:
        st.info("Belum ada hasil forecasting. Jalankan analisis terlebih dahulu.")

elif st.session_state.page == "📊 Contributors":
    st.markdown("## 🧩 Analisis Kontributor")
    if not st.session_state.forecast_run:
        st.info("Jalankan analisis terlebih dahulu untuk melihat kontribusi.")
    else:
        sub_tab1, sub_tab2 = st.tabs(["📊 Kontribusi Historis (2001-2025)", "🔮 Kontribusi Forecast (Masa Depan)"])
        with sub_tab1:
            st.markdown("### Kontribusi Berdasarkan Data Historis")
            filtered_data = df_raw.copy()
            if country_codes and len(country_codes) > 0:
                filtered_data = filtered_data[filtered_data['partnerCode'].isin(country_codes)]
            if hs6_codes and len(hs6_codes) > 0:
                filtered_data = filtered_data[filtered_data['productCode'].isin(hs6_codes)]
            
            if filtered_data.empty:
                st.info("Tidak ada data untuk filter yang dipilih.")
            else:
                st.markdown("#### 🌍 Top 10 Negara (Total 2001-2025)")
                top_countries = filtered_data.groupby('partnerLabel')['Value'].sum().sort_values(ascending=True).tail(10).reset_index()
                fig_c = px.bar(top_countries, x='Value', y='partnerLabel', orientation='h', title="Top 10 Mitra Dagang (Historis)")
                fig_c.update_layout(height=400, font=dict(size=14), title_font=dict(size=16), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_c, use_container_width=True)
                
                st.markdown("#### 📦 Top 10 Komoditas (HS6) - Total 2001-2025")
                top_hs6 = filtered_data.groupby(['productCode', 'productLabel'])['Value'].sum().sort_values(ascending=False).head(10).reset_index()
                top_hs6_display = top_hs6[['productCode', 'productLabel', 'Value']].rename(
                    columns={'productCode': 'Kode HS6', 'productLabel': 'Deskripsi', 'Value': 'Total (USD Ribu)'}
                )
                top_hs6_display['Total (USD Ribu)'] = top_hs6_display['Total (USD Ribu)'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(top_hs6_display, use_container_width=True)
                
                st.markdown("#### 📈 CAGR per Negara (Top 10)")
                cagr_data = []
                for country in filtered_data['partnerLabel'].unique():
                    temp = filtered_data[filtered_data['partnerLabel'] == country].groupby('Year')['Value'].sum()
                    if len(temp) > 1:
                        start = temp.iloc[0]
                        end = temp.iloc[-1]
                        if start > 0:
                            cagr = (end / start) ** (1 / (len(temp)-1)) - 1
                            cagr_data.append({'Negara': country, 'CAGR': cagr})
                if cagr_data:
                    df_cagr = pd.DataFrame(cagr_data).sort_values('CAGR', ascending=True).tail(10).reset_index(drop=True)
                    fig_cagr = px.bar(df_cagr, x='CAGR', y='Negara', orientation='h', title="Top 10 Negara dengan Pertumbuhan Tertinggi (Historis)")
                    fig_cagr.update_layout(height=400, font=dict(size=14), title_font=dict(size=16), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_cagr, use_container_width=True)
        
        with sub_tab2:
            forecast_start = st.session_state.forecast_future_data['Year'].min() if st.session_state.forecast_future_data is not None else '???'
            forecast_end = st.session_state.forecast_future_data['Year'].max() if st.session_state.forecast_future_data is not None else '???'
            st.markdown(f"### 🔮 Kontribusi Forecast ({forecast_start} - {forecast_end})")
            if not st.session_state.best_model:
                st.info("Tidak ada model terbaik untuk melakukan forecast kontributor.")
            else:
                steps_future = forecast_horizon
                if steps_future <= 0:
                    st.info("Silakan pilih tahun ramalan di sidebar.")
                else:
                    df_country = st.session_state.country_forecast_cache
                    if df_country is not None and not df_country.empty:
                        st.markdown("#### 🌍 Top 10 Negara (Total Forecast)")
                        top_countries_f = df_country.head(10).sort_values('Total_Forecast', ascending=True)
                        fig_cf = px.bar(top_countries_f, x='Total_Forecast', y='Label', 
                                       orientation='h', title=f"Top 10 Negara Berdasarkan Forecast (Model: {st.session_state.best_model})")
                        fig_cf.update_layout(height=400, font=dict(size=14), title_font=dict(size=16), 
                                             yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_cf, use_container_width=True)
                        
                        st.dataframe(
                            df_country.head(10).style.format({'Total_Forecast': '{:,.0f}'}),
                            use_container_width=True
                        )
                        
                        st.markdown("#### 📈 Kontribusi Top Negara (Forecast)")
                        df_cagr_f = df_country.head(10).sort_values('Total_Forecast', ascending=True)
                        fig_cagr_f = px.bar(df_cagr_f, x='Total_Forecast', y='Label', 
                                            orientation='h', title="Top 10 Kontribusi Forecast")
                        fig_cagr_f.update_layout(height=400, font=dict(size=14), title_font=dict(size=16), 
                                                 yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_cagr_f, use_container_width=True)
                    else:
                        st.info("Tidak ada data forecast untuk negara (mungkin filter kosong atau data tidak cukup).")
                    
                    df_hs6 = st.session_state.hs6_forecast_cache
                    if df_hs6 is not None and not df_hs6.empty:
                        st.markdown("#### 📦 Top 10 Komoditas (HS6) - Total Forecast")
                        top_hs6_f = df_hs6.head(10)
                        st.dataframe(
                            top_hs6_f.rename(columns={'Code': 'Kode HS6', 'Label': 'Deskripsi', 'Total_Forecast': 'Total Forecast (USD Ribu)'})
                            .style.format({'Total Forecast (USD Ribu)': '{:,.0f}'}),
                            use_container_width=True
                        )
                    else:
                        st.info("Tidak ada data forecast untuk HS6.")

elif st.session_state.page == "📤 Export":
    st.markdown("## 📤 Export Hasil Forecast")
    if not st.session_state.forecast_run:
        st.info("Jalankan analisis terlebih dahulu untuk mengekspor data.")
    else:
        direction = st.session_state.direction
        
        # 1. Unduh Forecast Agregat per Tahun
        if st.session_state.forecast_future_data is not None:
            df_agg_year = st.session_state.forecast_future_data.copy()
            sql_lines = []
            for _, row in df_agg_year.iterrows():
                sql_lines.append(f"INSERT INTO forecast_{direction.lower()}_aggregate (tahun, nilai) VALUES ({int(row['Year'])}, {row['Value']:.0f});")
            st.download_button(
                label="📥 Unduh Forecast Agregat per Tahun (SQL)",
                data="\n".join(sql_lines),
                file_name=f"forecast_aggregate_{direction}_{datetime.now().strftime('%Y%m%d')}.sql",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Tidak ada data forecast agregat.")
        
        # 2. Unduh Forecast Agregat per HS6
        if st.session_state.forecast_detail_df is not None and not st.session_state.forecast_detail_df.empty:
            df_detail = st.session_state.forecast_detail_df
            df_hs6_agg = df_detail.groupby(['productCode', 'productLabel'])['nilai'].sum().reset_index()
            df_hs6_agg = df_hs6_agg.sort_values('nilai', ascending=False)
            sql_lines = []
            for _, row in df_hs6_agg.iterrows():
                sql_lines.append(f"INSERT INTO forecast_{direction.lower()}_hs6_aggregate (productCode, productLabel, total_nilai) VALUES ('{row['productCode']}', '{row['productLabel']}', {row['nilai']:.0f});")
            st.download_button(
                label="📥 Unduh Forecast Agregat per HS6 (SQL)",
                data="\n".join(sql_lines),
                file_name=f"forecast_hs6_aggregate_{direction}_{datetime.now().strftime('%Y%m%d')}.sql",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Tidak ada data forecast per HS6.")
        
        # 3. Unduh Forecast Agregat per Negara
        if st.session_state.forecast_detail_df is not None and not st.session_state.forecast_detail_df.empty:
            df_detail = st.session_state.forecast_detail_df
            df_country_agg = df_detail.groupby(['partnerCode', 'partnerLabel'])['nilai'].sum().reset_index()
            df_country_agg = df_country_agg.sort_values('nilai', ascending=False)
            sql_lines = []
            for _, row in df_country_agg.iterrows():
                sql_lines.append(f"INSERT INTO forecast_{direction.lower()}_country_aggregate (partnerCode, partnerLabel, total_nilai) VALUES ('{row['partnerCode']}', '{row['partnerLabel']}', {row['nilai']:.0f});")
            st.download_button(
                label="📥 Unduh Forecast Agregat per Negara (SQL)",
                data="\n".join(sql_lines),
                file_name=f"forecast_country_aggregate_{direction}_{datetime.now().strftime('%Y%m%d')}.sql",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Tidak ada data forecast per negara.")
        
        # 4. Unduh Forecast Detail per Tahun per HS6
        if st.session_state.forecast_detail_df is not None and not st.session_state.forecast_detail_df.empty:
            df_detail = st.session_state.forecast_detail_df
            sql_lines = []
            for _, row in df_detail.iterrows():
                sql_lines.append(
                    f"INSERT INTO forecast_{direction.lower()}_detail (tahun, partnerCode, partnerLabel, productCode, productLabel, nilai) "
                    f"VALUES ({row['tahun']}, '{row['partnerCode']}', '{row['partnerLabel']}', '{row['productCode']}', '{row['productLabel']}', {row['nilai']:.0f});"
                )
            st.download_button(
                label="📥 Unduh Forecast Detail per Tahun per HS6 (SQL)",
                data="\n".join(sql_lines),
                file_name=f"forecast_detail_{direction}_{datetime.now().strftime('%Y%m%d')}.sql",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Tidak ada data forecast detail.")
        
        # 5. Unduh RCA Forecast SQL
        st.markdown("---")
        if st.session_state.rca_forecast_cache is None and st.session_state.best_model:
            with st.spinner("Menghitung RCA Forecast..."):
                rca_forecast = calculate_rca_forecast_all(
                    df_raw, 
                    country_codes, 
                    hs6_codes, 
                    st.session_state.best_model, 
                    forecast_horizon
                )
                st.session_state.rca_forecast_cache = rca_forecast
        rca_forecast = st.session_state.rca_forecast_cache
        if rca_forecast is not None and not rca_forecast.empty:
            sql_lines_rca = []
            for _, row in rca_forecast.iterrows():
                sql_lines_rca.append(f"INSERT INTO rca_forecast_{direction.lower()}_table (partnerCode, partnerLabel, productCode, productLabel, RCA, Value) VALUES ('{row['partnerCode']}', '{row['partnerLabel']}', '{row['productCode']}', '{row['productLabel']}', {row['RCA']:.4f}, {row['Value']:.0f});")
            st.download_button(
                label="📥 Unduh RCA Forecast SQL",
                data="\n".join(sql_lines_rca),
                file_name=f"rca_forecast_{direction}_{datetime.now().strftime('%Y%m%d')}.sql",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Tidak ada data RCA forecast.")
        
        # 6. Unduh RCA Dataset (Historis 2025)
        st.markdown("---")
        rca_hist = calculate_rca_for_year(df_raw, 2025)
        if not rca_hist.empty:
            sql_lines_rca_hist = []
            for _, row in rca_hist.iterrows():
                sql_lines_rca_hist.append(
                    f"INSERT INTO rca_historis_{direction.lower()}_2025 (partnerCode, partnerLabel, productCode, productLabel, RCA, Value) "
                    f"VALUES ('{row['partnerCode']}', '{row['partnerLabel']}', '{row['productCode']}', '{row['productLabel']}', {row['RCA']:.4f}, {row['Value']:.0f});"
                )
            st.download_button(
                label="📥 Unduh RCA Dataset (2025) SQL",
                data="\n".join(sql_lines_rca_hist),
                file_name=f"rca_dataset_{direction}_2025_{datetime.now().strftime('%Y%m%d')}.sql",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Data RCA historis 2025 tidak tersedia.")

elif st.session_state.page == "🔍 Structural Break":
    st.markdown("## 🔍 Structural Break Detection (Data Historis)")
    if not st.session_state.forecast_run or st.session_state.series_full is None:
        st.info("Jalankan analisis terlebih dahulu.")
    else:
        series_full = st.session_state.series_full
        f_stat, p_val = chow_breakpoint_test(series_full)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 F-Statistic (Chow Test)", f"{f_stat:.3f}" if f_stat else "N/A")
        with col2:
            if p_val is not None:
                is_break = p_val < 0.05
                st.metric("🎯 P-Value", f"{p_val:.4f}")
                if is_break:
                    st.error("🚨 **Structural Break Terdeteksi!**")
                else:
                    st.success("✅ Tidak terdeteksi structural break signifikan.")
            else:
                st.info("Data terlalu pendek (< 10 tahun) untuk melakukan Chow Test.")
        
        st.markdown("### 📈 Visualisasi Stabilitas Tren (Data Historis)")
        series_full['Rolling_Mean_3'] = series_full['Value'].rolling(window=3).mean()
        series_full['Rolling_Std_3'] = series_full['Value'].rolling(window=3).std()
        
        fig_break = go.Figure()
        fig_break.add_trace(go.Scatter(x=series_full['Year'], y=series_full['Value'],
                                       mode='lines+markers', name='Data Historis',
                                       line=dict(color='blue', width=2)))
        fig_break.add_trace(go.Scatter(x=series_full['Year'], y=series_full['Rolling_Mean_3'],
                                       mode='lines', name='Moving Average (3 tahun)',
                                       line=dict(color='orange', width=3, dash='dash')))
        fig_break.add_trace(go.Scatter(
            x=pd.concat([series_full['Year'], series_full['Year'][::-1]]),
            y=pd.concat([series_full['Rolling_Mean_3'] + series_full['Rolling_Std_3'],
                         (series_full['Rolling_Mean_3'] - series_full['Rolling_Std_3'])[::-1]]),
            fill='toself', fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,165,0,0)'), name='±1 Std Dev'
        ))
        if p_val is not None and p_val < 0.05:
            mid_year = series_full['Year'].iloc[len(series_full)//2]
            fig_break.add_vline(x=mid_year, line_dash="dash", line_color="red", 
                               annotation_text=f"Break Point ({mid_year})")
        fig_break.update_layout(height=450, template='plotly_white')
        st.plotly_chart(fig_break, use_container_width=True)
        st.caption("""
        **Interpretasi:**  
        - Jika Moving Average (garis oranye) menyimpang jauh dari data historis, itu menandakan perubahan tren.  
        - Jika P-Value < 0.05, ada bukti statistik bahwa perubahan tersebut signifikan.
        """)

elif st.session_state.page == "📖 Panduan Istilah":
    st.markdown("## 📖 Panduan Istilah untuk Pemula")
    col1, col2, col3 = st.columns(3)
    
    terms = [
        ("📉 RMSE", """
        - Ukuran rata-rata kesalahan prediksi dalam satuan **USD Ribu**.
        - Semakin kecil angkanya, semakin akurat modelnya.
        - Contoh: RMSE = 500 berarti rata-rata prediksi meleset sekitar USD 500.000.
        """),
        ("📊 MAE", """
        - Rata-rata selisih absolut antara prediksi dan nilai aktual.
        - Lebih mudah dipahami karena tidak memperbesar error besar seperti RMSE.
        - Semakin kecil, semakin baik.
        """),
        ("🎯 MAPE", """
        - Rata-rata kesalahan prediksi dalam bentuk **persentase**.
        - Contoh: MAPE = 10% berarti rata-rata prediksi Anda meleset 10%.
        - **Metrik utama** penentu model terbaik di Leaderboard.
        """),
        ("📈 CAGR", """
        - Rata-rata pertumbuhan per tahun selama periode tertentu.
        - Contoh: CAGR 5% berarti nilai perdagangan tumbuh rata-rata 5% per tahun.
        - Membantu melihat tren naik/turun/stagnan.
        """),
        ("🔬 Structural Break", """
        - Uji statistik untuk mendeteksi perubahan drastis pada tren data di **masa lalu**.
        - **F-Statistic**: Nilai besar menandakan adanya perubahan.
        - **P-Value**: Jika < 0.05, perubahan signifikan secara statistik.
        - Hanya untuk data historis, bukan forecast masa depan.
        - Break point yang terdeteksi biasanya disebabkan oleh faktor eksternal seperti krisis ekonomi, pandemi, atau perubahan kebijakan perdagangan.
        """),
        ("🧩 Confidence Interval", """
        - Rentang nilai di mana prediksi sebenarnya diperkirakan berada (biasanya 95%).
        - Semakin lebar, semakin tinggi ketidakpastian.
        - Menggunakan **±1.96 × standar deviasi** dari residual atau data historis.
        """),
        ("⚠️ Sparsity & Skewness", """
        - **Sparsity**: Persentase sel data bernilai 0. Jika >70%, agregasi ke HS2/negara lebih baik.
        - **Skewness**: Angka > 5 menandakan ada nilai ekstrem (misal migas). Disarankan transformasi logaritma.
        """)
    ]
    
    for i, (title, content) in enumerate(terms):
        if i % 3 == 0:
            with col1:
                with st.expander(f"**{title}**", expanded=False):
                    st.markdown(content)
        elif i % 3 == 1:
            with col2:
                with st.expander(f"**{title}**", expanded=False):
                    st.markdown(content)
        else:
            with col3:
                with st.expander(f"**{title}**", expanded=False):
                    st.markdown(content)

elif st.session_state.page == "📊 RCA & EPD":
    st.markdown("## 📊 Revealed Comparative Advantage (RCA) & Export Product Dynamics (EPD)")
    if not st.session_state.forecast_run:
        st.info("Jalankan analisis terlebih dahulu untuk menampilkan RCA/EPD.")
    else:
        mode_rca = st.radio(
            "Pilih periode analisis",
            ["Historis (2025)", "Forecast (2026-2030)"],
            horizontal=True,
            key="rca_mode",
            label_visibility="collapsed"
        )
        
        if country_codes and len(country_codes) > 0:
            default_country = country_codes[0]
        else:
            top_country = df_raw.groupby('partnerCode')['Value'].sum().sort_values(ascending=False).index[0]
            default_country = top_country
        
        selected_rca_country_code = default_country
        country_label = df_raw[df_raw['partnerCode'] == selected_rca_country_code]['partnerLabel'].iloc[0]
        st.markdown(f"**Analisis untuk negara:** {selected_rca_country_code} - {country_label}")
        
        if mode_rca == "Historis (2025)":
            df_rca_2025 = calculate_rca_for_year(df_raw, 2025)
            if df_rca_2025.empty:
                st.warning("Data tahun 2025 tidak ditemukan.")
            else:
                df_rca_country = df_rca_2025[df_rca_2025['partnerCode'] == selected_rca_country_code]
                df_rca_country = df_rca_country.sort_values('RCA', ascending=False)
                
                st.markdown(f"#### 📋 Top HS6 dengan RCA Tertinggi (Historis)")
                if not df_rca_country.empty:
                    top_rca = df_rca_country[['productCode', 'productLabel', 'RCA', 'Value']].copy()
                    top_rca['RCA'] = top_rca['RCA'].round(2)
                    top_rca['Value (USD Ribu)'] = top_rca['Value'].apply(lambda x: f"{x:,.0f}")
                    st.dataframe(top_rca.rename(columns={'productCode': 'Kode HS6', 'productLabel': 'Deskripsi'}), use_container_width=True)
                    
                    df_2023 = df_raw[(df_raw['Year'] == 2023) & (df_raw['partnerCode'] == selected_rca_country_code)]
                    df_2025 = df_raw[(df_raw['Year'] == 2025) & (df_raw['partnerCode'] == selected_rca_country_code)]
                    if not df_2023.empty and not df_2025.empty:
                        growth_products = df_2025.set_index('productCode')['Value'] / df_2023.set_index('productCode')['Value'] - 1
                        growth_products = growth_products.replace([np.inf, -np.inf], 0).fillna(0).reset_index()
                        growth_products.columns = ['productCode', 'Growth']
                        df_epd = get_epd_matrix(df_rca_country, growth_products)
                        if not df_epd.empty:
                            st.markdown(f"#### 📈 EPD Matrix: RCA vs Pertumbuhan 2023-2025")
                            fig_epd = px.scatter(
                                df_epd, x='Growth', y='RCA', color='Quadrant',
                                hover_data={'productCode': True, 'productLabel': True},
                                title="EPD Matrix (RCA vs Growth)",
                                labels={'Growth': 'Pertumbuhan 2023-2025', 'RCA': 'RCA 2025'}
                            )
                            fig_epd.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
                            fig_epd.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                            fig_epd.update_layout(height=500, template='plotly_white')
                            st.plotly_chart(fig_epd, use_container_width=True)
                            
                            st.markdown("#### 📋 Klasifikasi EPD")
                            st.dataframe(
                                df_epd[['productCode', 'productLabel', 'RCA', 'Growth', 'Quadrant']]
                                .rename(columns={'productCode': 'Kode HS6', 'productLabel': 'Deskripsi', 'Growth': 'Growth (%)'})
                                .style.format({'Growth (%)': '{:.2%}', 'RCA': '{:.2f}'}),
                                use_container_width=True
                            )
                            
                            st.markdown("#### ⚡ Perubahan Paling Drastis")
                            top_rca_hs = df_epd.loc[df_epd['RCA'].idxmax()]
                            top_growth_hs = df_epd.loc[df_epd['Growth'].idxmax()]
                            st.write(f"- **RCA Tertinggi:** {top_rca_hs['productCode']} - {top_rca_hs['productLabel']} (RCA: {top_rca_hs['RCA']:.2f})")
                            st.write(f"- **Pertumbuhan Tertinggi:** {top_growth_hs['productCode']} - {top_growth_hs['productLabel']} (Growth: {top_growth_hs['Growth']:.2%})")
                        else:
                            st.info("Tidak ada data untuk EPD.")
                else:
                    st.info("Tidak ada data RCA untuk negara ini.")
        else:  # Forecast
            st.markdown(f"#### 🔮 Analisis RCA Forecast (2026-2030) dengan Model: {st.session_state.best_model if st.session_state.best_model else '-'}")
            if not st.session_state.best_model:
                st.warning("Tidak ada model terbaik. Jalankan analisis terlebih dahulu.")
            else:
                if st.session_state.rca_forecast_cache is None:
                    with st.spinner("Menghitung RCA Forecast (semua HS untuk negara terpilih)..."):
                        rca_forecast = calculate_rca_forecast_all(
                            df_raw, 
                            [selected_rca_country_code], 
                            hs6_codes, 
                            st.session_state.best_model, 
                            forecast_horizon
                        )
                        st.session_state.rca_forecast_cache = rca_forecast
                else:
                    rca_forecast = st.session_state.rca_forecast_cache
                
                if rca_forecast is None or rca_forecast.empty:
                    st.info("Belum ada data forecast RCA. Silakan hitung terlebih dahulu dengan tombol di bawah.")
                    if st.button("Hitung RCA Forecast Sekarang"):
                        with st.spinner("Menghitung RCA Forecast..."):
                            rca_forecast = calculate_rca_forecast_all(
                                df_raw, 
                                [selected_rca_country_code],
                                hs6_codes, 
                                st.session_state.best_model, 
                                forecast_horizon
                            )
                            st.session_state.rca_forecast_cache = rca_forecast
                            st.rerun()
                else:
                    df_rca_country = rca_forecast[rca_forecast['partnerCode'] == selected_rca_country_code]
                    df_rca_country = df_rca_country.sort_values('RCA', ascending=False)
                    
                    if not df_rca_country.empty:
                        st.markdown(f"#### 📋 Top HS6 dengan RCA Tertinggi (Forecast)")
                        top_rca = df_rca_country[['productCode', 'productLabel', 'RCA', 'Value']].copy()
                        top_rca['RCA'] = top_rca['RCA'].round(2)
                        top_rca['Value (USD Ribu)'] = top_rca['Value'].apply(lambda x: f"{x:,.0f}")
                        st.dataframe(top_rca.rename(columns={'productCode': 'Kode HS6', 'productLabel': 'Deskripsi'}), use_container_width=True)
                        
                        df_2025_actual = df_raw[(df_raw['Year'] == 2025) & (df_raw['partnerCode'] == selected_rca_country_code)]
                        if not df_2025_actual.empty and not df_rca_country.empty:
                            actual_2025 = df_2025_actual.set_index('productCode')['Value']
                            forecast_vals = df_rca_country.set_index('productCode')['Value']
                            growth_series = (forecast_vals / actual_2025) - 1
                            growth_series = growth_series.replace([np.inf, -np.inf], 0).fillna(0).reset_index()
                            growth_series.columns = ['productCode', 'Growth']
                            df_epd_forecast = get_epd_matrix(df_rca_country, growth_series)
                            if not df_epd_forecast.empty:
                                st.markdown(f"#### 📈 EPD Matrix Forecast: RCA vs Pertumbuhan Forecast")
                                fig_epd_f = px.scatter(
                                    df_epd_forecast, x='Growth', y='RCA', color='Quadrant',
                                    hover_data={'productCode': True, 'productLabel': True},
                                    title="EPD Matrix Forecast (RCA vs Growth Forecast)",
                                    labels={'Growth': 'Pertumbuhan Forecast vs 2025', 'RCA': 'RCA Forecast'}
                                )
                                fig_epd_f.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
                                fig_epd_f.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                                fig_epd_f.update_layout(height=500, template='plotly_white')
                                st.plotly_chart(fig_epd_f, use_container_width=True)
                                
                                st.markdown("#### 📋 Klasifikasi EPD Forecast")
                                st.dataframe(
                                    df_epd_forecast[['productCode', 'productLabel', 'RCA', 'Growth', 'Quadrant']]
                                    .rename(columns={'productCode': 'Kode HS6', 'productLabel': 'Deskripsi', 'Growth': 'Growth (%)'})
                                    .style.format({'Growth (%)': '{:.2%}', 'RCA': '{:.2f}'}),
                                    use_container_width=True
                                )
                                
                                st.markdown("#### ⚡ Perubahan Paling Drastis (Forecast)")
                                top_rca_hs = df_epd_forecast.loc[df_epd_forecast['RCA'].idxmax()]
                                top_growth_hs = df_epd_forecast.loc[df_epd_forecast['Growth'].idxmax()]
                                st.write(f"- **RCA Tertinggi:** {top_rca_hs['productCode']} - {top_rca_hs['productLabel']} (RCA: {top_rca_hs['RCA']:.2f})")
                                st.write(f"- **Pertumbuhan Tertinggi:** {top_growth_hs['productCode']} - {top_growth_hs['productLabel']} (Growth: {top_growth_hs['Growth']:.2%})")
                            else:
                                st.info("Tidak cukup data untuk EPD forecast.")
                    else:
                        st.info(f"Tidak ada data forecast RCA untuk negara ini. Mungkin negara ini tidak memiliki data forecast yang cukup. Coba pilih negara lain.")

# ================== FOOTER ==================
st.markdown("---")
st.caption(f"Python 3.12 | {len(df_raw):,} baris data")
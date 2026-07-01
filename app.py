import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import warnings
import time
from datetime import datetime

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
</style>
""", unsafe_allow_html=True)

# ================== SESSION STATE INIT ==================
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

# ================== DATA LOADING (CACHED) ==================
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

# ================== OPTIMIZED FORECAST CONTRIBUTORS ==================
def compute_forecast_for_entity(df_melted, code, model_name, horizon, level='country'):
    """
    Fungsi helper untuk menghitung forecast satu entitas (negara atau HS6).
    Dibatasi hanya 50 entitas teratas untuk menghindari lambat.
    """
    if level == 'country':
        id_col = 'partnerCode'
        label_col = 'partnerLabel'
    else:
        id_col = 'productCode'
        label_col = 'productLabel'
    
    # Ambil data entity
    entity_df = df_melted[df_melted[id_col] == code]
    if entity_df.empty:
        return None
    
    label = entity_df[label_col].iloc[0]
    series_df = entity_df.groupby('Year')['Value'].sum().reset_index().sort_values('Year')
    
    if len(series_df) < 3:
        return None
    
    # Forecast
    pred_vals, _, _ = forecast_future(model_name, series_df, horizon)
    if pred_vals is not None and len(pred_vals) > 0:
        total_forecast = pred_vals.sum()
        return {
            'Code': code,
            'Label': label,
            'Total_Forecast': total_forecast
        }
    return None

@st.cache_data(ttl=3600)
def get_forecast_contributors_optimized(df_melted, model_name, horizon, level='country', filter_codes=None, limit=50):
    """
    Menghitung kontributor forecast hanya untuk top `limit` entitas berdasarkan total historis.
    """
    if level == 'country':
        id_col = 'partnerCode'
        label_col = 'partnerLabel'
    else:
        id_col = 'productCode'
        label_col = 'productLabel'
    
    # Jika ada filter, gunakan filter, tapi batasi jumlahnya
    if filter_codes and len(filter_codes) > 0:
        codes_to_process = filter_codes[:limit]
    else:
        # Hitung total historis per entitas, ambil top `limit`
        totals = df_melted.groupby([id_col, label_col])['Value'].sum().reset_index()
        totals = totals.sort_values('Value', ascending=False).head(limit)
        codes_to_process = totals[id_col].tolist()
    
    results = []
    # Gunakan progress bar sederhana (tidak bisa di-cache, jadi kita jalankan di luar cache)
    # Kita kembalikan list of dict, nanti di main kita loop dengan progress
    # Karena fungsi ini di-cache, progress tidak bisa ditampilkan. Jadi kita panggil dari main dengan loop.
    # Fungsi ini hanya menyiapkan data yang dibutuhkan.
    # Kita akan pindahkan logika loop ke main.
    return codes_to_process

# ================== SIDEBAR ==================
with st.sidebar:
    direction = st.radio(
        "📂 Arah Perdagangan",
        ["Impor", "Ekspor"],
        horizontal=True,
        key="direction"
    )
    st.markdown(f"## ⚙️ Konfigurasi Data {direction} Indonesia")
    
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
    st.markdown("### 🗓️ Split Data")
    train_end = st.slider(
        "Tahun Akhir Training (dan Awal Testing)",
        2010, 2022, 2020,
        key="train_end"
    )
    st.caption("**Rekomendasi:** 2020")
    test_start = train_end + 1
    if test_start > 2025:
        test_start = 2025
    st.caption(f"Training: 2001 - {train_end} | Testing: {test_start} - 2025")
    
    st.markdown("---")
    forecast_horizon = st.slider("🔮 Tahun Ramalan", 1, 25, 5, key="horizon")
    
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
    st.session_state.forecast_run = True
    st.session_state.all_results = None
    st.session_state.best_model = None
    st.session_state.forecast_future_data = None
    st.session_state.country_forecast_cache = None
    st.session_state.hs6_forecast_cache = None
    
    series_full = get_filtered_series(df_raw, country_codes, hs6_codes)
    
    if series_full.empty:
        st.error("Tidak ada data untuk kombinasi pilihan ini.")
        st.stop()
    
    zero_count = (series_full['Value'] == 0).sum()
    if zero_count > 10:
        st.warning(f"⚠️ Data memiliki {zero_count} tahun bernilai 0 dari {len(series_full)} tahun. Ramalan mungkin tidak akurat.")
    
    col1, col2, col3, col4 = st.columns(4)
    total_val = series_full['Value'].sum()
    avg_val = series_full['Value'].mean()
    start_val = series_full[series_full['Year'] == 2001]['Value'].values[0] if 2001 in series_full['Year'].values else series_full['Value'].iloc[0]
    end_val = series_full[series_full['Year'] == 2025]['Value'].values[0] if 2025 in series_full['Year'].values else series_full['Value'].iloc[-1]
    cagr = (end_val / start_val) ** (1 / (len(series_full)-1)) - 1 if start_val > 0 else 0
    
    col1.metric("💰 Total Periode", f"USD {total_val:,.0f} Ribu")
    col2.metric("📈 Rata-rata / Tahun", f"USD {avg_val:,.0f} Ribu")
    col3.metric("📊 CAGR (25 Tahun)", f"{cagr:.2%}")
    col4.metric("📆 Tahun Terakhir (2025)", f"USD {end_val:,.0f} Ribu")
    
    train_series = series_full[series_full['Year'] <= train_end]
    test_series = series_full[series_full['Year'] >= test_start]
    
    if len(test_series) < 1:
        st.warning("Tidak ada data testing. Menggunakan data terakhir untuk validasi.")
        test_series = series_full.tail(2)
        train_series = series_full.head(len(series_full)-2)
        test_start = test_series['Year'].min()
    
    st.markdown("---")
    st.subheader(f"⚡ Evaluasi & Perbandingan Model (Train: 2001-{train_end}, Test: {test_start}-2025)")
    
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
    
    # Simpan forecast future untuk structural break
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
                
                # ======== OPTIMIZED FORECAST CONTRIBUTORS ========
                with st.spinner("Menghitung kontributor forecast (maksimal 50 entitas)..."):

                    # --- COUNTRY CONTRIBUTORS ---
                    if not country_codes or len(country_codes) > 1:
                        # Ambil daftar kode negara (filter atau top 50 berdasarkan historis)
                        if country_codes:
                            codes_to_process = country_codes[:50]
                        else:
                            # Hitung top 50 negara berdasarkan total historis
                            totals = df_raw.groupby('partnerCode')['Value'].sum().reset_index()
                            totals = totals.sort_values('Value', ascending=False).head(50)
                            codes_to_process = totals['partnerCode'].tolist()
                        
                        country_results = []
                        country_progress = st.progress(0, text="Memproses negara...")
                        for idx, code in enumerate(codes_to_process):
                            country_progress.progress((idx+1)/len(codes_to_process), text=f"Memproses negara {idx+1}/{len(codes_to_process)}")
                            result = compute_forecast_for_entity(df_raw, code, best_model_name, steps_future, level='country')
                            if result:
                                country_results.append(result)
                        country_progress.empty()
                        
                        if country_results:
                            df_country = pd.DataFrame(country_results).sort_values('Total_Forecast', ascending=False)
                            st.session_state.country_forecast_cache = df_country
                        else:
                            st.session_state.country_forecast_cache = pd.DataFrame()
                    
                    # --- HS6 CONTRIBUTORS ---
                    if not hs6_codes or len(hs6_codes) > 1:
                        if hs6_codes:
                            codes_to_process = hs6_codes[:50]
                        else:
                            totals = df_raw.groupby('productCode')['Value'].sum().reset_index()
                            totals = totals.sort_values('Value', ascending=False).head(50)
                            codes_to_process = totals['productCode'].tolist()
                        
                        hs6_results = []
                        hs6_progress = st.progress(0, text="Memproses HS6...")
                        for idx, code in enumerate(codes_to_process):
                            hs6_progress.progress((idx+1)/len(codes_to_process), text=f"Memproses HS6 {idx+1}/{len(codes_to_process)}")
                            result = compute_forecast_for_entity(df_raw, code, best_model_name, steps_future, level='hs6')
                            if result:
                                hs6_results.append(result)
                        hs6_progress.empty()
                        
                        if hs6_results:
                            df_hs6 = pd.DataFrame(hs6_results).sort_values('Total_Forecast', ascending=False)
                            st.session_state.hs6_forecast_cache = df_hs6
                        else:
                            st.session_state.hs6_forecast_cache = pd.DataFrame()
    
    st.session_state.all_results = df_results
    st.session_state.best_model = best_model_name
    st.session_state.all_predictions = all_predictions
    st.session_state.series_full = series_full
    st.session_state.train_series = train_series
    st.session_state.test_series = test_series
    st.session_state.display_name = display_name

# ================== TABS ==================
if st.session_state.forecast_run and st.session_state.all_results is not None:
    df_results = st.session_state.all_results
    all_predictions = st.session_state.all_predictions
    series_full = st.session_state.series_full
    test_series = st.session_state.test_series
    display_name = st.session_state.display_name
    best_model_name = st.session_state.best_model
    
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Preview",
        "📈 Leaderboard & Forecast",
        "📊 Contributors",
        "📤 Export Data",
        "🔍 Structural Break",
        "📖 Panduan Istilah"
    ])
    
    with tab0:
        st.markdown("## 📊 Preview Data Historis (Agregat per Tahun)")
        st.markdown(f"**Filter:** {display_name}")
        preview_df = series_full.copy()
        if not preview_df.empty:
            n_rows = st.selectbox("Jumlah baris yang ditampilkan", [10, 25, 50, 100], index=0)
            st.dataframe(
                preview_df.head(n_rows).style.format({'Value': '{:,.0f}'}),
                use_container_width=True
            )
            st.caption(f"Total {len(preview_df)} tahun data historis (2001-2025)")
            fig_preview = px.line(preview_df, x='Year', y='Value', title="Data Historis Agregat")
            fig_preview.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig_preview, use_container_width=True)
        else:
            st.info("Tidak ada data untuk filter yang dipilih.")
    
    with tab1:
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
                        
                        st.markdown("#### Tabel Vertikal (tahun di baris)")
                        df_future_vertical = pd.DataFrame({
                            'Tahun': pred_vals.index,
                            'Prediksi (USD Ribu)': pred_vals.values
                        })
                        if lower is not None and upper is not None:
                            df_future_vertical['Lower Bound'] = lower.values
                            df_future_vertical['Upper Bound'] = upper.values
                        st.dataframe(df_future_vertical, use_container_width=True)
                        
                        st.markdown("#### Tabel Horizontal (tahun di kolom)")
                        years = pred_vals.index.tolist()
                        df_wide = pd.DataFrame({'Metric': ['Prediksi', 'Lower Bound', 'Upper Bound']})
                        for y in years:
                            row_pred = pred_vals[y] if y in pred_vals.index else None
                            row_lower = lower[y] if lower is not None and y in lower.index else None
                            row_upper = upper[y] if upper is not None and y in upper.index else None
                            df_wide[str(y)] = [row_pred, row_lower, row_upper]
                        st.dataframe(df_wide, use_container_width=True)
                    else:
                        st.warning("Gagal menghasilkan ramalan masa depan.")
            else:
                st.info("Pilih tahun ramalan di sidebar.")
        else:
            st.info("Tidak ada model terbaik untuk melakukan ramalan.")
    
    with tab2:
        st.markdown("## 🧩 Analisis Kontributor")
        
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
            st.markdown("### Kontribusi Berdasarkan Forecast (Model Terbaik)")
            if not best_model_name:
                st.info("Tidak ada model terbaik untuk melakukan forecast kontributor.")
            else:
                steps_future = forecast_horizon
                if steps_future <= 0:
                    st.info("Silakan pilih tahun ramalan di sidebar.")
                else:
                    # Country
                    df_country = st.session_state.country_forecast_cache
                    if df_country is not None and not df_country.empty:
                        st.markdown(f"#### 🌍 Top 10 Negara (Total Forecast {forecast_horizon} Tahun Ke Depan)")
                        top_countries_f = df_country.head(10).sort_values('Total_Forecast', ascending=True)
                        fig_cf = px.bar(top_countries_f, x='Total_Forecast', y='Label', 
                                       orientation='h', title=f"Top 10 Negara Berdasarkan Forecast (Model: {best_model_name})")
                        fig_cf.update_layout(height=400, font=dict(size=14), title_font=dict(size=16), 
                                             yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_cf, use_container_width=True)
                        
                        st.dataframe(
                            df_country.head(10).style.format({'Total_Forecast': '{:,.0f}'}),
                            use_container_width=True
                        )
                    else:
                        st.info("Tidak ada data forecast untuk negara (mungkin data kosong atau model gagal).")
                    
                    # HS6
                    df_hs6 = st.session_state.hs6_forecast_cache
                    if df_hs6 is not None and not df_hs6.empty:
                        st.markdown(f"#### 📦 Top 10 Komoditas (HS6) - Total Forecast {forecast_horizon} Tahun Ke Depan")
                        top_hs6_f = df_hs6.head(10)
                        st.dataframe(
                            top_hs6_f.rename(columns={'Code': 'Kode HS6', 'Label': 'Deskripsi', 'Total_Forecast': 'Total Forecast (USD Ribu)'})
                            .style.format({'Total Forecast (USD Ribu)': '{:,.0f}'}),
                            use_container_width=True
                        )
                    else:
                        st.info("Tidak ada data forecast untuk HS6.")
    
    with tab3:
        st.markdown("## 📤 Export Hasil")
        export_long = pd.DataFrame({'Tahun': test_series['Year'].values})
        for model, preds in all_predictions.items():
            export_long[model] = preds.values
        export_long['Aktual'] = test_series['Value'].values
        
        csv_data = export_long.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download CSV (tahun di baris)", data=csv_data, file_name=f"forecast_{direction}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
        
        try:
            import openpyxl
            engine = 'openpyxl'
        except ImportError:
            try:
                import xlsxwriter
                engine = 'xlsxwriter'
                st.info("ℹ️ 'openpyxl' tidak terinstal, menggunakan 'xlsxwriter'.")
            except ImportError:
                st.error("❌ Instal `pip install openpyxl` untuk export Excel.")
                engine = None
        
        if engine:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine=engine) as writer:
                export_long.to_excel(writer, sheet_name='Forecast_Long', index=False)
                years = test_series['Year'].values
                wide_data = []
                for model, preds in all_predictions.items():
                    row = {'Model': model}
                    for i, y in enumerate(years):
                        row[str(y)] = preds.iloc[i] if i < len(preds) else None
                    wide_data.append(row)
                row_actual = {'Model': 'Aktual'}
                for i, y in enumerate(years):
                    row_actual[str(y)] = test_series['Value'].iloc[i]
                wide_data.append(row_actual)
                df_wide_export = pd.DataFrame(wide_data)
                df_wide_export.to_excel(writer, sheet_name='Forecast_Wide', index=False)
                meta_df = pd.DataFrame({
                    'Parameter': ['Arah', 'Filter', 'Best Model', 'Tanggal Ekspor'],
                    'Nilai': [direction, display_name, st.session_state.best_model, datetime.now().strftime('%Y-%m-%d %H:%M')]
                })
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)
            excel_data = excel_buffer.getvalue()
            st.download_button(label="📥 Download Excel (dua sheet: Long & Wide)", data=excel_data, file_name=f"forecast_{direction}_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        
        st.markdown("### 🗄️ Export ke MySQL")
        with st.expander("🔐 Konfigurasi Koneksi MySQL", expanded=False):
            db_host = st.text_input("Host", "localhost")
            db_user = st.text_input("User", "root")
            db_pass = st.text_input("Password", type="password")
            db_name = st.text_input("Database Name", "trade_db")
            table_name = st.text_input("Table Name", f"forecast_{direction.lower()}")
            if st.button("📤 Upload ke MySQL", type="primary"):
                try:
                    import sqlalchemy
                    from sqlalchemy import create_engine
                    engine = create_engine(f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}")
                    export_long.to_sql(table_name, con=engine, if_exists='replace', index=False)
                    st.success(f"✅ Berhasil upload {len(export_long)} baris ke tabel `{table_name}` (long format).")
                except ImportError:
                    st.error("❌ Instal `pip install sqlalchemy pymysql`")
                except Exception as e:
                    st.error(f"❌ Gagal: {e}")
    
    with tab4:
        st.markdown("## 🔍 Structural Break Detection (Data Historis)")
        st.markdown("""
        **Apa itu Structural Break?**  
        Structural Break adalah perubahan signifikan pada tren data di suatu titik waktu di **masa lalu**. 
        Chow Test digunakan untuk mendeteksi apakah perubahan tersebut signifikan secara statistik.
        """)
        
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
                    years = series_full['Year'].values
                    mid_idx = len(years) // 2
                    break_year = years[mid_idx]
                    st.warning(f"""
                    📌 **Break point terdeteksi sekitar tahun {break_year}.**  
                    Ini menandakan ada perubahan signifikan pada tren ekonomi di tahun tersebut, 
                    yang mungkin disebabkan oleh:
                    - Krisis ekonomi global (2008, 2020)
                    - Perubahan kebijakan perdagangan
                    - Pandemi COVID-19
                    - Perubahan harga komoditas global
                    """)
                else:
                    st.success("✅ Tidak terdeteksi structural break signifikan pada data historis.")
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
        - Break point yang terdeteksi biasanya disebabkan oleh faktor eksternal seperti krisis ekonomi, pandemi, atau perubahan kebijakan perdagangan.
        """)
        
        if best_model_name and forecast_horizon > 0:
            st.markdown("---")
            st.markdown("### 📊 Proyeksi Forecast (Skenario Baseline)")
            pred_vals, lower, upper = forecast_future(best_model_name, series_full, forecast_horizon)
            if pred_vals is not None:
                fig_scenario = go.Figure()
                fig_scenario.add_trace(go.Scatter(x=series_full['Year'], y=series_full['Value'],
                                                 mode='lines+markers', name='Data Historis',
                                                 line=dict(color='blue', width=2)))
                fig_scenario.add_trace(go.Scatter(x=pred_vals.index, y=pred_vals,
                                                 mode='lines+markers', name=f'Forecast ({best_model_name})',
                                                 line=dict(color='red', width=2, dash='dash')))
                if lower is not None and upper is not None:
                    x_vals = pd.Series(pred_vals.index)
                    y_upper = upper
                    y_lower = lower
                    fig_scenario.add_trace(go.Scatter(
                        x=pd.concat([x_vals, x_vals[::-1]]),
                        y=pd.concat([y_upper, y_lower[::-1]]),
                        fill='toself', fillcolor='rgba(255,0,0,0.15)',
                        line=dict(color='rgba(255,0,0,0)'),
                        name='Confidence Interval (95%)'
                    ))
                fig_scenario.update_layout(
                    height=350,
                    template='plotly_white',
                    title=f"Proyeksi {forecast_horizon} Tahun ke Depan"
                )
                st.plotly_chart(fig_scenario, use_container_width=True)
                st.caption("""
                **Catatan:** Structural Break hanya berlaku untuk data historis (masa lalu). 
                Forecast di atas adalah proyeksi model untuk masa depan, tetapi tidak dapat diuji structural break-nya 
                karena data belum terjadi.
                """)
    
    with tab5:
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
                    with st.expander(title, expanded=False):
                        st.markdown(content)
            elif i % 3 == 1:
                with col2:
                    with st.expander(title, expanded=False):
                        st.markdown(content)
            else:
                with col3:
                    with st.expander(title, expanded=False):
                        st.markdown(content)

else:
    if not run_btn:
        st.info("👈 Atur parameter di sidebar dan klik **Jalankan Full Analysis** untuk memulai.")

# ================== FOOTER ==================
st.markdown("---")
st.caption(f"Python 3.12 | {len(df_raw):,} baris data")

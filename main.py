import requests
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import ta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools


# Main Title
st.title('Forecasting Tool for Swing Trading')
st.subheader('by Shane Peterson')

### Ticker Selection Searchbar
st.subheader('-- Choose a Stock --')
selected_stock = st.text_input("Enter Symbol", value="goog").upper()

# Get Ticker Metadata
# ------------------------------------------------------------------
YOUR_API_TOKEN = "675c95da1bcad9.78017425" 
EXCHANGE_CODE = "US" 

url = f'https://eodhd.com/api/exchange-symbol-list/{EXCHANGE_CODE}?api_token={YOUR_API_TOKEN}&fmt=json'

try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    tickers_data = response.json()

    # Extract ticker symbols from the response (structure might vary)
    tickers = [item['Code'] for item in tickers_data] 

    # print(tickers)

except requests.exceptions.RequestException as e:
    data_load_state = st.text(f"-- Error fetching data: {e} --")

# ------------------------------------------------------------------
# Extract ticker name using symbol
# ------------------------------------------------------------------
for item in tickers_data:
    if item['Code'] == selected_stock:
        ticker_name = item['Name']
    
# Get ticker raw data
@st.cache_data # caches data from different tickers
def load_data(ticker):
    data = yf.download(ticker, period='max') # returns relevant data in df
    data.reset_index(inplace=True) # reset multindex, output is index list of tuples
    cols = list(data.columns) # convert index to list
    cols[0] = ('Date', '') 
    cols = [i[0] for i in cols] # return first element of cols tuples
    data.columns = cols # set as column names
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    return data

# Check input against list of tickers
data_load_state = st.text("-- Loading Data... --")
if selected_stock in tickers:
    data = load_data(selected_stock)
    data_load_state.text(f"-- {ticker_name} Data Loaded. --")
else:
    data_load_state.text(f"-- {selected_stock} not a valid Symbol. Please enter a symbol from the picker in the Appendix below. --")

# Change Data to datetime64[ns] datatype
data.Date = pd.to_datetime(data.Date)
data.Date = data.Date.astype('datetime64[ns]')

# ------------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------------

# Get Ticker Stats
# ------------------------------------------------------------------
stats = {}
for item in tickers_data:
    if item['Code'] == selected_stock:
        stats['Symbol'] = item['Code']
        # stats['Name'] = item['Name']
        stats['Current Price'] = round(data.Close.iloc[-1], 2)
        stats['Current Volume'] = data.Volume.iloc[-1]
        data['daily_returns'] = data.Close.pct_change()
        volatility = data.daily_returns.std() * np.sqrt(252)
        if volatility < 0.2:
            category = "Low"
            percentiles=(0.15, 0.85)
        elif volatility < 0.4:
            category = "Medium-Low"
            percentiles=(0.1, 0.9)
        elif volatility < 0.6:
            category = "Medium"
            percentiles=(0.1, 0.9)
        elif volatility < 0.8:
            category = "Medium-High"
            percentiles=(0.05, 0.95)
        else:
            category = "High"
            percentiles=(0.05, 0.95)
        stats['Annualized Volatility'] = category
        stats['Percentage Change'] = str(round(data['daily_returns'].mean() * 100, 4)) + ' %'
        stats['IPO'] = min(data.Date)
        stats['Historical Low'] = round(min(data.Low), 2)
        stats['HL Date'] = data.Date[data.Low.idxmin()]
        stats['Historical High'] = round(max(data.High), 2)
        stats['HH Date'] = data.Date[data.High.idxmax()]
stats_window_df = pd.DataFrame(stats, index=[0])

# Get Stock Age & Set training & forecast periods
# ------------------------------------------------------------------
if len(data)/365 < 8:
    period_unit = int(len(data)/4)
    forecast_period = period_unit
    train_period = len(data)
    stock_age = 'young'
else:
    period_unit = 365
    forecast_period = period_unit
    train_period = forecast_period * 4 if volatility < 0.6 else forecast_period * 8
    stock_age = 'seasoned'

# Get stats window
st.write(stats_window_df)

# ------------------------------------------------------------------
# Process Indicators
# ------------------------------------------------------------------
data['SMA50'] = data['Close'].rolling(window=50).mean()
indicator_bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
data['bb_upper'] = indicator_bb.bollinger_hband()
data['bb_lower'] = indicator_bb.bollinger_lband()

# ------------------------------------------------------------------
# Function & indicators for raw data candlestick graph
# ------------------------------------------------------------------
@st.cache_resource
def plot_raw_data(data):
    fig = go.Figure()
    # Add candlestick trace
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'))
    # Add SMA trace
    fig.add_trace(go.Scatter(x=data['Date'], 
                            y=data['SMA50'], 
                            name='SMA50', 
                            line=dict(color='black', width=2, dash='dash'),
                            visible='legendonly'))
    # Add upper Bollinger Band
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['bb_upper'], 
                             line=dict(color='red', width=1), 
                             name='Upper BB',
                             visible='legendonly'))
    # Add lower Bollinger Band
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['bb_lower'], 
                             line=dict(color='green', width=1), 
                             name='Lower BB',
                             visible='legendonly'))
    # Labels
    fig.layout.update(title_text=f"Time Series Data: {ticker_name} '{selected_stock}'",
                xaxis_rangeslider_visible=True,
                yaxis_title='Price',
                xaxis_title='Date')
    # Calculate default date range
    if stock_age == 'seasoned':
        end_date = data['Date'].max()
        start_date = end_date - pd.Timedelta(days=365)
        fig.update_xaxes(range=[start_date, end_date])

    st.plotly_chart(fig)
# Plot raw data
plot_raw_data(data)

# Graph notes
# ------------------------------------------------------------------
st.subheader('-- Chart Tips --')
with st.expander('Click here to expand'):
    st.write('* Use the slider (above) to select a date range')
    st.write('* Click items in the legend to show/hide indicators')
    st.write('* Hover in the upper-right corner of graph to reveal controls. Go fullscreen and explore!')

# ------------------------------------------------------------------
# FORECASTING
# ------------------------------------------------------------------

# Windsorize Function
# ------------------------------------------------------------------
def dynamic_winsorize(df, column, window_size=30, percentiles=percentiles):
    """
    Winsorizes data within a rolling window.

    Args:
        df: DataFrame containing the data.
        column: Name of the column to winsorize.
        window_size: Size of the rolling window.
        percentiles: Tuple containing the lower and upper percentiles.

    Returns:
        DataFrame with the winsorized column.
    """

    df['rolling_lower'] = df[column].rolling(window=window_size).quantile(percentiles[0])
    df['rolling_upper'] = df[column].rolling(window=window_size).quantile(percentiles[1])

    df['winsorized'] = df[column]
    df.loc[df[column] < df['rolling_lower'], 'winsorized'] = df['rolling_lower']
    df.loc[df[column] > df['rolling_upper'], 'winsorized'] = df['rolling_upper']

    return df

# Apply dynamic winsorization to raw data
data = dynamic_winsorize(data, 'Close')

# Get training data
# ------------------------------------------------------------------
df_train = data[['Date', 'Close', 'winsorized']]
df_train = df_train.rename(columns={'Date': 'ds'})

df_train = df_train[-train_period:] 

# Lambda function for cross validation metrics
# ------------------------------------------------------------------
cv_func = lambda model_name: cross_validation(model_name, 
                                              initial=f'{train_period} days', 
                                              period=f'{period_unit} days', 
                                              horizon=f'{forecast_period} days')

# Get metrics for baseline & winsorized models
# ------------------------------------------------------------------

scores_df = pd.DataFrame(columns=['mse', 'rmse', 'mae', 'mape'])

@st.cache_resource
def model_drafts(df_train, scores_df=scores_df):
    for i in ['Close', 'winsorized']:
        m = Prophet()
        df_train_renamed = df_train[['ds', i]].rename(columns={i: 'y'})
        m.fit(df_train_renamed)
        df_cv = cv_func(m)
        df_p = performance_metrics(df_cv, rolling_window=1)
        scores_df = pd.concat([scores_df, df_p[['mse', 'rmse', 'mae', 'mape']]], ignore_index=True)
    return scores_df

scores_df = model_drafts(df_train)

df_train = df_train.rename(columns={'winsorized': 'y'})

# Prepare for grid search of combos of all pararmeters
# ------------------------------------------------------------------
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}
# Generate combos of all pararmeters
# ------------------------------------------------------------------
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

@st.cache_resource
def tune_and_train_final_model(df_train, all_params, forecast_period, scores_df=scores_df):
    rmses = []
    for params in all_params:
        m = Prophet(**params).fit(df_train)
        df_cv = cv_func(m)
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    best_params_dict = dict(tuning_results.sort_values('rmse').reset_index(drop=True).drop('rmse', axis='columns').iloc[0])

    m = Prophet(**best_params_dict)
    m.fit(df_train)
    df_cv = cv_func(m)
    df_p = performance_metrics(df_cv, rolling_window=1)
    scores_df = pd.concat([scores_df, df_p[['mse', 'rmse', 'mae', 'mape']]], ignore_index=True)
    future = m.make_future_dataframe(periods=forecast_period)
    forecast = m.predict(future)
    
    return m, scores_df, forecast, best_params_dict

m, scores_df, forecast, best_params_dict = tune_and_train_final_model(df_train, all_params, forecast_period)

# Merge entire forecast w actual data & indicators
# ------------------------------------------------------------------
forecast_candlestick_df = pd.merge(
    left=data,
    right=forecast,
    right_on='ds',
    left_on='Date',
    how='right')[['ds', 'Open', 'High', 'Low', 'Close', 'yhat', 'yhat_lower', 'yhat_upper', 'SMA50', 'bb_upper', 'bb_lower']]
forecast_candlestick_df.rename(columns={'ds': 'Date'}, inplace=True) # keep naming convention and ds data. Date does not contain forecast date values.

# Get metrics 
# ------------------------------------------------------------------
scores_df.index = ['Baseline Model', 'Winsorized Model', 'Final Model']
scores_df = scores_df.reindex(sorted(scores_df.columns), axis=1)

st.write('-- Accuracy Metrics --')
st.write('Overall Accuracy:')
st.subheader(f'{100-(round(scores_df['mape'].iloc[2]*100, 2))}%')
st.write('')
st.dataframe(scores_df.loc[['Final Model']], width=500)

# Graph notes
# ------------------------------------------------------------------
st.write('-- Tips for Accuracy Metrics --')
with st.expander('Click here to expand'):
    st.write("In the context of time series forecasting, 'error' refers to the difference between the actual value of a variable at a specific point in time and the value predicted by a forecasting model. In this case, the metrics will specifically measure the error between the stock's closing price and the forecast trained on the closing price.")
    st.write(f"* Mean Absolute Error (MAE) - a MAE of {round(scores_df['mae'].iloc[2], 4)} implies that, on average, the model's predictions are off by approximately ${round(scores_df['mae'].iloc[2], 2)}.")
    st.write(f"* Mean Absolute Percentage Error (MAPE) - a MAPE of {round(scores_df['mape'].iloc[2], 4)} means that, on average, the model's predictions are {round(scores_df['mape'].iloc[2] * 100, 2)}% off from the actual values.")
    st.write('* Mean Squared Error (MSE) - this squares the errors, giving more weight to larger errors. A lower MSE indicates better accuracy.')
    st.write(f"* Root Mean Squared Error (RMSE) -  The square root of MSE. It is in the same units as the original data, making it easier to interpret. The RMSE of {round(scores_df['rmse'].iloc[2], 4)} suggests that the model's predictions can deviate from the actual values by up to ${round(scores_df['rmse'].iloc[2], 2)} in some cases.")

# Function & Indicators for Forecasted Candlestick Graph
# ------------------------------------------------------------------
@st.cache_resource
def plot_forecast(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], 
                         y=data['yhat_lower'], 
                         line=dict(color='lightblue', width=2), 
                         name='Forecast Lower Bound'))
    # Add upper forecast bound (yhat_upper)
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['yhat_upper'], 
                             line=dict(color='lightblue', width=2), 
                             name='Forecast Upper Bound',
                             fill='tonextx', # Fill between forecast and upper bound
                             fillcolor=None,
                             opacity=0.01))
    # Add forecast line (yhat)
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['yhat'], 
                             line=dict(color='blue', width=2), 
                             name='Forecast',
                             mode='lines'))
    # Add upper Bollinger Band
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['bb_upper'], 
                             line=dict(color='red', width=1), 
                             name='Upper BB',
                             visible='legendonly'))
    # Add lower Bollinger Band
    fig.add_trace(go.Scatter(x=data['Date'], 
                             y=data['bb_lower'], 
                             line=dict(color='green', width=1), 
                             name='Lower BB',
                             visible='legendonly'))
    # Add SMA trace
    fig.add_trace(go.Scatter(x=data['Date'], 
                            y=data['SMA50'], 
                            name='SMA50', 
                            line=dict(color='black', width=2, dash='dash'),
                            visible='legendonly'))
    # Add candlestick trace
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'))
    # Labels
    fig.layout.update(
        title_text=f"Time Series Data: {ticker_name} '{selected_stock}'",
                xaxis_rangeslider_visible=True,
                yaxis_title='Price',
                xaxis_title='Date')
    # Calculate default date range
    end_date = data['Date'].max()
    start_date = end_date - pd.Timedelta(days=period_unit*((train_period/period_unit)-1))
    fig.update_xaxes(range=[start_date, end_date])
    st.plotly_chart(fig)

plot_forecast(forecast_candlestick_df)

# ------------------------------------------------------------------
### APPENDIX
# ------------------------------------------------------------------
about_the_app = """
-- The App --

As a passionate swing trader, I developed this application to streamline my decision-making process. It leverages fundamental data science concepts, including data engineering and analytics, to provide actionable insights.

The app features a user-friendly interface with a candlestick chart, Bollinger Bands, and a Simple Moving Average (SMA) for visual analysis of price trends. I've also integrated a proprietary forecasting tool that analyzes historical patterns to generate mathematically driven predictions.

By selecting a stock ticker, the app displays key performance indicators (KPIs) like historical highs/lows, percentage change, volatility, and current price alongside the chart. This comprehensive tool empowers me to make more informed trading decisions and refine my trading strategies.

"""

about_the_model = f"""
-- The Model --

This application was designed to enhance my swing trading strategy. It utilizes a Prophet forecasting model, fine-tuned with techniques like winsorization and hyperparameter tuning to optimize its accuracy.

Key Model Enhancements:

* Adaptive Winsorization: The winsorization thresholds are dynamically adjusted based on the stock's volatility. Tighter thresholds are applied to less volatile stocks, while looser thresholds are used for more volatile stocks to better capture their price movements.
* Adaptive Training Data: The training data size is dynamically adjusted based on the stock's volatility and available data. For more volatile stocks, the model leverages a larger training window (when possible) to capture historical patterns more effectively.

By combining these refinements with a cross-validated grid search to optimize changepoint_prior_scale and seasonality_prior_scale (e.g., for '{selected_stock}', optimal values are: changepoint_prior_scale: {best_params_dict["changepoint_prior_scale"]:.3f}, seasonality_prior_scale: {best_params_dict["seasonality_prior_scale"]:.3f}), this application provides me with a robust forecasting tool that can be used in conjunction with visual aids like candlestick charts, Bollinger Bands, and SMAs to identify potential entry and exit points for swing trades.

Cross-validation ensures that the hyperparameters selected are not overfitted to a specific subset of the data. By evaluating the model's performance on multiple subsets of the data during the grid search, we can select hyperparameters that generalize better to unseen data and potentially improve the model's out-of-sample performance.
"""
about_swing_trading = """
-- Swing Trading --

This application was designed to enhance my swing trading strategy. Swing trading focuses on capturing short-term price movements, and this tool provides me with valuable insights.

I've integrated a Prophet forecasting model, fine-tuned with techniques like winsorization and hyperparameter tuning to optimize its accuracy. The model dynamically adjusts its training data based on the stock's volatility, aiming to improve prediction accuracy.

By combining these analytical tools with visual aids like candlestick charts, Bollinger Bands, and SMAs, I'm able to identify potential entry and exit points with greater confidence, ultimately refining my trading decisions.

"""

st.subheader('-- Appendix --') # button to hide / unhide
with st.expander('Click here to expand'):
    st.subheader('-- Ticker List --') # button to hide / unhide
    st.write(pd.DataFrame(tickers_data))
    st.subheader('-- Raw Data --') # button to hide / unhide
    st.write(data)
    st.subheader('-- Forecast Grid --')
    st.write(forecast)
    st.subheader('-- Forecast Components --')
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    st.subheader('-- Model Iterations --')
    st.write('-- Baseline Model --')
    st.dataframe(scores_df.loc[['Baseline Model']], width=500)
    st.write('-- Winsorized Model --')
    st.dataframe(scores_df.loc[['Winsorized Model']], width=500)
    st.write('-- Final Model --')
    st.dataframe(scores_df.loc[['Final Model']], width=500)
    st.subheader('-- About --')
    st.write(about_the_app)
    st.write(about_the_model)
    st.write(about_swing_trading)
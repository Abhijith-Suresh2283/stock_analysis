import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from stocknews import StockNews
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas_ta as ta
from datetime import datetime, timedelta
import requests

def calculate_risk_metrics(data):
    # Ensure data is properly indexed and sorted
    data = data.sort_index()
    
    # Calculate returns and clean up any NaN values
    returns = data['Close'].pct_change()
    returns = returns.dropna()
    
    # Calculate base metrics with error handling
    try:
        volatility = float(returns.std() * np.sqrt(252))  # Annualized volatility
        sharpe_ratio = float((returns.mean() * 252) / volatility)  # Sharpe ratio
        var_95 = float(np.percentile(returns, 5))  # 95% Value at Risk
        max_drawdown = float(((data['Close'].cummax() - data['Close'])/data['Close'].cummax()).max())
    except Exception as e:
        st.error(f"Error calculating basic metrics: {str(e)}")
        return None
    
    beta = 1.0  # Default beta value
    
    try:
        # Ensure index is timezone-naive
        if data.index.tz is not None:
            data_index = data.index.tz_localize(None)
        else:
            data_index = data.index
            
        # Get market data for the exact same period
        market_data = yf.download('^GSPC', 
                                start=data_index[0] - pd.Timedelta(days=5),  # Add buffer for market holidays
                                end=data_index[-1] + pd.Timedelta(days=5),
                                progress=False)
        
        # Calculate market returns
        market_returns = market_data['Close'].pct_change()
        market_returns = market_returns.dropna()
        
        # Create a date range that exists in both series
        common_dates = returns.index.intersection(market_returns.index)
        
        # Reindex both series to common dates
        stock_returns_aligned = returns[common_dates]
        market_returns_aligned = market_returns[common_dates]
        
        # Verify we have data to work with
        if len(stock_returns_aligned) > 0:
            # Convert to numpy arrays for calculation
            stock_arr = stock_returns_aligned.values
            market_arr = market_returns_aligned.values
            
            # Calculate beta
            covariance = np.cov(stock_arr, market_arr)[0,1]
            market_variance = np.var(market_arr)
            
            if market_variance != 0:
                beta = float(covariance / market_variance)
            else:
                st.warning("Market variance is zero, using default beta value")
            
    except Exception as e:
        st.warning(f"Could not calculate beta: {str(e)}. Using default value of 1.0")
    
    # Verify all metrics are valid numbers
    metrics = {
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Value at Risk (95%)': var_95,
        'Maximum Drawdown': max_drawdown,
        'Beta': beta
    }
    
    # Check for invalid values
    for key, value in metrics.items():
        if not np.isfinite(value):  # Check for inf or nan
            metrics[key] = 0.0
            st.warning(f"Invalid {key} value detected, using 0.0")
    
    return metrics


def get_risk_level(metrics):
    try:
        # Ensure all values are valid floats
        volatility_score = min(max(float(metrics['Volatility']) * 100 / 50, 0), 1)
        var_score = min(max(abs(float(metrics['Value at Risk (95%)'])) * 100 / 10, 0), 1)
        drawdown_score = min(max(float(metrics['Maximum Drawdown']) / 0.5, 0), 1)
        beta_score = min(max(abs(float(metrics['Beta'])) / 2, 0), 1)
        
        # Calculate overall risk score (0-1)
        risk_score = (volatility_score + var_score + drawdown_score + beta_score) / 4
        
        # Define risk levels
        if risk_score < 0.3:
            return "Low", "🟢", risk_score
        elif risk_score < 0.7:
            return "Moderate", "🟡", risk_score
        else:
            return "High", "🔴", risk_score
            
    except Exception as e:
        st.error(f"Error calculating risk level: {str(e)}")
        return "Unknown", "⚪", 0.0

# Function to fetch USD to INR conversion rate
def get_usd_to_inr_rate():
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD")
        data = response.json()
        return data["rates"]["INR"]
    except Exception as e:
        st.warning(f"Could not fetch current exchange rate: {e}. Using fallback rate.")
        return 83.0  # Fallback rate if API fails
    
USD_TO_INR = get_usd_to_inr_rate()

def get_popular_tickers():
    # List of popular stock tickers
    popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    popular_stocks = []
    
    for ticker_symbol in popular_tickers:
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                price_usd = info['regularMarketPrice']
                price_inr = price_usd * USD_TO_INR
                name = info.get('shortName', ticker_symbol)
                
                popular_stocks.append({
                    'Symbol': ticker_symbol,
                    'Name': name,
                    'Price (₹)': f"₹{price_inr:.2f}"
                })
        except Exception as e:
            continue
            
    return pd.DataFrame(popular_stocks)
st.title('Stock Dashboard')

# Sidebar for user input
st.sidebar.title("Settings")
ticker = st.sidebar.text_input('Ticker', value='AAPL')

# Display popular tickers in sidebar
st.sidebar.title("Popular Tickers")
try:
    popular_df = get_popular_tickers()
    if not popular_df.empty:
        st.sidebar.dataframe(popular_df, hide_index=True)
except Exception as e:
    st.sidebar.warning(f"Could not fetch popular tickers: {str(e)}")


if ticker:
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get 1 year of historical data

    # Fetch stock data and currency info
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    # Get currency from Yahoo Finance
    stock_info = stock.info
    currency = stock_info.get('currency', 'USD')  # Default to USD if missing

    # Fetch exchange rate dynamically (only if the stock is in USD)
    converted = False
    if currency == 'USD':
        exchange_rate = get_usd_to_inr_rate()
        data['Close'] = data['Close'] * exchange_rate
        converted = True

    if not data.empty:
        # Get current stock price using proper pandas indexing
        current_price = data['Close'].iloc[-1].item()  # Convert to native Python float
        
        # Set currency symbol dynamically
        currency_symbol = "₹"

        # Display current stock price with styling
        st.markdown("""
        <style>
        .big-font {
            font-size:24px !important;
            font-weight:bold;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f"<p class='big-font'>{ticker} Current Price: {currency_symbol}{current_price:.2f}</p>", unsafe_allow_html=True)

        # Display stock price chart
        fig = px.line(data.reset_index(), x='Date', y='Close', title=f'{ticker} Stock Price')
        st.plotly_chart(fig)

        # Tabs setup
        pricing_data, risk_analysis, news, tech_indicator, prediction = st.tabs(["Pricing Data", "Risk Analysis", "Top 10 News", "Tech Indicators", "Stock Price Prediction"])
        # Pricing Data tab
        with pricing_data:
            st.header('Price Movements')
            data2 = data.copy()
            data2['% Change'] = data2['Close'].pct_change()
            data2.dropna(inplace=True)
            st.write(data2)
            annual_return = data2['% Change'].mean() * 252 * 100
            st.write(f'Annual Return: {annual_return:.2f}%')
            stdev = np.std(data2['% Change']) * np.sqrt(252)
            st.write(f'Standard Deviation: {stdev * 100:.2f}%')

        # News tab
        with news:
            st.header(f'News of {ticker}')
            try:
                sn = StockNews(ticker, save_news=False)
                df_news = sn.read_rss()
                for i in range(min(10, len(df_news))):
                    st.subheader(f'News {i+1}')
                    st.write(df_news['published'][i])
                    st.write(df_news['title'][i])
                    st.write(df_news['summary'][i])
                    st.write(f"Title Sentiment: {df_news['sentiment_title'][i]}")
                    st.write(f"News Sentiment: {df_news['sentiment_summary'][i]}")
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
        
        # Technical Indicators tab
        with tech_indicator:
            st.subheader('Technical Analysis Dashboard:')
            
            if {'Low', 'High', 'Open', 'Volume'}.issubset(data.columns):
                indicators = {
                    'Moving Average (SMA)': ta.sma,
                    'Exponential Moving Average (EMA)': ta.ema,
                    'Relative Strength Index (RSI)': ta.rsi,
                    'MACD': ta.macd,
                    'Bollinger Bands': ta.bbands
                }
                
                st.write("Available Technical Indicators:", list(indicators.keys()))
                
                technical_indicator = st.selectbox('Tech Indicator', options=list(indicators.keys()))
                method = indicators.get(technical_indicator, None)
                
                if method:
                    try:
                        if technical_indicator in ['Moving Average (SMA)', 'Exponential Moving Average (EMA)']:
                            indicator = method(data['Close'], length=20)
                        else:
                            indicator = method(data['Close'])
                        
                        if isinstance(indicator, pd.DataFrame):
                            indicator_df = indicator
                        else:
                            indicator_df = pd.DataFrame(indicator)
                        
                        indicator_df['Close'] = data['Close'].astype(float)  # Ensure numeric type
                        indicator_df = indicator_df.astype(float)  # Convert entire DataFrame to float
                        indicator_df.index = data.index  # Set index to match dates

                        # Create Plot
                        fig_ind = px.line(indicator_df.reset_index(), x='Date', y=indicator_df.columns, title=f"{technical_indicator} for {ticker}")

                        st.plotly_chart(fig_ind)
                        st.write(indicator_df)
                    except Exception as e:
                        st.error(f"Error applying indicator: {e}")
                else:
                    st.error(f"Indicator {technical_indicator} not found.")

            else:
                st.error("Missing columns in the data for technical analysis.")

            # Add Trading Signals Section
            st.subheader("🔔 Trading Signals & Alerts")
            
            # Calculate Signals
            data['SMA_50'] = ta.sma(data['Close'], length=50)
            data['SMA_200'] = ta.sma(data['Close'], length=200)
            data['RSI'] = ta.rsi(data['Close'], length=14)
            
            latest_price = data['Close'].iloc[-1]
            latest_sma_50 = data['SMA_50'].iloc[-1]
            latest_sma_200 = data['SMA_200'].iloc[-1]
            latest_rsi = data['RSI'].iloc[-1]
            
            buy_signal = False
            sell_signal = False

            # Buy Signal Condition (Golden Cross + RSI)
            if latest_sma_50 > latest_sma_200 and latest_rsi < 30:
                buy_signal = True
                st.success(f"📈 BUY Alert: {ticker} is in an uptrend (SMA 50 > SMA 200) and RSI is oversold ({latest_rsi:.2f}).")

            # Sell Signal Condition (Death Cross + RSI)
            elif latest_sma_50 < latest_sma_200 and latest_rsi > 70:
                sell_signal = True
                st.error(f"📉 SELL Alert: {ticker} is in a downtrend (SMA 50 < SMA 200) and RSI is overbought ({latest_rsi:.2f}).")

            # No Strong Signal
            else:
                st.info("⚡ No strong Buy/Sell signals detected at the moment.")


        # Add Risk Analysis tab
        with risk_analysis:
            st.header('Risk Analysis')
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(data)
            risk_level, risk_emoji, risk_score = get_risk_level(risk_metrics)
            
            # Display risk level with custom styling
            st.markdown(f"""
            <div style='background-color: {"#E8F5E9" if risk_level == "Low" else "#FFF3E0" if risk_level == "Moderate" else "#FFEBEE"}; 
                        padding: 20px; 
                        border-radius: 10px; 
                        margin-bottom: 20px'>
                <h2 style='color: black;'>{risk_emoji} Risk Level: {risk_level}</h2>
                <p style='font-size: 16px; color: black;'>Risk Score: {risk_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create two columns for metrics display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Key Risk Metrics")
                st.write(f"📊 Volatility: {risk_metrics['Volatility']:.2%}")
                st.write(f"📈 Beta: {risk_metrics['Beta']:.2f}")
                st.write(f"📉 Maximum Drawdown: {risk_metrics['Maximum Drawdown']:.2%}")
            
            with col2:
                st.subheader("Additional Metrics")
                st.write(f"💹 Sharpe Ratio: {risk_metrics['Sharpe Ratio']:.2f}")
                st.write(f"⚠️ Value at Risk (95%): {risk_metrics['Value at Risk (95%)']:.2%}")
            
            # Add risk metric explanations
            with st.expander("Understanding Risk Metrics"):
                st.markdown("""
                * **Volatility**: Measures the stock's price fluctuation. Higher volatility indicates more uncertainty.
                * **Beta**: Measures the stock's sensitivity to market movements. Beta > 1 means more volatile than the market.
                * **Maximum Drawdown**: The largest peak-to-trough decline. Shows worst-case historical loss.
                * **Sharpe Ratio**: Risk-adjusted return measure. Higher is better, negative values are concerning.
                * **Value at Risk**: The potential loss in value of a portfolio over a period. Shows downside risk.
                """)

        # Stock Price Prediction tab
        with prediction:
            st.subheader("Stock Price Prediction")

            if len(data) < 100:
                st.error("Not enough data for prediction. Please try again later.")
            else:
                try:
                    # Prepare data for prediction
                    close_prices = data['Close'].values.reshape(-1, 1)
                    
                    # Scale the data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(close_prices)

                    # Prepare sequences for prediction
                    sequence_length = 100
                    sequences = [scaled_data[i:(i + sequence_length)] for i in range(len(scaled_data) - sequence_length)]
                    sequences = np.array(sequences)

                    # Load the model
                    model = load_model('keras_model.h5')

                    # Prepare last sequence for prediction
                    last_sequence = scaled_data[-100:].reshape(1, 100, 1)

                    # Make prediction for next month
                    predicted_scaled = model.predict(last_sequence)
                    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0].item()  # Convert to native Python float

                    # Calculate price change
                    price_change = predicted_price - current_price
                    price_change_percentage = (price_change / current_price) * 100

                    future_date = (end_date + timedelta(days=30)).strftime('%Y-%m-%d')
                    
                    # Display prediction analysis
                    st.subheader('Price Prediction Analysis')
                    
                    color = "#90EE90" if price_change > 0 else "#FFB6C0"
                    emoji = "📈" if price_change > 0 else "📉"
                    trend = "increase" if price_change > 0 else "decrease"

                    st.markdown(f"""
                        <div style='background-color: {color}; padding: 20px; border-radius: 5px;'>
                            <h3>{emoji} Price Expected to {trend.capitalize()}</h3>
                            <p>Current Price: {currency_symbol}{current_price:.2f}</p>
                            <p>Predicted Price (By next month): {currency_symbol}{predicted_price:.2f}</p>
                            <p>The stock price is predicted to {trend} by {abs(price_change_percentage):.2f}% in the next month.</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

    else:
        st.write("No data found for the given ticker.")
else:
    st.write("Please enter a valid ticker.")

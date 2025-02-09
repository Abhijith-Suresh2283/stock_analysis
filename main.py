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

# Streamlit app setup
st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker', value='AAPL')

if ticker:
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get 1 year of historical data

    # Fetch stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if not data.empty:
        # Get current stock price
        current_price = data['Adj Close'][-1]
        
        # Display current stock price with styling
        st.markdown("""
        <style>
        .big-font {
            font-size:24px !important;
            font-weight:bold;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f"<p class='big-font'>{ticker} Current Price: ${current_price:.2f}</p>", unsafe_allow_html=True)

        # Display stock price chart
        fig = px.line(data, x=data.index, y='Adj Close', title=f'{ticker} Stock Price')
        st.plotly_chart(fig)

        # Tabs setup
        pricing_data, news, tech_indicator, prediction = st.tabs(["Pricing Data", "Top 10 News", "Tech Indicators", "Stock Price Prediction"])

        # Pricing Data tab
        with pricing_data:
            st.header('Price Movements')
            data2 = data.copy()
            data2['% Change'] = data2['Adj Close'] / data2['Adj Close'].shift(1) - 1
            data2.dropna(inplace=True)
            st.write(data2)
            annual_return = data2['% Change'].mean() * 252 * 100
            st.write('Annual Return is', annual_return, '%')
            stdev = np.std(data2['% Change']) * np.sqrt(252)
            st.write('Standard Deviation is', stdev * 100, '%')

        # News tab
        with news:
            st.header(f'News of {ticker}')
            sn = StockNews(ticker, save_news=False)
            df_news = sn.read_rss()
            for i in range(min(10, len(df_news))):
                st.subheader(f'News {i+1}')
                st.write(df_news['published'][i])
                st.write(df_news['title'][i])
                st.write(df_news['summary'][i])
                title_sentiment = df_news['sentiment_title'][i]
                st.write(f'Title Sentiment: {title_sentiment}')
                news_sentiment = df_news['sentiment_summary'][i]
                st.write(f'News Sentiment: {news_sentiment}')
        
        # Technical Indicators tab
        with tech_indicator:
            st.subheader('Technical Analysis Dashboard:')
            if 'Low' in data.columns and 'High' in data.columns and 'Open' in data.columns and 'Volume' in data.columns:
                indicators = {
                    'Moving Average': ta.sma,
                    'Exponential Moving Average': ta.ema,
                    'Relative Strength Index': ta.rsi,
                    'MACD': ta.macd,
                    'Bollinger Bands': ta.bbands
                }
                
                st.write("Available Technical Indicators:", list(indicators.keys()))
                
                technical_indicator = st.selectbox('Tech Indicator', options=list(indicators.keys()))
                method = indicators.get(technical_indicator, None)
                
                if method:
                    try:
                        if technical_indicator == 'Moving Average':
                            indicator = method(data['Close'], length=20)
                        elif technical_indicator == 'Exponential Moving Average':
                            indicator = method(data['Close'], length=20)
                        elif technical_indicator == 'Relative Strength Index':
                            indicator = method(data['Close'])
                        elif technical_indicator == 'MACD':
                            indicator = method(data['Close'])
                        elif technical_indicator == 'Bollinger Bands':
                            indicator = method(data['Close'])
                        
                        if isinstance(indicator, pd.DataFrame):
                            indicator_df = indicator
                        else:
                            indicator_df = pd.DataFrame(indicator)
                        
                        indicator_df['Close'] = data['Close']
                        figW_ind_new = px.line(indicator_df)
                        st.plotly_chart(figW_ind_new)
                        st.write(indicator_df)
                    except Exception as e:
                        st.error(f"Error applying indicator: {e}")
                else:
                    st.error(f"Indicator {technical_indicator} not found.")
            else:
                st.error("Missing columns in the data for technical analysis.")
        # Stock Price Prediction tab
        with prediction:
            st.subheader("Stock Price Prediction")

            if len(data) < 100:
                st.error("Not enough data for prediction. Please try again later.")
            else:
                # Prepare data for prediction
                close_prices = data['Close'].values.reshape(-1, 1)
                
                # Scale the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(close_prices)

                # Prepare sequences for prediction
                sequence_length = 100
                sequences = []
                for i in range(len(scaled_data) - sequence_length):
                    sequences.append(scaled_data[i:(i + sequence_length)])
                sequences = np.array(sequences)

                # Load the model
                model = load_model('keras_model.h5')

                # Prepare the last sequence for prediction
                last_sequence = scaled_data[-100:]
                last_sequence = last_sequence.reshape(1, 100, 1)

                # Make prediction for next month
                predicted_scaled = model.predict(last_sequence)
                predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

                # Calculate price change
                price_change = predicted_price - current_price
                price_change_percentage = (price_change / current_price) * 100

                future_date = (end_date + timedelta(days=30)).strftime('%Y-%m-%d')
                
                # Display prediction analysis
                st.subheader('Price Prediction Analysis')
                
                # Create a colored box based on the prediction
                if price_change > 0:
                    st.markdown(f"""
                        <div style='background-color: #90EE90; padding: 20px; border-radius: 5px;'>
                            <h3>📈 Price Expected to Increase</h3>
                            <p>Current Price: ${current_price:.2f}</p>
                            <p>Predicted Price ({future_date}): ${predicted_price:.2f}</p>
                            <p>The stock price is predicted to increase by {abs(price_change_percentage):.2f}% in the next month.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='background-color: #FFB6C0; padding: 20px; border-radius: 5px;'>
                            <h3>📉 Price Expected to Decrease</h3>
                            <p>Current Price: ${current_price:.2f}</p>
                            <p>Predicted Price ({future_date}): ${predicted_price:.2f}</p>
                            <p>The stock price is predicted to decrease by {abs(price_change_percentage):.2f}% in the next month.</p>
                        </div>
                        """, unsafe_allow_html=True)

    else:
        st.write("No data found for the given ticker.")
else:
    st.write("Please enter a valid ticker.")
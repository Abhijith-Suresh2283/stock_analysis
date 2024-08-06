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

# Streamlit app setup
st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker', value='AAPL')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input('End Date')

# Validate inputs
if ticker and start_date < end_date:
    # Fetch stock data from yfinance
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if not data.empty:
        # Display stock price chart
        fig = px.line(data, x=data.index, y='Adj Close', title=ticker)
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
            for i in range(min(10, len(df_news))):  # Avoid index errors
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
                        
                        # Ensure the indicator result is a DataFrame
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

            # Use the same ticker, start date, and end date as above
            df = yf.download(ticker, start=start_date, end=end_date)

            st.subheader(f'Data from {start_date} to {end_date}')
            st.write(df.describe())

            st.subheader('Closing price vs time chart')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df['Close'])
            st.pyplot(fig)

            st.subheader('Closing price vs time chart with 100 MA')

            ma100 = df['Close'].rolling(100).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(ma100, label='100 MA')
            plt.plot(df['Close'], label='Close')
            plt.legend()
            st.pyplot(fig)

            st.subheader('Closing price vs time chart with 100 MA and 200 MA')
            ma100 = df['Close'].rolling(100).mean()
            ma200 = df['Close'].rolling(200).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(ma100, 'r', label='100 MA')
            plt.plot(ma200, 'g', label='200 MA')
            plt.plot(df['Close'], 'b', label='Close')
            plt.legend()
            st.pyplot(fig)

            data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

            x_train = []
            y_train = []

            for i in range(100, data_training_array.shape[0]):
                x_train.append(data_training_array[i-100:i])
                y_train.append(data_training_array[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)

            model = load_model('keras_model.h5')

            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            st.subheader('Prediction vs Original')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig)

    else:
        st.write("No data found for the given ticker and date range.")
else:
    st.write("Please enter a valid ticker and ensure the start date is before the end date.")

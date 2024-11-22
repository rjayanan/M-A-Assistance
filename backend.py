import streamlit as st
from groq import Groq
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Company to Ticker Conversion
def get_ticker_symbol(company_name):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}&quotesCount=1&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = json.loads(response.text)
        
        if 'quotes' in data and len(data['quotes']) > 0:
            ticker = data['quotes'][0]['symbol']
            company_name = data['quotes'][0]['shortname']
            return ticker, company_name
        else:
            return None, None
    except Exception as e:
        st.error(f"Error finding ticker symbol: {str(e)}")
        return None, None

# Extract ticker symbols from AI response
def extract_tickers(text):
    ticker_pattern = r'\(([A-Z]+)\)'
    tickers = re.findall(ticker_pattern, text)
    return tickers

# AI Integration with structured response format
def ai_request(message):
    formatted_message = f"""{message}
    Please format your response as a numbered list with exactly 3 suggestions.
    For each company, include:
    1. Company name
    2. Ticker symbol in parentheses
    3. Brief description
    
    Example format:
    1. Company Name (TICK) - Brief description
    2. Company Name (TICK) - Brief description
    3. Company Name (TICK) - Brief description"""
    
    client = Groq(
        api_key="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",
    )
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": formatted_message}],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Improved Stock Data Fetching
def get_stock_data(ticker_symbol):
    try:
        if not ticker_symbol or not isinstance(ticker_symbol, str):
            return None
            
        ticker_symbol = ticker_symbol.strip().upper()
        stock = yf.Ticker(ticker_symbol)
        stock_data = stock.history(period="1y")
        
        if stock_data.empty:
            st.warning(f"No historical data available for {ticker_symbol}")
            return None
        
        return stock_data
        
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker_symbol}: {str(e)}")
        return None

# LSTM Helper Functions
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def create_lstm_model(look_back):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_lstm_model(X_train, y_train, look_back):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    model = create_lstm_model(look_back)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model

# Stock Chart Display with LSTM
def display_stock_chart(stock_data, company_name, ticker_symbol):
    if stock_data is None or stock_data.empty:
        st.warning(f"No stock data available for {company_name} ({ticker_symbol})")
        return
    
    try:
        # Prepare data for LSTM
        look_back = 60
        close_prices = stock_data['Close']
        
        # Prepare LSTM data
        X, y, scaler = prepare_lstm_data(close_prices, look_back)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train LSTM model
        lstm_model = train_lstm_model(X_train, y_train, look_back)
        
        # Predict future prices
        last_look_back = close_prices[-look_back:].values
        last_look_back_scaled = scaler.transform(last_look_back.reshape(-1, 1))
        
        predictions = []
        current_batch = last_look_back_scaled.reshape((1, look_back, 1))
        
        for _ in range(30):
            current_pred = lstm_model.predict(current_batch)[0]
            predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], current_pred.reshape((1, 1, 1)), axis=1)
        
        predictions = scaler.inverse_transform(np.array(predictions))
        
        # Create prediction dates
        last_date = stock_data.index[-1]
        pred_dates = pd.date_range(start=last_date, periods=31)[1:]
        
        # Plotting
        fig = go.Figure()
        
        # Historical Close Price
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name='Historical Close Price',
                line=dict(color='#0066ff', width=2)
            )
        )
        
        # Moving Averages
        stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['MA20'],
                name='20-day MA',
                line=dict(color='#ff9900', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['MA50'],
                name='50-day MA',
                line=dict(color='#ff0000', width=1.5)
            )
        )
        
        # LSTM Predictions
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=predictions.flatten(),
                name='LSTM Price Prediction',
                line=dict(color='#00ff00', width=2, dash='dot'),
                mode='lines'
            )
        )
        
        fig.update_layout(
            title=f"{company_name} ({ticker_symbol}) Stock Performance with LSTM Prediction",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional model performance insights
        st.subheader("LSTM Prediction Insights")
        st.write("Next 30 Days Predicted Prices:")
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted Price': predictions.flatten()
        })
        st.dataframe(pred_df)
        
    except Exception as e:
        st.error(f"Error displaying chart or running LSTM prediction: {str(e)}")
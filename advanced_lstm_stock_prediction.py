import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import google.generativeai as genai
from sklearn.metrics import mean_squared_error

# Configure Google Generative AI
genai.configure(api_key="")  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Fetch stock data from Yahoo Finance
def fetch_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data['MA20'] = data['Close'].rolling(window=20).mean()  # Moving Average 20 Days
    data['MA50'] = data['Close'].rolling(window=50).mean()  # Moving Average 50 Days
    data['RSI'] = compute_RSI(data['Close'])  # Relative Strength Index (RSI)
    data = data[['Close', 'Volume', 'MA20', 'MA50', 'RSI']].dropna()
    return data

# Calculate RSI (Relative Strength Index)
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

# Prepare data for LSTM model
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Build a more complex LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot predictions
def plot_predictions(real, predicted, stock_symbol):
    plt.figure(figsize=(10, 6))
    plt.plot(real, label="Real Prices", color='blue')
    plt.plot(predicted, label="Predicted Prices", color='red')
    plt.title(f"{stock_symbol} Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

# Generate insights using Google Generative AI
def generate_insights(stock_symbol, mse, predictions):
    prompt = f"Generate insights on {stock_symbol} stock prediction model with an MSE of {mse}. Describe the future trends and risk management strategies for investors based on this model. Include technical indicators like moving averages and RSI."
    response = model.generate(prompt=prompt)
    return response.generations[0]['text']

# Streamlit app for frontend
def stock_prediction_app():
    st.title("Advanced LSTM Stock Prediction with AI Insights")

    # User input for stock symbol and dates
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('2023-01-01'))

    if st.button("Predict Stock Prices"):
        # Fetch and prepare stock data
        stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
        X, y, scaler = prepare_data(stock_data.values)

        # Split data into training and testing sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build and train the LSTM model
        lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

        # Predict stock prices
        predictions = lstm_model.predict(X_test)
        predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], X_test.shape[2] - 1))), axis=1))[:, 0]
        
        real_prices = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2] - 1))), axis=1))[:, 0]

        # Calculate MSE
        mse = mean_squared_error(real_prices, predictions)
        st.write(f"Mean Squared Error: {mse}")

        # Plot the real vs predicted stock prices
        st.write("### Stock Price Prediction")
        plot_predictions(real_prices, predictions, stock_symbol)

        # Generate insights using Google Generative AI
        st.write("### AI-Generated Insights")
        insights = generate_insights(stock_symbol, mse, predictions)
        st.write(insights)

# Run the Streamlit app
if __name__ == "__main__":
    stock_prediction_app()

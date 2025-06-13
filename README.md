# Stock-trend-Predictor
Stock Trend Predictor is a Flask web application that allows users to forecast future stock prices based on historical data using LSTM (Long Short-Term Memory) deep learning models. Users can select a stock ticker and a historical time period (e.g., last 6 months, 1 year, etc>).
# 📈 Stock Trend Predictor

A web application to predict future stock prices using an LSTM deep learning model built with Keras and TensorFlow, powered by Flask on the backend and Chart.js on the frontend for visualization.

---

## 🚀 Features

- Input any valid stock ticker (e.g., AAPL, MSFT, TSLA)
- Choose a historical time period:  
  `6 months`, `1 year`, `2 years`, `5 years`
- Automatically predicts future prices for the same number of days as in the selected period
- Forecast results are displayed in an interactive chart using Chart.js
- Handles invalid tickers or insufficient data gracefully

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Data Source:** Yahoo Finance (via `yfinance`)
- **Model:** LSTM (TensorFlow / Keras)
- **Libraries:** NumPy, Pandas, scikit-learn

---

## 🧠 How It Works

1. User inputs a stock ticker and selects a time range.
2. App fetches historical stock data using `yfinance`.
3. Data is scaled and sequences are prepared for LSTM.
4. A simple LSTM model is trained on the fly (for demo purposes).
5. Forecast is generated for the same number of days as the selected time period.
6. Output is scaled back to original values and plotted using Chart.js.

---

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/stock-trend-predictor.git
   cd stock-trend-predictor

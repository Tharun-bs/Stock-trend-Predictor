from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Map period to prediction days
PERIOD_TO_PRED_DAYS = {
    "6mo": 180,
    "1y": 360,
    "2y": 720,
    "5y": 1800,
}

@app.route("/", methods=["GET", "POST"])
def index():
    forecast = []
    stock = ''
    error = None
    prediction_days = 0

    if request.method == "POST":
        stock = request.form["ticker"].upper()
        period = request.form["period"]
        prediction_days = PERIOD_TO_PRED_DAYS.get(period, 7)

        data = yf.download(stock, period=period)

        if data.empty:
            error = "Invalid stock symbol or period"
            return render_template("index.html", error=error, forecast=[], stock=stock)

        close_prices = data["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        window_size = 60
        if len(scaled_data) <= window_size:
            error = "Not enough data to make predictions."
            return render_template("index.html", error=error, forecast=[], stock=stock)

        X, y = create_sequences(scaled_data, window_size)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        last_seq = scaled_data[-window_size:]
        input_seq = last_seq.reshape(1, window_size, 1)

        for _ in range(prediction_days):
            pred = model.predict(input_seq, verbose=0)[0][0]
            forecast.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten().tolist()

    return render_template("index.html", forecast=forecast, stock=stock, error=error, prediction_days=prediction_days)

if __name__ == "__main__":
    app.run(debug=True)

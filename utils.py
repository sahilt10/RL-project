import yfinance as yf
import pandas as pd

def load_price_data(ticker_str, start_date, end_date):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    try:
        data = yf.download(tickers, start=start_date, end=end_date, timeout=30)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data
    except Exception as e:
        print("⚠️ Error loading data:", e)
        return None

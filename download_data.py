import pandas as pd
import yfinance as yf

tickers = pd.read_csv("all_37_v1.csv").columns[1:]

data = pd.DataFrame()

for ticker in tickers:
    print(ticker)
    cur = yf.download(ticker + ".TW", start="2024-07-01", end="2024-07-02", auto_adjust=False)
    # tmp = cur["Adj Close"].pct_change().dropna()
    tmp = cur["Adj Close"].dropna()
    data[ticker] = tmp

data.to_csv("price_20240701.csv")

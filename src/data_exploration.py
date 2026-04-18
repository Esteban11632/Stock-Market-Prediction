import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker = "AAPL"

df = yf.download(ticker, start="2000-01-01")
print(df.head())

df.Close.plot(figsize=(10, 5))
plt.title("Close")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()
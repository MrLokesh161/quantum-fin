import yfinance as yf
import pandas as pd

START_DATE = "1990-01-01"
# yfinance treats `end` as exclusive; tomorrow includes today's completed bar.
END_DATE = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# -----------------------------
# Stock Market Indices
# -----------------------------

# S&P 500
sp500 = yf.download("^GSPC", start=START_DATE, end=END_DATE)
sp500.to_csv("datasets/sp500.csv")

# NASDAQ Composite
nasdaq = yf.download("^IXIC", start=START_DATE, end=END_DATE)
nasdaq.to_csv("datasets/nasdaq.csv")

# Dow Jones
dow = yf.download("^DJI", start=START_DATE, end=END_DATE)
dow.to_csv("datasets/dowjones.csv")

# NIFTY 50
nifty = yf.download("^NSEI", start=START_DATE, end=END_DATE)
nifty.to_csv("datasets/nifty50.csv")

# FTSE 100
ftse = yf.download("^FTSE", start=START_DATE, end=END_DATE)
ftse.to_csv("datasets/ftse100.csv")

# Nikkei 225
# nikkei = yf.download("^N225", start=START_DATE, end=END_DATE)
# nikkei.to_csv("datasets/nikkei225.csv")

# Hang Seng
# hangseng = yf.download("^HSI", start=START_DATE, end=END_DATE)
# hangseng.to_csv("datasets/hangseng.csv")


# -----------------------------
# Cryptocurrencies
# -----------------------------

# Bitcoin
# btc = yf.download("BTC-USD", start=START_DATE, end=END_DATE)
# btc.to_csv("datasets/bitcoin.csv")

# Ethereum
# eth = yf.download("ETH-USD", start=START_DATE, end=END_DATE)
# eth.to_csv("datasets/ethereum.csv")

# Binance Coin
# bnb = yf.download("BNB-USD", start=START_DATE, end=END_DATE)
# bnb.to_csv("datasets/bnb.csv")

# Solana
# sol = yf.download("SOL-USD", start=START_DATE, end=END_DATE)
# sol.to_csv("datasets/solana.csv")


# -----------------------------
# Volatility
# -----------------------------

# VIX
# vix = yf.download("^VIX", start=START_DATE, end=END_DATE)
# vix.to_csv("datasets/vix.csv")


# -----------------------------
# Commodities
# -----------------------------

# Gold
# gold = yf.download("GC=F", start=START_DATE, end=END_DATE)
# gold.to_csv("datasets/gold.csv")

# Silver
# silver = yf.download("SI=F", start=START_DATE, end=END_DATE)
# silver.to_csv("datasets/silver.csv")

# Crude Oil
# oil = yf.download("CL=F", start=START_DATE, end=END_DATE)
# oil.to_csv("datasets/crude_oil.csv")

# Natural Gas
# gas = yf.download("NG=F", start=START_DATE, end=END_DATE)
# gas.to_csv("datasets/natural_gas.csv")


# -----------------------------
# Currency Index
# -----------------------------

# US Dollar Index
# dxy = yf.download("DX-Y.NYB", start=START_DATE, end=END_DATE)
# dxy.to_csv("datasets/dollar_index.csv")


# -----------------------------
# Government Bonds
# -----------------------------

# 10-Year Treasury Yield
# treasury10 = yf.download("^TNX", start=START_DATE, end=END_DATE)
# treasury10.to_csv("datasets/treasury10.csv")

# 13-Week Treasury Bill
# treasury3m = yf.download("^IRX", start=START_DATE, end=END_DATE)
# treasury3m.to_csv("datasets/treasury3m.csv")


# -----------------------------
# Sector ETFs
# -----------------------------

# Technology
# xlk = yf.download("XLK", start=START_DATE, end=END_DATE)
# xlk.to_csv("datasets/technology.csv")

# Financial
# xlf = yf.download("XLF", start=START_DATE, end=END_DATE)
# xlf.to_csv("datasets/financial.csv")

# Healthcare
# xlv = yf.download("XLV", start=START_DATE, end=END_DATE)
# xlv.to_csv("datasets/healthcare.csv")

# Energy
# xle = yf.download("XLE", start=START_DATE, end=END_DATE)
# xle.to_csv("datasets/energy.csv")

# Consumer
# xly = yf.download("XLY", start=START_DATE, end=END_DATE)
# xly.to_csv("datasets/consumer.csv")

print("Download completed.")
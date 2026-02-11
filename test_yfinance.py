"""
Quick test script to verify yfinance is working
Run this to diagnose issues: python test_yfinance.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Testing yfinance installation...")
print(f"yfinance version: {yf.__version__}")
print()

# Test 1: Simple ticker
print("Test 1: Fetching AAPL data...")
try:
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d")
    if not hist.empty:
        print(f"✅ Success! Latest AAPL price: ${hist['Close'].iloc[-1]:.2f}")
        print(f"   Columns: {list(hist.columns)}")
    else:
        print("❌ Empty dataframe returned")
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 2: Download method
print("Test 2: Using yf.download()...")
try:
    df = yf.download("MSFT", period="1mo", progress=False)
    if not df.empty:
        print(f"✅ Success! Downloaded {len(df)} days of MSFT data")
        print(f"   Columns: {list(df.columns)}")
    else:
        print("❌ Empty dataframe returned")
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 3: Multiple tickers
print("Test 3: Multiple tickers...")
try:
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data = {}
    for t in tickers:
        hist = yf.Ticker(t).history(period="5d")
        if not hist.empty:
            data[t] = hist['Close']
            print(f"✅ {t}: ${hist['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ {t}: No data")
    
    if data:
        df = pd.DataFrame(data)
        print(f"\n✅ Combined dataframe shape: {df.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Test 4: Crypto
print("Test 4: Crypto ticker (BTC-USD)...")
try:
    btc = yf.Ticker("BTC-USD")
    hist = btc.history(period="5d")
    if not hist.empty:
        print(f"✅ Success! BTC price: ${hist['Close'].iloc[-1]:.2f}")
    else:
        print("❌ Empty dataframe returned")
except Exception as e:
    print(f"❌ Error: {e}")
print()

print("=" * 50)
print("If all tests passed, yfinance is working correctly!")
print("If tests failed, try: pip install --upgrade yfinance")

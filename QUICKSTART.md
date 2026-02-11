# 🚀 Quick Start Guide

Get started with Portfolio Analyzer in 3 simple steps!

## Step 1: Install Dependencies

Open your terminal and run:

```bash
pip install -r requirements.txt
```

## Step 2: Launch the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Step 3: Enter Your Portfolio

### On the "Portfolio Input" tab:

1. **Add your holdings** using the table:
   - Click on any cell to edit
   - Add new rows with the "+" button
   - Delete rows with the trash icon

2. **Example Entry**:
   ```
   Ticker: AAPL
   Shares: 10
   Cost Basis: 150
   ```

3. **Click "Analyze Portfolio"**

That's it! The app will:
- Fetch current prices
- Calculate portfolio weights automatically
- Show your total portfolio value
- Display P&L if you entered cost basis
- Analyze performance vs S&P 500

## 📊 What You'll See

### Portfolio Summary
- Total portfolio value
- Number of positions
- Concentration ratio
- Detailed position breakdown

### Analysis Tabs
- **Overview**: Returns, Sharpe ratio, alpha vs benchmark
- **Risk Metrics**: VaR, CVaR, correlation heatmap
- **Factor Analysis**: Monthly performance breakdown
- **Stress Test**: How your portfolio would perform in 2020 crash

## 💡 Pro Tips

1. **No Math Required**: Just enter how many shares you own - the app calculates weights for you
2. **Cost Basis Optional**: Add it if you want to track profits/losses
3. **Crypto Support**: Use tickers like `BTC-USD`, `ETH-USD`
4. **Dynamic Editing**: Add/remove positions anytime and re-analyze
5. **Save Your Data**: Bookmark the page - session data persists while the app is open

## 🎯 Sample Portfolio to Try

Copy this into your portfolio table:

| Ticker  | Shares | Cost Basis |
|---------|--------|------------|
| AAPL    | 10     | 150        |
| MSFT    | 15     | 280        |
| TSLA    | 5      | 200        |
| BTC-USD | 0.5    | 35000      |

Click "Analyze Portfolio" and explore the tabs!

## ❓ Need Help?

- Check `README.md` for detailed documentation
- Common issues? See the Troubleshooting section
- Make sure tickers are valid Yahoo Finance symbols

---

**Happy Analyzing! 📈**

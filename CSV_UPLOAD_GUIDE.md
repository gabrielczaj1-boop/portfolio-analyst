# 📂 CSV Upload Guide

## How to Use the Broker CSV Upload Feature

The Portfolio Analyzer can automatically import your holdings from broker CSV exports.

### ✅ What You Need

Your CSV file must contain at least these columns:
- **Symbol** (or "Ticker"): Stock/crypto ticker symbols
- **Quantity** (or "Shares"): Number of shares (preferred)
- **Value** (or "Market Value"): Current value of each position

The parser will work with either Quantity OR Value (or both!).

### 📋 Supported Formats

The parser is designed to handle messy broker exports and will:

1. **Find the data automatically** - Skips header/metadata rows
2. **Clean currency formatting** - Removes $, €, £, ¥ and commas
3. **Handle negative values** - Converts ($500.00) to -500.0
4. **Filter summary rows** - Removes "Total", "Balances", "Cash balance", etc.
5. **Calculate shares** - Derives share quantities from values and current prices

### 📝 Example CSV Structure

**Format 1: With Quantity (Best)**
```csv
Exported on: 01/20/2026

Symbol  ,Description ,Quantity ,Price ,Value 
AAPL ,Apple Inc,50,$225.50,"$11,275.00"
MSFT ,Microsoft Corp,30,$415.30,"$12,459.00"
GOOGL ,Alphabet Inc,15,$145.75,"$2,186.25"

Cash balance,,,,$0.00
Total,,,,,"$25,920.25"
```

**Format 2: Value Only (Also Works)**
```csv
Account Summary

Symbol,Market Value
AAPL,"$11,275.00"
MSFT,"$12,459.00"
GOOGL,"$2,186.25"

Total,"$25,920.25"
```

### 🚀 How to Import

1. **Export from your broker** - Download positions/holdings as CSV
2. **Click "Browse files"** in the Portfolio Input tab
3. **Upload your CSV**
4. The app will automatically:
   - Parse the file
   - Extract tickers and values
   - Calculate proper weights
   - Populate the data editor
5. **Click "Analyze Portfolio"**

### ⚠️ Common Issues

**"Could not find 'Symbol' column"**
- Make sure your CSV has a column named "Symbol", "Ticker", or similar
- Check that the file has headers

**"No valid holdings found"**
- Verify that your Symbol and Value columns have actual data
- Make sure values are not all zero or negative

**"Error parsing CSV"**
- Try opening the CSV in Excel/Numbers to verify it's formatted correctly
- Make sure there are no special characters in ticker symbols

### 💡 Supported Brokers

This feature is designed to work with exports from:
- **Merrill Lynch / Bank of America** ✅ Tested
- Fidelity
- Charles Schwab
- TD Ameritrade
- E*TRADE
- Robinhood
- Interactive Brokers
- Vanguard
- Most other brokers that export Symbol + Quantity/Value

### 🔍 What Gets Filtered Out

The parser automatically removes:
- Cash balances
- Money market accounts (BLACKROCK LIQUIDITY, etc.)
- Sweep funds
- Total/subtotal rows
- Pending activity
- Margin balances
- Settlement funds
- Account summary rows
- Empty rows

### 📊 After Import

Once imported, you can:
- Edit any ticker symbols
- Update share quantities
- Add cost basis for P&L tracking
- Add or remove positions manually

### 🎯 Example Files

Check `sample_broker_export.csv` in the project folder for a working example.

---

**Need Help?** If your broker's export format isn't working, please share a sample (with sensitive data removed) and we can add support for it!

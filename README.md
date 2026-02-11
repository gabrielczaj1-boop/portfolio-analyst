# 📊 Portfolio Analyzer

A professional, plug-and-play portfolio analysis tool built with Python, Streamlit, and Plotly. Features a sleek dark mode UI and comprehensive risk analytics.

## ✨ Features

### 💼 Portfolio Input Tab
- **Simple Holdings Entry**: Just enter ticker + shares (no math required!)
- **Automatic Weight Calculation**: App calculates portfolio weights based on current prices
- **Portfolio Value**: Instant calculation of total portfolio value
- **P&L Tracking**: Enter cost basis to see unrealized gains/losses
- **Position Summary**: Visual breakdown of portfolio allocation
- **Concentration Analysis**: Identifies if your portfolio is too concentrated

### 📈 Overview Tab
- **Key Performance Indicators**: Annualized Return, Alpha, Sharpe Ratio, Max Drawdown
- **Visual Analytics**: Interactive cumulative returns chart vs benchmark
- **Additional Metrics**: Volatility, Beta, Total Return, Win Rate

### ⚠️ Risk Metrics Tab
- **Advanced Risk Metrics**: Sortino Ratio, Value at Risk (VaR), Conditional VaR (CVaR)
- **Correlation Analysis**: Heatmap showing asset correlation matrix
- **Risk Contribution**: Pie chart showing which assets drive portfolio risk
- **Returns Distribution**: Histogram with VaR visualization

### 🔬 Factor Analysis Tab
- **Monthly Returns Heatmap**: Year-over-month performance visualization
- **Monthly Statistics**: Best/worst months, average returns, positive month percentage
- **Asset Breakdown**: Individual asset performance comparison

### 💥 Stress Test Tab
- **COVID-19 Crash Analysis**: Historical stress test (Feb-Mar 2020)
- **Comparative Analysis**: Portfolio vs benchmark performance during crisis
- **Risk Metrics**: Volatility, VaR, and downside statistics during stress period
- **Actionable Insights**: Automated recommendations based on stress test results

## 🚀 Quick Start

### Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the application**:
```bash
python -m streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📖 Usage

### Portfolio Input (Tab 1)

**The app makes it super simple!** Just enter your actual holdings - no need to calculate weights or percentages.

#### Option 1: Upload Broker CSV
1. **Click "Browse files"** and upload your broker's CSV export
2. The app automatically extracts tickers and calculates shares
3. Supported: Most broker formats with Symbol and Value columns

#### Option 2: Manual Entry
1. **Enter Your Holdings**:
   - **Ticker**: Stock/crypto symbol (e.g., AAPL, MSFT, BTC-USD)
   - **Shares**: How many shares you own (e.g., 10, 15.5, 0.25)
   - **Cost Basis** (Optional): Your average purchase price (for P&L tracking)

2. **Click "Analyze Portfolio"**:
   - The app automatically:
     - Fetches current market prices
     - Calculates your portfolio weights
     - Computes total portfolio value
     - Shows profit/loss if you entered cost basis
     - Compares performance vs S&P 500 benchmark

### Example Input

| Ticker | Shares | Cost Basis (Optional) |
|--------|--------|-----------------------|
| AAPL   | 10     | $150.00              |
| MSFT   | 15     | $280.00              |
| TSLA   | 5      | $200.00              |
| BTC-USD| 0.5    | $35,000.00           |

The app will automatically calculate that this portfolio is worth ~$X and show you that AAPL is ~Y% of your portfolio.

### Navigation

Use the tabs at the top to switch between different analysis views:
- **Portfolio Input**: Enter your holdings (start here!)
- **Overview**: Performance summary and cumulative returns
- **Risk Metrics**: Comprehensive risk analysis
- **Factor Analysis**: Monthly performance breakdown
- **Stress Test**: Historical crisis simulation

## 🎨 Design Features

- **Pure White Backdrop**: Clean Zinnia.com-inspired design with maximum whitespace
- **Minimal Aesthetic**: Subtle shadows (1-3px), clean borders, no gradients
- **Professional Typography**: Clear hierarchy with perfect spacing
- **Wide Layout**: Optimized for large screens
- **Card-Based Design**: Metrics in clean white cards with minimal shadows
- **Interactive Charts**: Hover, zoom, and pan on all visualizations (Plotly White theme)
- **Clean Buttons**: Solid purple buttons with subtle shadows
- **Real-time Feedback**: Loading spinners and toast notifications
- **Tooltips**: Every metric includes helpful explanations

## 🛠️ Technical Stack

- **Frontend**: Streamlit 1.31.0
- **Data**: yfinance 0.2.36 (Yahoo Finance API)
- **Visualization**: Plotly 5.18.0
- **Analytics**: NumPy, Pandas, SciPy
- **Caching**: Streamlit's built-in caching (1-hour TTL)

## 📊 Metrics Explained

### Performance Metrics

- **Annualized Return**: Average yearly return accounting for compounding
- **Alpha**: Excess return vs benchmark after adjusting for risk (beta)
- **Sharpe Ratio**: Risk-adjusted return (>1 good, >2 excellent)
- **Max Drawdown**: Largest peak-to-trough decline

### Risk Metrics

- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
- **Value at Risk (VaR)**: Maximum expected loss at 95% confidence
- **Conditional VaR (CVaR)**: Average loss when VaR is exceeded
- **Beta**: Sensitivity to market movements (1.0 = moves with market)

## 🔧 Architecture

The application follows a modular design:

```
Portfolio Analyst/
├── app.py              # Main Streamlit UI
├── data_engine.py      # Data fetching and calculations
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

### Modular Components

**Data Engine** (`data_engine.py`):
- Handles all data fetching with caching
- Implements financial calculations
- Pure Python functions (no UI dependencies)
- Fully testable and reusable

**UI Layer** (`app.py`):
- Streamlit interface
- Visualization components
- User input handling
- Tab-based navigation

## ⚠️ Important Notes

1. **Data Source**: Uses Yahoo Finance (yfinance). Some tickers may not be available.
2. **Historical Data**: Fetches 3 years of data by default
3. **Caching**: Data is cached for 1 hour to improve performance
4. **Risk-Free Rate**: Default 2% for Sharpe/Sortino calculations
5. **Trading Days**: Uses 252 trading days per year for annualization

## 🐛 Troubleshooting

### Common Issues

**"Failed to fetch data for..."**
- Check that tickers are valid Yahoo Finance symbols
- Ensure you have an internet connection
- Some cryptocurrencies require `-USD` suffix (e.g., `BTC-USD`)
- Try searching the ticker on Yahoo Finance website first to verify

**"Please add at least one holding"**
- Make sure you've entered at least one row in the holdings table
- Both Ticker and Shares are required fields

**"Total portfolio value is zero"**
- Check that share quantities are greater than zero
- Verify tickers are valid and have current price data

## 📝 Example Portfolios

### Conservative (Low Risk)
```
Ticker | Shares | Cost Basis
SPY    | 20     | $380.00
AGG    | 30     | $105.00
GLD    | 15     | $165.00
VNQ    | 10     | $85.00
```

### Aggressive Growth (Tech)
```
Ticker | Shares | Cost Basis
AAPL   | 25     | $145.00
MSFT   | 15     | $300.00
NVDA   | 10     | $400.00
TSLA   | 8      | $220.00
```

### Crypto-Heavy
```
Ticker  | Shares | Cost Basis
BTC-USD | 0.5    | $35,000.00
ETH-USD | 5.0    | $2,000.00
SPY     | 10     | $400.00
GLD     | 5      | $175.00
```

### Balanced Mix
```
Ticker | Shares | Cost Basis
SPY    | 15     | $400.00
QQQ    | 10     | $350.00
IWM    | 20     | $180.00
EFA    | 15     | $70.00
```

## 🔐 Disclaimer

This tool is for **educational and informational purposes only**. It is not financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

## 📄 License

This project is open-source and available for educational use.

---

**Built with ❤️ by a Quant Developer**

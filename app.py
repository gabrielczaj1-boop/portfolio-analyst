"""
Portfolio Analyzer - Professional Quant Dashboard
A sleek, plug-and-play tool for portfolio analysis and risk management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import data_engine as de
import styles

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global styles (dark AI-company aesthetic)
styles.apply_global_styles()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'portfolio_configured' not in st.session_state:
    st.session_state.portfolio_configured = False
if 'holdings_df' not in st.session_state:
    st.session_state.holdings_df = pd.DataFrame({
        'Ticker': ['', '', ''],
        'Shares': [0.0, 0.0, 0.0],
        'Cost Basis (Optional)': [None, None, None],
        'Buy Date': [pd.NaT, pd.NaT, pd.NaT],
    })
if 'benchmark' not in st.session_state:
    st.session_state.benchmark = 'SPY'

# ============================================================================
# MAIN APP - HEADER
# ============================================================================

st.markdown("""
    <div style='text-align: center; padding: 60px 0 40px 0;'>
        <h1 style='font-size: 56px; font-weight: 700; color: #f1f5f9 !important; 
                   letter-spacing: -1.5px; margin-bottom: 16px; line-height: 1.1;
                   font-family: Inter, system-ui, sans-serif;'>
            Portfolio Analyzer
        </h1>
        <p style='font-size: 20px; color: #94a3b8 !important; font-weight: 400; line-height: 1.6;
                  font-family: Inter, system-ui, sans-serif;'>
            Professional portfolio analytics and risk management
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# TAB NAVIGATION
# ============================================================================

tab0, tab1, tab2, tab3, tab_news, tab4 = st.tabs([
    "💼 Portfolio Input",
    "📈 Overview",
    "⚠️ Risk Metrics", 
    "🔬 Factor Analysis",
    "📰 News",
    "🧠 Financial Advisor",
])

# ============================================================================
# TAB 0: PORTFOLIO INPUT
# ============================================================================

with tab0:
    st.markdown("""
        <div style='padding: 20px 0 20px 0;'>
            <h2 style='color: #f1f5f9 !important; margin-bottom: 12px; font-weight: 600;'>Enter your portfolio holdings</h2>
            <p style='color: #94a3b8 !important; font-size: 17px; line-height: 1.7; margin-bottom: 24px;'>
                Simply enter your holdings below. <strong style='color: #e2e8f0 !important;'>You don't need to calculate weights</strong> — 
                the app will do it automatically based on current market prices.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background: rgba(30,41,59,0.65); padding: 20px; border-radius: 10px;
                    border: 1px solid rgba(148,163,184,0.12); margin-bottom: 25px;
                    backdrop-filter: blur(12px);'>
            <h3 style='color: #f1f5f9 !important; margin: 0 0 10px 0; font-size: 16px; font-weight: 600;'>📂 Import from Broker CSV</h3>
            <p style='color: #94a3b8 !important; font-size: 14px; margin: 0;'>
                Upload your broker export (must contain 'Symbol' and 'Value' columns)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Track if we've already processed the current file
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV export from your broker. The file should contain Symbol and Value columns.",
        label_visibility="collapsed",
        key="csv_uploader"
    )
    
    # Only process if there's a new file (check by name to avoid reprocessing)
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Only process if this is a new file
        if st.session_state.last_uploaded_file != file_id:
            try:
                with st.spinner("Parsing CSV file..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    parsed_df = de.clean_broker_csv(uploaded_file)
                    
                    if parsed_df is None or parsed_df.empty:
                        st.error("No data was parsed from the CSV file.")
                    else:
                        # Mark this file as processed
                        st.session_state.last_uploaded_file = file_id
                        
                        # Update holdings
                        st.session_state.holdings_df = parsed_df
                        
                        tickers_list = parsed_df['Ticker'].tolist()
                        st.success(f"Successfully imported {len(parsed_df)} holdings: {', '.join(tickers_list)}")
                    
            except Exception as e:
                import traceback
                st.error(f"Error parsing CSV: {str(e)}")
                
                with st.expander("Debug Information"):
                    st.code(traceback.format_exc())
                
                with st.expander("Troubleshooting Tips"):
                    st.markdown("""
                    **Your CSV should have columns like:**
                    - `Symbol` or `Ticker` - Stock symbols
                    - `Quantity` or `Shares` - Number of shares
                    - `Value` or `Market Value` - Position value (optional)
                    
                    **Common issues:**
                    - File might use a different column name
                    - File might have unusual encoding
                    - Data might start many rows down
                    
                    **Try:**
                    1. Open your CSV in Excel and check column names
                    2. Make sure there's a "Symbol" column
                    3. Save as CSV UTF-8 format
                    """)
        else:
            # File already processed - show confirmation
            if 'holdings_df' in st.session_state and not st.session_state.holdings_df.empty:
                tickers_list = st.session_state.holdings_df['Ticker'].tolist()
                st.success(f"CSV loaded: {len(tickers_list)} holdings ({', '.join(tickers_list)})")
    
    st.markdown("<div style='margin: 25px 0 15px 0;'><hr style='border: none; border-top: 1px solid rgba(148,163,184,0.12);'></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(30,41,59,0.65); padding: 24px 28px; border-radius: 12px;
                    border: 1px solid rgba(148,163,184,0.12); margin-bottom: 24px;
                    backdrop-filter: blur(12px);'>
            <p style='color: #f1f5f9 !important; font-size: 16px; font-weight: 600; margin: 0 0 14px 0;'>
                Enter your holdings below
            </p>
            <p style='color: #94a3b8 !important; font-size: 13px; line-height: 1.8; margin: 0;'>
                Click on any cell to edit. Use the <strong style='color: #e2e8f0 !important;'>+ button</strong> at the bottom to add more rows.<br>
                <strong style='color: #818cf8 !important;'>Ticker</strong> — Stock or crypto symbol (e.g. AAPL, MSFT, BTC-USD) &nbsp;&nbsp;
                <strong style='color: #818cf8 !important;'>Shares</strong> — Number of shares you own &nbsp;&nbsp;
                <strong style='color: #818cf8 !important;'>Cost Basis</strong> — Optional: avg purchase price &nbsp;&nbsp;
                <strong style='color: #818cf8 !important;'>Buy Date</strong> — Optional: date purchased
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Data editor for portfolio input (simple holdings mode)
    edited_df = st.data_editor(
        st.session_state.holdings_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="Stock or crypto symbol (e.g., AAPL, MSFT, BTC-USD)",
                required=True,
                default="",
            ),
            "Shares": st.column_config.NumberColumn(
                "Shares",
                help="Number of shares you own",
                min_value=0.0,
                required=True,
                format="%.4f",
                default=0.0,
            ),
            "Cost Basis (Optional)": st.column_config.NumberColumn(
                "Cost Basis",
                help="Your average purchase price per share (optional, for P&L)",
                min_value=0.0,
                format="$%.2f",
            ),
            "Buy Date": st.column_config.DatetimeColumn(
                 "Buy Date",
                 help="Optional: date you first bought this position (used for since-buy-date performance)",
            ),
        },
        hide_index=True,
    )
    
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
    
    # Analyze button (centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze Portfolio", type="primary", use_container_width=True)
    
    # Process portfolio when button is clicked
    if analyze_button:
        # Validate input
        if edited_df.empty or len(edited_df) == 0:
            st.error("❌ Please add at least one holding to your portfolio")
            st.stop()
        
        # Clean up the dataframe
        clean_df = edited_df.copy()
        clean_df['Ticker'] = clean_df['Ticker'].str.strip().str.upper()
        clean_df = clean_df[clean_df['Ticker'].notna() & (clean_df['Ticker'] != '')]
        clean_df = clean_df[clean_df['Shares'] > 0]
        
        if clean_df.empty:
            st.error("❌ Please enter valid tickers and share quantities")
            st.stop()
        
        # Fetch current prices
        tickers = clean_df['Ticker'].tolist()
        
        with st.spinner(f"🔄 Fetching current prices for {len(tickers)} ticker(s)..."):
            try:
                current_prices = de.fetch_current_prices(tickers)
                st.toast("✅ Prices fetched successfully!", icon="✅")
            except Exception as e:
                st.error(f"❌ Error fetching prices: {str(e)}")
                st.info("💡 Try checking if your tickers are correct at https://finance.yahoo.com")
                with st.expander("📋 Your tickers"):
                    st.write(tickers)
                st.stop()
        
        # If CSV was uploaded with values, calculate actual shares
        if '_original_value' in clean_df.columns:
            clean_df = de.calculate_shares_from_values(clean_df, current_prices)
        
        # Calculate weights and portfolio value
        try:
            weights, position_values, total_value = de.calculate_portfolio_weights(clean_df, current_prices)
        except Exception as e:
            st.error(f"❌ Error calculating portfolio: {str(e)}")
            st.stop()
        
        # Store in session state (use default benchmark S&P 500)
        st.session_state.holdings_df = clean_df
        st.session_state.tickers = tickers
        st.session_state.weights = weights
        st.session_state.current_prices = current_prices
        st.session_state.position_values = position_values
        st.session_state.total_value = total_value
        st.session_state.benchmark = 'SPY'  # Default to S&P 500
        st.session_state.portfolio_configured = True
        
        st.success("✅ Portfolio configured successfully! Switch to other tabs to view analysis.")
        st.balloons()
    
    # Display current portfolio summary if configured
    if st.session_state.portfolio_configured:
        st.markdown("<div style='margin: 50px 0 30px 0; border-top: 1px solid rgba(148,163,184,0.12); padding-top: 40px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom: 25px; color: #f1f5f9 !important;'>Current Portfolio Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Portfolio Value",
                value=f"${st.session_state.total_value:,.2f}",
                help="Current market value of all holdings"
            )
        
        with col2:
            st.metric(
                label="Number of Positions",
                value=len(st.session_state.tickers),
                help="Total number of different assets"
            )
        
        with col3:
            avg_weight = 1.0 / len(st.session_state.tickers)
            concentration = max(st.session_state.weights) / avg_weight
            st.metric(
                label="Concentration Ratio",
                value=f"{concentration:.2f}x",
                help="Largest position vs equal weight (1.0 = perfectly balanced)"
            )
        
        # Position details table
        st.markdown("<h4 style='margin-top: 30px; margin-bottom: 15px; color: #f1f5f9 !important; font-weight: 500;'>Position Details</h4>", unsafe_allow_html=True)
        position_data = []
        for i, ticker in enumerate(st.session_state.tickers):
            shares = st.session_state.holdings_df.iloc[i]['Shares']
            current_price = st.session_state.current_prices[ticker]
            position_value = st.session_state.position_values[ticker]
            weight = st.session_state.weights[i]
            
            position_data.append({
                'Ticker': ticker,
                'Shares': shares,
                'Current Price': f"${current_price:,.2f}",
                'Market Value': f"${position_value:,.2f}",
                'Weight': f"{weight*100:.2f}%"
            })
        
        positions_df = pd.DataFrame(position_data)
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
        
        # P&L Analysis if cost basis provided
        pnl_df = de.calculate_portfolio_pnl(st.session_state.holdings_df, st.session_state.current_prices)
        if pnl_df is not None:
            st.markdown("<h4 style='margin-top: 30px; margin-bottom: 15px; color: #f1f5f9 !important; font-weight: 500;'>Profit & Loss Analysis</h4>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_cost = pnl_df['Cost Value'].sum()
            total_current = pnl_df['Current Value'].sum()
            total_pnl = pnl_df['P&L'].sum()
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            
            with col1:
                st.metric("Total Cost Basis", f"${total_cost:,.2f}")
            with col2:
                st.metric("Current Value", f"${total_current:,.2f}")
            with col3:
                st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            with col4:
                st.metric("Total Return", f"{total_pnl_pct:.2f}%", delta=f"{total_pnl_pct:.2f}%")
            
            # Detailed P&L table
            st.dataframe(
                pnl_df.style.format({
                    'Shares': '{:.4f}',
                    'Cost Basis': '${:.2f}',
                    'Current Price': '${:.2f}',
                    'Cost Value': '${:,.2f}',
                    'Current Value': '${:,.2f}',
                    'P&L': '${:,.2f}',
                    'P&L %': '{:.2f}%'
                }).background_gradient(subset=['P&L %'], cmap='RdYlGn', vmin=-50, vmax=50),
                use_container_width=True,
                hide_index=True
            )

# Check if portfolio is configured for other tabs
if not st.session_state.portfolio_configured:
    with tab1:
        st.info("👈 Please configure your portfolio in the **Portfolio Input** tab first")
    with tab2:
        st.info("👈 Please configure your portfolio in the **Portfolio Input** tab first")
    with tab3:
        st.info("👈 Please configure your portfolio in the **Portfolio Input** tab first")
    with tab_news:
        st.info("👈 Please configure your portfolio in the **Portfolio Input** tab first")
    with tab4:
        st.info("👈 Please configure your portfolio in the **Portfolio Input** tab first")
    st.stop()

# Load portfolio data from session state
tickers = st.session_state.tickers
weights = st.session_state.weights
benchmark = st.session_state.benchmark

# ============================================================================
# DATA LOADING FOR ANALYSIS
# ============================================================================

# Fetch historical data with progress indicator
with st.spinner("🔄 Fetching historical market data..."):
    try:
        # Fetch portfolio data (max available history)
        prices_df = de.fetch_price_data(tickers, years=10)
        
        # Fetch benchmark data
        benchmark_prices = de.fetch_price_data([benchmark], years=10)
        
        st.toast("✅ Historical data loaded!", icon="✅")
        
    except Exception as e:
        st.error(f"❌ Error fetching historical data: {str(e)}")
        st.error("💡 **Tip:** Make sure you have a stable internet connection and the tickers are valid Yahoo Finance symbols.")
        st.info("Try refreshing the page or checking ticker symbols at https://finance.yahoo.com")
        
        # Show debug info
        with st.expander("🔍 Debug Information"):
            st.write(f"**Tickers:** {', '.join(tickers)}")
            st.write(f"**Benchmark:** {benchmark}")
            st.write(f"**Error Details:** {str(e)}")
        
        st.stop()

# Calculate returns
returns_df = de.calculate_returns(prices_df)
portfolio_returns = de.calculate_portfolio_returns(returns_df, weights)
benchmark_returns = de.calculate_returns(benchmark_prices).iloc[:, 0]

# Historical window used for return statistics
if len(portfolio_returns) > 0:
    history_start_date = portfolio_returns.index[0].date()
    history_end_date = portfolio_returns.index[-1].date()
else:
    history_start_date = None
    history_end_date = None

# Portfolio return since user-entered buy dates (if available)
since_buy_return = None
since_buy_start_date = None
since_buy_end_date = history_end_date

if 'Buy Date' in st.session_state.holdings_df.columns and len(portfolio_returns) > 0:
    start_values = []
    end_values = []
    buy_dates = []

    latest_prices = prices_df.iloc[-1]

    for i, ticker in enumerate(tickers):
        row = st.session_state.holdings_df.iloc[i]
        buy_date_val = row.get('Buy Date', None)
        if pd.isna(buy_date_val):
            continue

        try:
            buy_ts = pd.to_datetime(buy_date_val).normalize()
        except Exception:
            continue

        if ticker not in prices_df.columns:
            continue

        series = prices_df[ticker].dropna()
        valid_idx = series.index[series.index >= buy_ts]
        if len(valid_idx) == 0:
            continue

        start_price = float(series.loc[valid_idx[0]])
        end_price = float(latest_prices[ticker])
        shares = float(row['Shares'])

        start_values.append(shares * start_price)
        end_values.append(shares * end_price)
        buy_dates.append(buy_ts.date())

    if start_values and sum(start_values) > 0:
        since_buy_return = (sum(end_values) / sum(start_values) - 1) * 100.0
        since_buy_start_date = min(buy_dates)

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    # ========================================================================
    # BROKERAGE-STYLE PORTFOLIO VIEW
    # ========================================================================
    
    # Check if CSV has day's change data
    has_csv_day_change = '_day_change' in st.session_state.holdings_df.columns
    has_csv_gain_loss = '_gain_loss_dollar' in st.session_state.holdings_df.columns
    
    # Calculate today's change - prefer CSV data
    if has_csv_day_change:
        csv_day_changes = st.session_state.holdings_df['_day_change'].dropna()
        if len(csv_day_changes) > 0:
            todays_change_dollar = csv_day_changes.sum()
            todays_change_pct = (todays_change_dollar / (st.session_state.total_value - todays_change_dollar) * 100) if st.session_state.total_value != todays_change_dollar else 0
        else:
            todays_change_dollar = 0
            todays_change_pct = 0
    elif len(portfolio_returns) > 0:
        todays_return = portfolio_returns.iloc[-1] if not pd.isna(portfolio_returns.iloc[-1]) else 0
        todays_change_pct = todays_return * 100
        todays_change_dollar = st.session_state.total_value * todays_return
    else:
        todays_change_pct = 0
        todays_change_dollar = 0
    
    # Calculate total gain/loss - prefer CSV data
    if has_csv_gain_loss:
        csv_gains = st.session_state.holdings_df['_gain_loss_dollar'].dropna()
        if len(csv_gains) > 0:
            total_gain = csv_gains.sum()
            
            # Use CSV's original values for consistency (same snapshot in time)
            if '_original_value' in st.session_state.holdings_df.columns:
                csv_values = st.session_state.holdings_df['_original_value'].dropna()
                csv_total_value = csv_values.sum() if len(csv_values) > 0 else st.session_state.total_value
            else:
                csv_total_value = st.session_state.total_value
            
            # Calculate return as gain / current value (more intuitive)
            total_gain_pct = (total_gain / csv_total_value * 100) if csv_total_value > 0 else 0
            total_cost = csv_total_value - total_gain
        else:
            total_cost = None
            total_gain = None
            total_gain_pct = None
    else:
        # Fallback to cost basis calculation
        pnl_df = de.calculate_portfolio_pnl(st.session_state.holdings_df, st.session_state.current_prices)
        if pnl_df is not None:
            total_cost = pnl_df['Cost Value'].sum()
            total_gain = pnl_df['P&L'].sum()
            total_current = pnl_df['Current Value'].sum()
            total_gain_pct = (total_gain / total_current * 100) if total_current > 0 else 0
        else:
            total_cost = None
            total_gain = None
            total_gain_pct = None
    
    # For P&L table display (used later)
    pnl_df = de.calculate_portfolio_pnl(st.session_state.holdings_df, st.session_state.current_prices)
    
    # Account Summary Header
    st.markdown("""
        <div style='background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); 
                    padding: 30px 35px; border-radius: 16px; margin-bottom: 30px;
                    box-shadow: 0 4px 24px rgba(99,102,241,0.25);'>
            <p style='color: rgba(255,255,255,0.75) !important; font-size: 13px; margin: 0 0 8px 0;
                      font-weight: 600; letter-spacing: .5px; text-transform: uppercase;'>
                TOTAL PORTFOLIO VALUE
            </p>
            <h1 style='color: #ffffff !important; font-size: 42px; margin: 0 0 15px 0; font-weight: 700; letter-spacing: -1px;'>
                ${:,.2f}
            </h1>
        </div>
    """.format(st.session_state.total_value), unsafe_allow_html=True)
    
    # Day's Change and Total Gain/Loss Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        change_class = "gain-green" if todays_change_dollar >= 0 else "loss-red"
        change_arrow = "▲" if todays_change_dollar >= 0 else "▼"
        st.markdown(f"""
            <div style='background: rgba(30,41,59,0.65); padding: 20px 25px; border-radius: 12px; 
                        border: 1px solid rgba(148,163,184,0.12); backdrop-filter: blur(12px);'>
                <p style='color: #94a3b8 !important; font-size: 13px; margin: 0 0 8px 0; font-weight: 500;'>Today's Change</p>
                <p class='{change_class}' style='font-size: 24px; margin: 0; font-weight: 600;'>
                    {change_arrow} ${abs(todays_change_dollar):,.2f}
                </p>
                <p class='{change_class}' style='font-size: 14px; margin: 4px 0 0 0;'>
                    {'+' if todays_change_pct >= 0 else ''}{todays_change_pct:.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if total_gain is not None:
            gain_class = "gain-green" if total_gain >= 0 else "loss-red"
            gain_arrow = "▲" if total_gain >= 0 else "▼"
            st.markdown(f"""
                <div style='background: rgba(30,41,59,0.65); padding: 20px 25px; border-radius: 12px; 
                            border: 1px solid rgba(148,163,184,0.12); backdrop-filter: blur(12px);'>
                    <p style='color: #94a3b8 !important; font-size: 13px; margin: 0 0 8px 0; font-weight: 500;'>Total Gain/Loss</p>
                    <p class='{gain_class}' style='font-size: 24px; margin: 0; font-weight: 600;'>
                        {gain_arrow} ${abs(total_gain):,.2f}
                    </p>
                    <p class='{gain_class}' style='font-size: 14px; margin: 4px 0 0 0;'>
                        {'+' if total_gain_pct >= 0 else ''}{total_gain_pct:.2f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: rgba(30,41,59,0.65); padding: 20px 25px; border-radius: 12px; 
                            border: 1px solid rgba(148,163,184,0.12); backdrop-filter: blur(12px);'>
                    <p style='color: #94a3b8 !important; font-size: 13px; margin: 0 0 8px 0; font-weight: 500;'>Total Gain/Loss</p>
                    <p style='color: #64748b !important; font-size: 18px; margin: 0; font-weight: 500;'>
                        Add cost basis to track
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style='background: rgba(30,41,59,0.65); padding: 20px 25px; border-radius: 12px; 
                        border: 1px solid rgba(148,163,184,0.12); backdrop-filter: blur(12px);'>
                <p style='color: #94a3b8 !important; font-size: 13px; margin: 0 0 8px 0; font-weight: 500;'>Positions</p>
                <p style='color: #f1f5f9 !important; font-size: 24px; margin: 0; font-weight: 600;'>
                    {len(st.session_state.tickers)}
                </p>
                <p style='color: #64748b !important; font-size: 14px; margin: 4px 0 0 0;'>
                    Holdings
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
    
    # ========================================================================
    # HOLDINGS TABLE (Brokerage Style)
    # ========================================================================
    
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px; font-weight: 600;'>Your Holdings</h3>", unsafe_allow_html=True)
    
    # Build holdings data - prioritize CSV data when available
    holdings_data = []
    for i, ticker in enumerate(st.session_state.tickers):
        row = st.session_state.holdings_df.iloc[i]
        shares = row['Shares']
        current_price = st.session_state.current_prices[ticker]
        position_value = st.session_state.position_values[ticker]
        weight = st.session_state.weights[i]
        
        # Get description from CSV if available
        description = row.get('_description', None) if '_description' in row else None
        
        # Get day's change - prefer CSV data, fallback to calculated
        if '_day_change' in row and row['_day_change'] is not None and not pd.isna(row['_day_change']):
            day_change_dollar = row['_day_change']
            # Estimate day % from dollar change
            day_change_pct = (day_change_dollar / (position_value - day_change_dollar) * 100) if position_value != day_change_dollar else 0
        else:
            # Fallback: calculate from returns
            if ticker in returns_df.columns and len(returns_df[ticker]) > 0:
                ticker_day_return = returns_df[ticker].iloc[-1] if not pd.isna(returns_df[ticker].iloc[-1]) else 0
            else:
                ticker_day_return = 0
            day_change_dollar = position_value * ticker_day_return
            day_change_pct = ticker_day_return * 100
        
        # Get P&L - prefer CSV data, fallback to calculated
        if '_gain_loss_dollar' in row and row['_gain_loss_dollar'] is not None and not pd.isna(row['_gain_loss_dollar']):
            total_pnl = row['_gain_loss_dollar']
            total_pnl_pct = row.get('_gain_loss_percent', None) if '_gain_loss_percent' in row else None
        elif pnl_df is not None and ticker in pnl_df['Ticker'].values:
            ticker_pnl = pnl_df[pnl_df['Ticker'] == ticker].iloc[0]
            total_pnl = ticker_pnl['P&L']
            total_pnl_pct = ticker_pnl['P&L %']
        else:
            total_pnl = None
            total_pnl_pct = None
        
        holdings_data.append({
            'Symbol': ticker,
            'Description': description if description else ticker,
            'Shares': shares,
            'Price': current_price,
            'Day Change': day_change_dollar,
            'Day %': day_change_pct,
            'Market Value': position_value,
            'Weight': weight * 100,
            'Total P&L': total_pnl,
            'P&L %': total_pnl_pct
        })
    
    holdings_df_display = pd.DataFrame(holdings_data)
    
    # Format the dataframe for display
    display_df = holdings_df_display.copy()
    display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:,.4f}" if x < 1 else f"{x:,.2f}")
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.2f}")
    display_df['Day Change'] = display_df['Day Change'].apply(lambda x: f"{'+'if x >= 0 else ''}${x:,.2f}" if x is not None and not pd.isna(x) else "—")
    display_df['Day %'] = display_df['Day %'].apply(lambda x: f"{'+'if x >= 0 else ''}{x:.2f}%" if x is not None and not pd.isna(x) else "—")
    display_df['Market Value'] = display_df['Market Value'].apply(lambda x: f"${x:,.2f}")
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.1f}%")
    display_df['Total P&L'] = display_df['Total P&L'].apply(lambda x: f"{'+'if x >= 0 else ''}${x:,.2f}" if x is not None and not pd.isna(x) else "—")
    display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{'+'if x >= 0 else ''}{x:.2f}%" if x is not None and not pd.isna(x) else "—")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Description": st.column_config.TextColumn("Name", width="medium"),
            "Shares": st.column_config.TextColumn("Shares", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Day Change": st.column_config.TextColumn("Day $", width="small"),
            "Day %": st.column_config.TextColumn("Day %", width="small"),
            "Market Value": st.column_config.TextColumn("Value", width="medium"),
            "Weight": st.column_config.TextColumn("Weight", width="small"),
            "Total P&L": st.column_config.TextColumn("Gain/Loss", width="small"),
            "P&L %": st.column_config.TextColumn("Return", width="small"),
        }
    )
    
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    
    # ========================================================================
    # PORTFOLIO ALLOCATION CHART
    # ========================================================================
    
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px; font-weight: 600;'>Portfolio Allocation</h3>", unsafe_allow_html=True)
    
    # Donut chart for allocation
    fig_allocation = go.Figure(data=[go.Pie(
        labels=st.session_state.tickers,
        values=[st.session_state.position_values[t] for t in st.session_state.tickers],
        hole=0.5,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=['#818cf8', '#a855f7', '#c084fc', '#f472b6', '#38bdf8', 
                   '#22d3ee', '#34d399', '#2dd4bf', '#fbbf24', '#fb923c'][:len(st.session_state.tickers)]
        ),
        textfont=dict(size=12, color='#e2e8f0')
    )])
    
    fig_allocation.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color='#e2e8f0', size=11)
        ),
        margin=dict(t=20, b=60, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        annotations=[dict(
            text=f'${st.session_state.total_value:,.0f}',
            x=0.5, y=0.5,
            font=dict(size=20, color='#e2e8f0', family='Arial'),
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig_allocation, use_container_width=True)
    
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    
    # ========================================================================
    # YOUR PORTFOLIO METRICS
    # ========================================================================
    
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px; font-weight: 600;'>Your Portfolio Metrics</h3>", unsafe_allow_html=True)
    
    # Calculate key metrics based on user's portfolio
    ann_return = de.calculate_annualized_return(portfolio_returns)
    ann_vol = de.calculate_annualized_volatility(portfolio_returns)
    sharpe = de.calculate_sharpe_ratio(portfolio_returns)
    max_dd = de.calculate_max_drawdown(portfolio_returns)
    alpha, beta = de.calculate_alpha_beta(portfolio_returns, benchmark_returns)

    # Detailed components for hover formulas
    rf_annual = 0.02
    rf_daily = rf_annual / 252
    excess_returns = portfolio_returns - rf_daily
    sharpe_numerator = excess_returns.mean() * 252 if len(excess_returns) > 0 else 0.0
    sharpe_denominator = portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0.0

    aligned_ab = pd.DataFrame(
        {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
    ).dropna()
    if not aligned_ab.empty:
        cov_pb = aligned_ab.cov().iloc[0, 1]
        var_b = aligned_ab["benchmark"].var()
        ann_port_ret_alpha = de.calculate_annualized_return(aligned_ab["portfolio"])
        ann_bench_ret_alpha = de.calculate_annualized_return(aligned_ab["benchmark"])
    else:
        cov_pb = var_b = ann_port_ret_alpha = ann_bench_ret_alpha = 0.0
    
    # -- helper shortcut --
    from styles import formula_card as _fc

    total_ret_raw = (1 + portfolio_returns).prod() - 1
    daily_std = portfolio_returns.std()
    n_days = len(portfolio_returns)

    # Inline-style snippets used inside every formula overlay
    from styles import math_frac, math_formula, m, mop

    S_ROW = "display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.06);"
    S_VAR = "color:#c4b5fd !important;font-weight:500;"
    S_NUM = "color:#fff !important;font-weight:600;font-family:'SF Mono','Fira Code',Consolas,monospace;"
    S_DIV = "border:none;border-top:1px solid rgba(255,255,255,.1);margin:8px 0;"
    S_RES = "display:flex;justify-content:space-between;padding:4px 0 0 0;"
    S_RES_VAR = "color:#a5b4fc !important;font-weight:600;"
    S_RES_NUM = "color:#34d399 !important;font-weight:700;font-size:14px;font-family:'SF Mono','Fira Code',Consolas,monospace;"
    S_NOTE = "font-size:11px;color:#94a3b8 !important;margin:8px 0 0 0;"

    def _row(label, val):
        return f"<div style='{S_ROW}'><span style='{S_VAR}'>{label}</span><span style='{S_NUM}'>{val}</span></div>"

    def _result(label, val):
        return f"<div style='{S_RES}'><span style='{S_RES_VAR}'>{label}</span><span style='{S_RES_NUM}'>{val}</span></div>"

    # Row 1: Key portfolio stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if total_gain is not None and total_gain_pct is not None:
            st.metric(
                label="Your Total Return (unrealized)",
                value=f"{total_gain_pct:+.2f}%",
                delta=f"${total_gain:+,.2f}",
                help="Unrealized gain/loss based on broker CSV snapshot"
            )
        else:
            st.markdown(_fc(
                "Annualized Return", f"{ann_return*100:.2f}%",
                math_formula(m("R"), mop("="), m("(1 + R"), m("<sub>total</sub>"), m(")"), mop(""), math_frac("252", "N"), mop("&minus; 1"))
                + _row("R<sub>total</sub>", f"{total_ret_raw:+.4f}")
                + _row("N (trading days)", f"{n_days}")
                + f"<hr style='{S_DIV}'>"
                + _result("= Annualized Return", f"{ann_return*100:+.2f}%")
                + f"<p style='{S_NOTE}'>{history_start_date} &rarr; {history_end_date}</p>"
            ), unsafe_allow_html=True)

    with col2:
        st.metric(
            label="Today's P&L",
            value=f"${todays_change_dollar:+,.2f}",
            delta=f"{todays_change_pct:+.2f}%",
            help="Today's change in your portfolio value"
        )

    with col3:
        st.markdown(_fc(
            "Portfolio Beta", f"{beta:.2f}",
            math_formula(m("&beta;"), mop("="), math_frac("Cov(R<sub>p</sub>, R<sub>b</sub>)", "Var(R<sub>b</sub>)"))
            + _row("Cov(R<sub>p</sub>, R<sub>b</sub>)", f"{cov_pb:.6f}")
            + _row("Var(R<sub>b</sub>)", f"{var_b:.6f}")
            + f"<hr style='{S_DIV}'>"
            + _result("= Beta", f"{beta:.4f}")
            + f"<p style='{S_NOTE}'>{history_start_date} &rarr; {history_end_date}</p>"
        ), unsafe_allow_html=True)

    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd*100:.2f}%",
            help="Largest peak-to-trough decline historically"
        )

    # Row 2: Risk metrics
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(_fc(
            "Volatility", f"{ann_vol*100:.2f}%",
            math_formula(m("&sigma;<sub>ann</sub>"), mop("="), m("&sigma;<sub>daily</sub>"), mop("&times;"), m("&radic;252"))
            + _row("&sigma;<sub>daily</sub>", f"{daily_std:.6f}")
            + _row("&radic;252", f"{np.sqrt(252):.4f}")
            + f"<hr style='{S_DIV}'>"
            + _result("= Annualized Vol", f"{ann_vol*100:.2f}%")
            + f"<p style='{S_NOTE}'>{history_start_date} &rarr; {history_end_date}</p>"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(_fc(
            "Sharpe Ratio", f"{sharpe:.2f}",
            math_formula(m("S"), mop("="), math_frac("E[R<sub>p</sub>] &minus; R<sub>f</sub>", "&sigma;<sub>p</sub>"))
            + _row("E[R<sub>p</sub>] (annualized)", f"{sharpe_numerator:.4f}")
            + _row("R<sub>f</sub> (annual)", f"{rf_annual:.2%}")
            + _row("&sigma;<sub>p</sub> (annualized)", f"{sharpe_denominator:.4f}")
            + f"<hr style='{S_DIV}'>"
            + _result("= Sharpe Ratio", f"{sharpe:.4f}")
            + f"<p style='{S_NOTE}'>{history_start_date} &rarr; {history_end_date}</p>"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(_fc(
            "Alpha", f"{alpha*100:+.2f}%",
            math_formula(m("&alpha;"), mop("="), m("R<sub>p</sub>"), mop("&minus;"), m("(R<sub>f</sub>"), mop("+"), m("&beta;(R<sub>b</sub>"), mop("&minus;"), m("R<sub>f</sub>))"))
            + _row("R<sub>p</sub> (annualized)", f"{ann_port_ret_alpha:.4f}")
            + _row("R<sub>b</sub> (annualized)", f"{ann_bench_ret_alpha:.4f}")
            + _row("&beta;", f"{beta:.4f}")
            + f"<hr style='{S_DIV}'>"
            + _result("= Alpha", f"{alpha*100:+.2f}%")
            + f"<p style='{S_NOTE}'>{history_start_date} &rarr; {history_end_date}</p>"
        ), unsafe_allow_html=True)

    with col4:
        weights_squared = sum([w**2 for w in st.session_state.weights])
        effective_positions = 1 / weights_squared if weights_squared > 0 else len(st.session_state.weights)
        wt_lines = "".join(
            _row(t, f"{w*100:.1f}%")
            for t, w in zip(st.session_state.tickers, st.session_state.weights)
        )
        st.markdown(_fc(
            "Diversification", f"{effective_positions:.1f}",
            math_formula(m("N<sub>eff</sub>"), mop("="), math_frac("1", "&Sigma; w<sub>i</sub>&sup2;"))
            + wt_lines
            + _row("&Sigma; w<sub>i</sub>&sup2;", f"{weights_squared:.4f}")
            + f"<hr style='{S_DIV}'>"
            + _result("= Effective Positions", f"{effective_positions:.2f} / {len(st.session_state.tickers)}")
        ), unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    
    # ========================================================================
    # TOP & BOTTOM PERFORMERS (from CSV data)
    # ========================================================================
    
    if has_csv_gain_loss and len(holdings_data) > 1:
        st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px; font-weight: 600;'>Position Performance</h3>", unsafe_allow_html=True)
        
        # Sort holdings by P&L %
        sorted_by_return = sorted(
            [h for h in holdings_data if h.get('P&L %') is not None],
            key=lambda x: x['P&L %'] if x['P&L %'] is not None else 0,
            reverse=True
        )
        
        if len(sorted_by_return) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div style='background: rgba(52,211,153,0.06); 
                                padding: 20px; border-radius: 12px; border: 1px solid rgba(52,211,153,0.15);'>
                        <p style='color: #34d399 !important; font-size: 13px; margin: 0 0 15px 0; font-weight: 600;
                                  letter-spacing: .5px; text-transform: uppercase;'>
                            TOP PERFORMERS
                        </p>
                """, unsafe_allow_html=True)
                
                for h in sorted_by_return[:3]:
                    pnl_pct = h.get('P&L %', 0) or 0
                    pnl_dollar = h.get('Total P&L', 0) or 0
                    st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                            <span style='color: #e2e8f0 !important; font-weight: 500;'>{h['Symbol']}</span>
                            <span style='color: #34d399 !important; font-weight: 600;'>+{pnl_pct:.1f}% (${pnl_dollar:+,.0f})</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div style='background: rgba(248,113,113,0.06); 
                                padding: 20px; border-radius: 12px; border: 1px solid rgba(248,113,113,0.15);'>
                        <p style='color: #f87171 !important; font-size: 13px; margin: 0 0 15px 0; font-weight: 600;
                                  letter-spacing: .5px; text-transform: uppercase;'>
                            UNDERPERFORMERS
                        </p>
                """, unsafe_allow_html=True)
                
                for h in sorted_by_return[-3:]:
                    pnl_pct = h.get('P&L %', 0) or 0
                    pnl_dollar = h.get('Total P&L', 0) or 0
                    color = "#f87171" if pnl_pct < 0 else "#34d399"
                    st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                            <span style='color: #e2e8f0 !important; font-weight: 500;'>{h['Symbol']}</span>
                            <span style='color: {color} !important; font-weight: 600;'>{pnl_pct:+.1f}% (${pnl_dollar:+,.0f})</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    
    # ========================================================================
    # PORTFOLIO GROWTH CHART
    # ========================================================================
    
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px; font-weight: 600;'>Portfolio Growth</h3>", unsafe_allow_html=True)
    
    # Button-style time period selector
    time_period = st.radio(
        "Time Period",
        options=["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "ALL"],
        index=4,  # Default to 1Y
        horizontal=True,
        label_visibility="collapsed",
        key="portfolio_time_period"
    )
    
    # Calculate date range based on selection
    end_date = portfolio_returns.index[-1]
    if time_period == "1M":
        start_date = end_date - pd.Timedelta(days=30)
    elif time_period == "3M":
        start_date = end_date - pd.Timedelta(days=90)
    elif time_period == "6M":
        start_date = end_date - pd.Timedelta(days=180)
    elif time_period == "YTD":
        start_date = pd.Timestamp(f"{end_date.year}-01-01")
    elif time_period == "1Y":
        start_date = end_date - pd.Timedelta(days=365)
    elif time_period == "3Y":
        start_date = end_date - pd.Timedelta(days=365*3)
    elif time_period == "5Y":
        start_date = end_date - pd.Timedelta(days=365*5)
    else:  # ALL
        start_date = portfolio_returns.index[0]
    
    # Filter returns to selected period and recalculate cumulative returns from that start
    filtered_portfolio_returns = portfolio_returns[portfolio_returns.index >= start_date]
    filtered_benchmark_returns = benchmark_returns[benchmark_returns.index >= start_date]
    
    # Calculate cumulative returns starting from the selected period (starts at 0%)
    portfolio_cumulative = de.calculate_cumulative_returns(filtered_portfolio_returns)
    benchmark_cumulative = de.calculate_cumulative_returns(filtered_benchmark_returns)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_cumulative.index,
        y=(portfolio_cumulative - 1) * 100,
        mode='lines',
        name='Portfolio',
        fill='tozeroy',
        line=dict(color='#818cf8', width=3),
        fillcolor='rgba(129, 140, 248, 0.12)'
    ))
    
    fig.add_trace(go.Scatter(
        x=benchmark_cumulative.index,
        y=(benchmark_cumulative - 1) * 100,
        mode='lines',
        name=f'Benchmark ({benchmark})',
        line=dict(color='#c084fc', width=2, dash='dash')
    ))
    
    # Calculate period return for display
    period_return = (portfolio_cumulative.iloc[-1] - 1) * 100 if len(portfolio_cumulative) > 0 else 0
    
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        height=450,
        xaxis_title="Date",
        yaxis_title="Return (%)",
        title=dict(
            text=f"<b>{time_period} Return: {period_return:+.2f}% "
                 f"({start_date.date()} → {end_date.date()})</b>",
            font=dict(size=14, color='#e2e8f0'),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#cbd5e1', size=12)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', size=12),
        xaxis=dict(
            title_font=dict(color='#cbd5e1', size=13), 
            tickfont=dict(color='#cbd5e1', size=11),
            gridcolor='rgba(148,163,184,0.1)'
        ),
        yaxis=dict(
            title_font=dict(color='#cbd5e1', size=13), 
            tickfont=dict(color='#cbd5e1', size=11),
            gridcolor='rgba(148,163,184,0.1)',
            zeroline=True,
            zerolinecolor='rgba(148,163,184,0.15)',
            zerolinewidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px;'>Additional Performance Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Volatility (Ann.)",
            value=f"{ann_vol*100:.2f}%",
            help="Annualized standard deviation of returns"
        )
    
    with col2:
        st.metric(
            label="Beta",
            value=f"{beta:.2f}",
            help="Sensitivity to benchmark movements. 1.0 = moves with market"
        )
    
    with col3:
        # Use actual P&L from CSV if available, otherwise show historical
        if total_gain is not None and total_gain_pct is not None:
            st.metric(
                label="Your Actual Return (unrealized)",
                value=f"{total_gain_pct:+.2f}%",
                delta=f"${total_gain:+,.2f}",
                help="Unrealized gain/loss from broker CSV snapshot (based on current value, not a specific start date)"
            )
        else:
            historical_return = (portfolio_cumulative.iloc[-1] - 1) * 100
            st.metric(
                label="Historical Return",
                value=f"{historical_return:.2f}%",
                help=f"Cumulative return from {history_start_date} to {history_end_date}"
            )
    
    with col4:
        # Prefer showing return since buy dates when available
        if since_buy_return is not None and since_buy_start_date is not None:
            st.metric(
                label="Since Buy-Date Return",
                value=f"{since_buy_return:+.2f}%",
                help=f"Portfolio return from first entered buy date "
                     f"({since_buy_start_date} → {since_buy_end_date}) using Yahoo Finance prices"
            )
        else:
            win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) * 100
            st.metric(
                label="Win Rate",
                value=f"{win_rate:.1f}%",
                help="Percentage of trading days with positive returns"
            )

# ============================================================================
# TAB 2: RISK METRICS
# ============================================================================

with tab2:
    st.markdown("<h2 style='color: #f1f5f9; margin-bottom: 30px;'>Risk Analysis & Metrics</h2>", unsafe_allow_html=True)
    
    # Risk metrics
    sortino = de.calculate_sortino_ratio(portfolio_returns)
    var_95 = de.calculate_var(portfolio_returns, confidence=0.95)
    cvar_95 = de.calculate_cvar(portfolio_returns, confidence=0.95)

    # Components for hover formulas
    rf_annual_rm = 0.02
    rf_daily_rm = rf_annual_rm / 252
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std_annual = downside_returns.std() * np.sqrt(252) if downside_returns.std() > 0 else 0.0
    tail_losses = portfolio_returns[portfolio_returns <= -var_95]
    cvar_tail_mean = tail_losses.mean() if not tail_losses.empty else 0.0
    
    from styles import formula_card as _fc_risk

    # Reuse the same inline-style snippets
    from styles import math_frac as mf_r, math_formula as mform_r, m as m_r, mop as mop_r

    S_ROW = "display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.06);"
    S_VAR = "color:#c4b5fd !important;font-weight:500;"
    S_NUM = "color:#fff !important;font-weight:600;font-family:'SF Mono','Fira Code',Consolas,monospace;"
    S_DIV = "border:none;border-top:1px solid rgba(255,255,255,.1);margin:8px 0;"
    S_RES = "display:flex;justify-content:space-between;padding:4px 0 0 0;"
    S_RES_VAR = "color:#a5b4fc !important;font-weight:600;"
    S_RES_NUM = "color:#34d399 !important;font-weight:700;font-size:14px;font-family:'SF Mono','Fira Code',Consolas,monospace;"
    S_NOTE = "font-size:11px;color:#94a3b8 !important;margin:8px 0 0 0;"

    def _rrow(label, val):
        return f"<div style='{S_ROW}'><span style='{S_VAR}'>{label}</span><span style='{S_NUM}'>{val}</span></div>"

    def _rresult(label, val):
        return f"<div style='{S_RES}'><span style='{S_RES_VAR}'>{label}</span><span style='{S_RES_NUM}'>{val}</span></div>"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(_fc_risk(
            "Sortino Ratio", f"{sortino:.2f}",
            mform_r(m_r("S"), mop_r("="), mf_r("R<sub>p</sub> &minus; R<sub>f</sub>", "&sigma;<sub>downside</sub>"))
            + _rrow("R<sub>p</sub> (annualized)", f"{ann_return:.4f}")
            + _rrow("R<sub>f</sub> (annual)", f"{rf_annual_rm:.2%}")
            + _rrow("&sigma;<sub>downside</sub> (ann.)", f"{downside_std_annual:.4f}")
            + f"<hr style='{S_DIV}'>"
            + _rresult("= Sortino", f"{sortino:.4f}")
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(_fc_risk(
            "Value at Risk (95%)", f"{var_95*100:.2f}%",
            mform_r(m_r("VaR<sub>95</sub>"), mop_r("="), m_r("&minus;Q(R<sub>p</sub>,"), m_r("0.05)"))
            + _rrow("5th percentile", f"{(-var_95)*100:.2f}%")
            + f"<hr style='{S_DIV}'>"
            + _rresult("= VaR 95", f"{var_95*100:.2f}%")
            + f"<p style='{S_NOTE}'>Maximum daily loss at 95% confidence</p>"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(_fc_risk(
            "Conditional VaR (95%)", f"{cvar_95*100:.2f}%",
            mform_r(m_r("CVaR<sub>95</sub>"), mop_r("="), m_r("&minus;E[R<sub>p</sub>"), mop_r("|"), m_r("R<sub>p</sub>"), mop_r("&le;"), m_r("&minus;VaR]"))
            + _rrow("Tail mean", f"{cvar_tail_mean*100:.2f}%")
            + _rrow("Tail observations", f"{len(tail_losses)}")
            + f"<hr style='{S_DIV}'>"
            + _rresult("= CVaR 95", f"{cvar_95*100:.2f}%")
            + f"<p style='{S_NOTE}'>Expected loss in the worst 5% of days</p>"
        ), unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    
    # Two column layout for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px;'>Asset Correlation Matrix</h3>", unsafe_allow_html=True)
        
        corr_matrix = de.calculate_correlation_matrix(returns_df)
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            aspect='auto'
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            coloraxis_colorbar=dict(
                title=dict(text="Correlation", font=dict(color='#cbd5e1', size=12)),
                tickfont=dict(color='#cbd5e1', size=11)
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', size=12),
            xaxis=dict(tickfont=dict(color='#cbd5e1', size=11)),
            yaxis=dict(tickfont=dict(color='#cbd5e1', size=11))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-top: 10px;'>📊 Correlation between asset returns. High correlation = less diversification benefit.</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px;'>Risk Contribution by Asset</h3>", unsafe_allow_html=True)
        
        risk_contrib = de.calculate_risk_contribution(returns_df, weights)
        
        fig = go.Figure(data=[go.Pie(
            labels=tickers,
            values=risk_contrib * 100,
            hole=0.4,
            marker=dict(
                colors=['#818cf8', '#a855f7', '#c084fc', '#38bdf8', '#22d3ee', '#34d399', '#f472b6'],
                line=dict(color='#0f172a', width=2)
            ),
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=12, color='#e2e8f0')
        )])
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(color='#e2e8f0', size=12)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=12),
            margin=dict(l=20, r=100, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-top: 10px;'>⚠️ Risk contribution = Weight × Volatility (normalized). Shows which assets drive portfolio risk.</p>", unsafe_allow_html=True)
    
    # Returns distribution
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px;'>Portfolio Returns Distribution</h3>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=portfolio_returns * 100,
        nbinsx=50,
        name='Returns',
        marker=dict(
            color='#818cf8',
            line=dict(color='#6366f1', width=1)
        ),
        opacity=0.8
    ))
    
    # Add VaR line
    fig.add_vline(
        x=-var_95 * 100,
        line_dash="dash",
        line_color="#c084fc",
        annotation_text=f"VaR 95%: {var_95*100:.2f}%",
        annotation_position="top",
        annotation=dict(font_color='#e2e8f0')
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', size=12),
        xaxis=dict(title_font=dict(color='#cbd5e1', size=13), tickfont=dict(color='#cbd5e1', size=11)),
        yaxis=dict(title_font=dict(color='#cbd5e1', size=13), tickfont=dict(color='#cbd5e1', size=11))
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: FACTOR ANALYSIS
# ============================================================================

with tab3:
    st.markdown("<h2 style='color: #f1f5f9; margin-bottom: 30px;'>Deep Dive: Monthly Performance Analysis</h2>", unsafe_allow_html=True)
    
    # Get monthly returns
    monthly_returns = de.get_monthly_returns(portfolio_returns)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns.values * 100,
        x=monthly_returns.columns,
        y=monthly_returns.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(monthly_returns.values * 100, 2),
        texttemplate='%{text}%',
        textfont={"size": 11, "color": "#1f2937"},
        colorbar=dict(
            title=dict(text="Return (%)", font=dict(color='#e2e8f0', size=12)),
            tickfont=dict(color='#e2e8f0', size=11)
        )
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=max(400, len(monthly_returns) * 40),
        xaxis_title="Month",
        yaxis_title="Year",
        yaxis=dict(autorange='reversed', title_font=dict(color='#e2e8f0', size=13), tickfont=dict(color='#e2e8f0', size=11)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', size=12),
        xaxis=dict(title_font=dict(color='#e2e8f0', size=13), tickfont=dict(color='#e2e8f0', size=11))
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-top: 10px;'>🔬 Monthly returns heatmap. Green = positive, Red = negative. Quickly spot seasonal patterns.</p>", unsafe_allow_html=True)
    
    # Monthly statistics
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px;'>Monthly Statistics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    monthly_data = (portfolio_returns + 1).resample('M').prod() - 1
    
    with col1:
        best_month = monthly_data.max()
        st.metric(
            label="Best Month",
            value=f"{best_month*100:.2f}%",
            help="Highest monthly return in the period"
        )
    
    with col2:
        worst_month = monthly_data.min()
        st.metric(
            label="Worst Month",
            value=f"{worst_month*100:.2f}%",
            help="Lowest monthly return in the period"
        )
    
    with col3:
        avg_monthly = monthly_data.mean()
        st.metric(
            label="Avg Monthly Return",
            value=f"{avg_monthly*100:.2f}%",
            help="Average monthly return"
        )
    
    with col4:
        positive_months = (monthly_data > 0).sum() / len(monthly_data) * 100
        st.metric(
            label="Positive Months",
            value=f"{positive_months:.1f}%",
            help="Percentage of months with positive returns"
        )
    
    # Asset contribution analysis
    st.markdown("<div style='margin: 40px 0; border-top: 1px solid rgba(148,163,184,0.12);'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #f1f5f9; margin-bottom: 20px;'>Asset Performance Breakdown</h3>", unsafe_allow_html=True)
    
    # Button-style time period selector
    asset_time_period = st.radio(
        "Asset Time Period",
        options=["3M", "6M", "YTD", "1Y", "3Y", "5Y", "ALL"],
        index=3,  # Default to 1Y
        horizontal=True,
        label_visibility="collapsed",
        key="asset_time_period"
    )
    
    # Calculate date range based on selection
    asset_end_date = returns_df.index[-1]
    if asset_time_period == "3M":
        asset_start_date = asset_end_date - pd.Timedelta(days=90)
    elif asset_time_period == "6M":
        asset_start_date = asset_end_date - pd.Timedelta(days=180)
    elif asset_time_period == "YTD":
        asset_start_date = pd.Timestamp(f"{asset_end_date.year}-01-01")
    elif asset_time_period == "1Y":
        asset_start_date = asset_end_date - pd.Timedelta(days=365)
    elif asset_time_period == "3Y":
        asset_start_date = asset_end_date - pd.Timedelta(days=365*3)
    elif asset_time_period == "5Y":
        asset_start_date = asset_end_date - pd.Timedelta(days=365*5)
    else:  # ALL
        asset_start_date = returns_df.index[0]
    
    # Filter returns to selected period
    filtered_returns_df = returns_df[returns_df.index >= asset_start_date]
    
    # Calculate individual asset cumulative returns from filtered period
    asset_cumulative = {}
    for ticker in tickers:
        if ticker in filtered_returns_df.columns:
            asset_cumulative[ticker] = de.calculate_cumulative_returns(filtered_returns_df[ticker])
    
    fig = go.Figure()
    
    # Modern gradient color palette
    colors = ['#818cf8', '#a855f7', '#c084fc', '#38bdf8', '#22d3ee', '#34d399', '#f472b6', '#fb923c']
    
    for i, ticker in enumerate(tickers):
        if ticker in asset_cumulative:
            fig.add_trace(go.Scatter(
                x=asset_cumulative[ticker].index,
                y=(asset_cumulative[ticker] - 1) * 100,
                mode='lines',
                name=ticker,
                line=dict(width=2, color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#cbd5e1', size=12)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', size=12),
        xaxis=dict(
            title_font=dict(color='#cbd5e1', size=13), 
            tickfont=dict(color='#cbd5e1', size=11),
            gridcolor='rgba(148,163,184,0.1)'
        ),
        yaxis=dict(
            title_font=dict(color='#cbd5e1', size=13), 
            tickfont=dict(color='#cbd5e1', size=11),
            gridcolor='rgba(148,163,184,0.1)',
            zeroline=True,
            zerolinecolor='rgba(148,163,184,0.15)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-top: 10px;'>Individual asset performance over the selected period. Returns start at 0% from the beginning of the period.</p>", unsafe_allow_html=True)



# ============================================================================
# TAB NEWS: PORTFOLIO NEWS
# ============================================================================

with tab_news:
    st.markdown("<h2 style='color: #f1f5f9 !important; margin-bottom: 10px;'>Portfolio News Feed</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8 !important; font-size: 15px; margin-bottom: 30px;'>Significant recent news for your holdings, sourced from Yahoo Finance</p>", unsafe_allow_html=True)

    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_portfolio_news(ticker_list):
        """Fetch news for each ticker, filter to significant articles."""
        import yfinance as yf
        from datetime import datetime, timezone, timedelta
        all_articles = []
        seen_titles = set()
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=14)

        for ticker_symbol in ticker_list:
            try:
                t = yf.Ticker(ticker_symbol)
                news_items = t.news or []
                for item in news_items:
                    # yfinance wraps data inside a 'content' dict
                    content = item.get("content", item)

                    title = content.get("title", "")
                    if not title or title in seen_titles:
                        continue

                    summary = content.get("summary", "")

                    # Link: try canonical or clickThrough url
                    link = ""
                    canonical = content.get("canonicalUrl") or {}
                    click_through = content.get("clickThroughUrl") or {}
                    link = canonical.get("url", "") or click_through.get("url", "")

                    # Publisher
                    provider = content.get("provider") or {}
                    publisher = provider.get("displayName", "Unknown")

                    # Published date (ISO string)
                    pub_date_str = content.get("pubDate", "")
                    try:
                        pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    except Exception:
                        pub_dt = datetime.now(tz=timezone.utc)

                    # Skip articles older than 14 days
                    if pub_dt < cutoff:
                        continue

                    # Thumbnail
                    thumbnail = ""
                    thumb_data = content.get("thumbnail") or {}
                    if isinstance(thumb_data, dict):
                        resolutions = thumb_data.get("resolutions", [])
                        if resolutions:
                            # Pick the smallest that's still reasonable
                            thumbnail = resolutions[0].get("url", "")

                    seen_titles.add(title)
                    all_articles.append({
                        "title": title,
                        "summary": summary,
                        "link": link,
                        "publisher": publisher,
                        "published": pub_dt,
                        "thumbnail": thumbnail,
                        "source_ticker": ticker_symbol,
                    })
            except Exception:
                continue

        # Sort by most recent first
        all_articles.sort(key=lambda x: x["published"], reverse=True)
        return all_articles

    with st.spinner("Fetching latest news..."):
        news_articles = fetch_portfolio_news(tickers)

    if not news_articles:
        st.markdown("""
            <div style='background: rgba(30,41,59,0.65); padding: 40px; border-radius: 12px;
                        border: 1px solid rgba(148,163,184,0.12); text-align: center;'>
                <p style='color: #94a3b8 !important; font-size: 16px; margin: 0;'>
                    No recent news found for your portfolio holdings.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Sort control and article count
        sort_col1, sort_col2 = st.columns([1, 3])
        with sort_col1:
            st.markdown("<p style='color: #94a3b8 !important; font-size: 12px; font-weight: 600; margin: 0 0 4px 0; letter-spacing: .3px;'>Sort By:</p>", unsafe_allow_html=True)
            news_sort = st.radio("Sort by", ["Most Recent", "Ticker"], key="news_sort", horizontal=True, label_visibility="collapsed")
        with sort_col2:
            st.markdown(f"<p style='color: #64748b !important; font-size: 13px; margin-top: 24px; text-align: right;'>{len(news_articles)} article{'s' if len(news_articles) != 1 else ''} found</p>", unsafe_allow_html=True)

        # Apply sorting
        if news_sort == "Ticker":
            ticker_order = {t: i for i, t in enumerate(tickers)}
            news_articles = sorted(news_articles, key=lambda a: (ticker_order.get(a["source_ticker"], 999), -a["published"].timestamp()))
        # else already sorted by most recent from fetch

        from datetime import datetime, timezone
        now_utc = datetime.now(tz=timezone.utc)

        # Get today's return per ticker for green/red badge color
        ticker_day_colors = {}
        for tk in tickers:
            try:
                if tk in returns_df.columns and len(returns_df[tk]) > 0:
                    last_ret = returns_df[tk].iloc[-1]
                    if not pd.isna(last_ret):
                        ticker_day_colors[tk] = "#00e676" if last_ret >= 0 else "#ff1744"
                    else:
                        ticker_day_colors[tk] = "#818cf8"
                else:
                    ticker_day_colors[tk] = "#818cf8"
            except Exception:
                ticker_day_colors[tk] = "#818cf8"

        for article in news_articles:
            # Time ago calculation
            delta = now_utc - article["published"]
            if delta.days > 0:
                time_ago = f"{delta.days}d ago"
            elif delta.seconds >= 3600:
                time_ago = f"{delta.seconds // 3600}h ago"
            else:
                time_ago = f"{max(1, delta.seconds // 60)}m ago"

            # Build summary snippet
            summary_html = ""
            if article.get("summary"):
                summary_html = (
                    "<p style='color:#94a3b8 !important;font-size:13px;margin:0 0 10px 0;"
                    "line-height:1.5;display:-webkit-box;-webkit-line-clamp:2;"
                    "-webkit-box-orient:vertical;overflow:hidden;'>"
                    + article["summary"] + "</p>"
                )

            # Build thumbnail
            thumb_html = ""
            if article.get("thumbnail"):
                thumb_url = article["thumbnail"]
                thumb_html = (
                    "<div style='flex-shrink:0;width:120px;height:80px;border-radius:8px;"
                    "overflow:hidden;margin-left:16px;'>"
                    "<img src='" + thumb_url + "' "
                    "style='width:100%;height:100%;object-fit:cover;' />"
                    "</div>"
                )

            # Ticker badge color: green if up today, red if down
            tk_sym = article["source_ticker"]
            tk_color = ticker_day_colors.get(tk_sym, "#818cf8")
            tk_bg = "rgba(0,230,118,0.15)" if tk_color == "#00e676" else "rgba(255,23,68,0.15)" if tk_color == "#ff1744" else "rgba(129,140,248,0.15)"

            # Build full card HTML as a single string
            card_html = (
                "<a href='" + article["link"] + "' target='_blank' style='text-decoration:none;display:block;'>"
                "<div style='background:rgba(30,41,59,0.65);padding:20px 24px;border-radius:12px;"
                "border:1px solid rgba(148,163,184,0.08);margin-bottom:12px;"
                "backdrop-filter:blur(12px);display:flex;align-items:center;cursor:pointer;'>"
                "<div style='flex:1;min-width:0;'>"
                "<p style='color:#f1f5f9 !important;font-size:15px;font-weight:600;margin:0 0 6px 0;"
                "line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;"
                "-webkit-box-orient:vertical;overflow:hidden;'>"
                + article["title"] + "</p>"
                + summary_html
                + "<div style='display:flex;align-items:center;flex-wrap:wrap;margin-top:6px;'>"
                "<span style='background:" + tk_bg + ";color:" + tk_color + " !important;"
                "padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700;"
                "letter-spacing:.3px;'>" + tk_sym + "</span>"
                "<span style='color:#475569 !important;font-size:12px;margin:0 10px;'>&bull;</span>"
                "<span style='color:#94a3b8 !important;font-size:12px;'>" + article["publisher"] + "</span>"
                "<span style='color:#475569 !important;font-size:12px;margin:0 10px;'>&bull;</span>"
                "<span style='color:#94a3b8 !important;font-size:12px;'>" + time_ago + "</span>"
                "</div></div>"
                + thumb_html
                + "</div></a>"
            )

            st.markdown(card_html, unsafe_allow_html=True)

        # Footer note
        st.markdown("""
            <div style='text-align: center; margin-top: 30px;'>
                <p style='color: #475569 !important; font-size: 12px;'>
                    News sourced from Yahoo Finance. Articles are filtered for relevance to your holdings.
                    Refreshes every 15 minutes.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 4: FINANCIAL ADVISOR
# ============================================================================

with tab4:
    st.markdown("<h2 style='color: #f1f5f9 !important; margin-bottom: 10px;'>Portfolio Research & Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8 !important; font-size: 15px; margin-bottom: 30px;'>Automated analysis based on your portfolio's computed metrics</p>", unsafe_allow_html=True)

    # Recalculate all metrics needed for advisor (some already computed in earlier tabs)
    adv_ann_return = de.calculate_annualized_return(portfolio_returns)
    adv_ann_vol = de.calculate_annualized_volatility(portfolio_returns)
    adv_sharpe = de.calculate_sharpe_ratio(portfolio_returns)
    adv_sortino = de.calculate_sortino_ratio(portfolio_returns)
    adv_max_dd = de.calculate_max_drawdown(portfolio_returns)
    adv_alpha, adv_beta = de.calculate_alpha_beta(portfolio_returns, benchmark_returns)
    adv_var_95 = de.calculate_var(portfolio_returns, confidence=0.95)
    adv_cvar_95 = de.calculate_cvar(portfolio_returns, confidence=0.95)
    adv_weights = list(st.session_state.weights)
    adv_tickers = list(st.session_state.tickers)
    adv_weights_sq = sum([w**2 for w in adv_weights])
    adv_n_eff = 1 / adv_weights_sq if adv_weights_sq > 0 else len(adv_weights)
    adv_corr = de.calculate_correlation_matrix(returns_df)
    adv_win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) * 100 if len(portfolio_returns) > 0 else 0
    adv_max_weight = max(adv_weights)
    adv_max_weight_idx = adv_weights.index(adv_max_weight)
    adv_max_weight_ticker = adv_tickers[adv_max_weight_idx]

    # ── Build advisory sections ──

    # Helper for section cards
    def advisor_card(icon, title, content, color="#818cf8"):
        st.markdown(f"""
            <div style='background: rgba(30,41,59,0.65); padding: 24px 28px; border-radius: 12px;
                        border: 1px solid rgba(148,163,184,0.12); margin-bottom: 20px;
                        border-left: 3px solid {color}; backdrop-filter: blur(12px);'>
                <p style='color: {color} !important; font-size: 13px; font-weight: 700; margin: 0 0 8px 0;
                          text-transform: uppercase; letter-spacing: .5px;'>
                    {icon} {title}
                </p>
                <div style='color: #cbd5e1 !important; font-size: 14px; line-height: 1.8;'>
                    {content}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── 1. Portfolio Health Score ──
    health_score = 0
    health_notes = []

    # Sharpe
    if adv_sharpe >= 1.0:
        health_score += 25
        health_notes.append("Excellent risk-adjusted returns (Sharpe &ge; 1.0)")
    elif adv_sharpe >= 0.5:
        health_score += 15
        health_notes.append("Decent risk-adjusted returns (Sharpe 0.5&ndash;1.0)")
    else:
        health_score += 5
        health_notes.append("Low risk-adjusted returns (Sharpe &lt; 0.5)")

    # Diversification
    if adv_n_eff >= len(adv_tickers) * 0.6:
        health_score += 25
        health_notes.append("Well-diversified &mdash; weight distribution is balanced")
    elif adv_n_eff >= len(adv_tickers) * 0.3:
        health_score += 15
        health_notes.append("Moderately diversified &mdash; some concentration risk")
    else:
        health_score += 5
        health_notes.append("Highly concentrated &mdash; a few positions dominate")

    # Max Drawdown
    if abs(adv_max_dd) < 0.15:
        health_score += 25
        health_notes.append("Drawdown well-contained (under 15%)")
    elif abs(adv_max_dd) < 0.30:
        health_score += 15
        health_notes.append("Moderate drawdown risk (15&ndash;30%)")
    else:
        health_score += 5
        health_notes.append("High drawdown risk (over 30%)")

    # Alpha
    if adv_alpha > 0:
        health_score += 25
        health_notes.append("Positive alpha &mdash; outperforming the benchmark")
    elif adv_alpha > -0.02:
        health_score += 15
        health_notes.append("Near-zero alpha &mdash; tracking close to benchmark")
    else:
        health_score += 5
        health_notes.append("Negative alpha &mdash; underperforming the benchmark")

    score_color = "#34d399" if health_score >= 70 else "#fbbf24" if health_score >= 45 else "#f87171"
    score_label = "Strong" if health_score >= 70 else "Moderate" if health_score >= 45 else "Needs Attention"

    st.markdown(f"""
        <div style='background: rgba(30,41,59,0.65); padding: 28px 32px; border-radius: 14px;
                    border: 1px solid rgba(148,163,184,0.12); margin-bottom: 30px;
                    backdrop-filter: blur(12px); text-align: center;'>
            <p style='color: #94a3b8 !important; font-size: 12px; font-weight: 600; margin: 0 0 8px 0;
                      text-transform: uppercase; letter-spacing: .8px;'>
                Portfolio Health Score
            </p>
            <p style='color: {score_color} !important; font-size: 56px; font-weight: 800; margin: 0;
                      line-height: 1; font-family: Inter, system-ui, sans-serif;'>
                {health_score}
            </p>
            <p style='color: {score_color} !important; font-size: 16px; font-weight: 600; margin: 6px 0 0 0;'>
                {score_label}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── 2. Return Analysis ──
    ret_text = f"Your portfolio's <strong style='color:#f1f5f9 !important;'>annualized return is {adv_ann_return*100:.2f}%</strong>. "
    if adv_ann_return > 0.12:
        ret_text += "This is a strong return profile, significantly outpacing typical long-term market averages (~8&ndash;10% annualized for the S&P 500). "
        ret_text += "Consider whether this is driven by a few high-growth holdings or broad-based gains. High returns often come with elevated risk."
    elif adv_ann_return > 0.06:
        ret_text += "This is a solid return roughly in line with or slightly above historical equity market averages. "
        ret_text += "The portfolio appears to be generating reasonable growth without excessive risk-taking."
    elif adv_ann_return > 0:
        ret_text += "This is a modest return, underperforming typical equity benchmarks. "
        ret_text += "Consider reviewing your asset selection. If capital preservation is your goal, this may be acceptable. Otherwise, assess whether low-performing positions are dragging returns."
    else:
        ret_text += "The portfolio is currently negative. This may reflect a challenging market environment or specific stock selection issues. "
        ret_text += "Review which holdings are underperforming and assess whether the investment thesis still holds."

    advisor_card("📈", "Return Analysis", ret_text, "#818cf8")

    # ── 3. Risk Assessment ──
    risk_text = f"<strong style='color:#f1f5f9 !important;'>Annualized volatility: {adv_ann_vol*100:.1f}%</strong> &bull; "
    risk_text += f"<strong style='color:#f1f5f9 !important;'>Max drawdown: {adv_max_dd*100:.1f}%</strong> &bull; "
    risk_text += f"<strong style='color:#f1f5f9 !important;'>Beta: {adv_beta:.2f}</strong><br><br>"

    if adv_ann_vol > 0.25:
        risk_text += "Volatility is <strong style='color:#f87171 !important;'>high</strong>. The portfolio experiences large daily swings. This level of volatility can lead to significant short-term losses and may not be suitable for risk-averse investors. "
    elif adv_ann_vol > 0.15:
        risk_text += "Volatility is <strong style='color:#fbbf24 !important;'>moderate</strong>, typical for a stock-heavy portfolio. "
    else:
        risk_text += "Volatility is <strong style='color:#34d399 !important;'>low</strong>, suggesting a conservative or well-hedged portfolio. "

    if adv_beta > 1.3:
        risk_text += f"With a beta of {adv_beta:.2f}, your portfolio amplifies market moves &mdash; it rises faster in bull markets but falls harder in downturns. "
    elif adv_beta > 0.8:
        risk_text += f"Beta of {adv_beta:.2f} indicates market-like sensitivity. "
    else:
        risk_text += f"Beta of {adv_beta:.2f} suggests defensive characteristics &mdash; less sensitive to broad market swings. "

    risk_text += f"<br><br>The maximum drawdown of <strong style='color:#f1f5f9 !important;'>{adv_max_dd*100:.1f}%</strong> represents the worst peak-to-trough decline observed. "
    if abs(adv_max_dd) > 0.30:
        risk_text += "This is a significant drawdown. Consider whether you could tolerate a similar decline without panic-selling."
    elif abs(adv_max_dd) > 0.15:
        risk_text += "This is within normal bounds for equity portfolios but still represents meaningful pain during downturns."
    else:
        risk_text += "This is a relatively contained drawdown, suggesting good downside protection."

    advisor_card("⚠️", "Risk Assessment", risk_text, "#f87171")

    # ── 4. Diversification Review ──
    div_text = f"<strong style='color:#f1f5f9 !important;'>Effective positions: {adv_n_eff:.1f}</strong> out of {len(adv_tickers)} holdings &bull; "
    div_text += f"<strong style='color:#f1f5f9 !important;'>Largest position: {adv_max_weight_ticker} ({adv_max_weight*100:.1f}%)</strong><br><br>"

    if adv_max_weight > 0.40:
        div_text += f"<strong style='color:#f87171 !important;'>Warning:</strong> {adv_max_weight_ticker} represents {adv_max_weight*100:.0f}% of your portfolio. "
        div_text += "A single position over 40% creates significant idiosyncratic risk. A negative event specific to this asset could severely impact your portfolio. "
        div_text += "Consider trimming to reduce concentration."
    elif adv_max_weight > 0.25:
        div_text += f"{adv_max_weight_ticker} at {adv_max_weight*100:.0f}% is a sizable position. Monitor it closely and consider rebalancing if it grows further."
    else:
        div_text += "Position sizing looks balanced. No single holding dominates the portfolio excessively."

    # Check for high correlations
    high_corr_pairs = []
    for i in range(len(adv_tickers)):
        for j in range(i+1, len(adv_tickers)):
            if adv_tickers[i] in adv_corr.columns and adv_tickers[j] in adv_corr.columns:
                corr_val = adv_corr.loc[adv_tickers[i], adv_tickers[j]]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((adv_tickers[i], adv_tickers[j], corr_val))

    if high_corr_pairs:
        div_text += "<br><br><strong style='color:#fbbf24 !important;'>Correlation alert:</strong> "
        for t1, t2, cv in high_corr_pairs[:3]:
            div_text += f"{t1}/{t2} ({cv:.2f}), "
        div_text = div_text.rstrip(", ")
        div_text += " have high correlation (&gt;0.8). These positions move together and provide limited diversification benefit. "
        div_text += "Consider substituting one with an uncorrelated asset."
    elif len(adv_tickers) > 1:
        div_text += "<br><br>No highly correlated pairs detected (&gt;0.8). The holdings provide good diversification benefit across different risk exposures."

    advisor_card("🎯", "Diversification Review", div_text, "#fbbf24")

    # ── 5. Risk-Adjusted Performance ──
    perf_text = f"<strong style='color:#f1f5f9 !important;'>Sharpe: {adv_sharpe:.2f}</strong> &bull; "
    perf_text += f"<strong style='color:#f1f5f9 !important;'>Sortino: {adv_sortino:.2f}</strong> &bull; "
    perf_text += f"<strong style='color:#f1f5f9 !important;'>Alpha: {adv_alpha*100:+.2f}%</strong><br><br>"

    if adv_sharpe >= 1.0:
        perf_text += "A Sharpe ratio above 1.0 is considered very good &mdash; you're earning strong returns per unit of risk taken. "
    elif adv_sharpe >= 0.5:
        perf_text += "A Sharpe ratio between 0.5&ndash;1.0 is acceptable but suggests room for improvement. "
    else:
        perf_text += "A Sharpe below 0.5 indicates poor compensation for the risk being taken. The same return could potentially be achieved with less volatility. "

    if adv_sortino > adv_sharpe:
        perf_text += f"The Sortino ratio ({adv_sortino:.2f}) is higher than the Sharpe ({adv_sharpe:.2f}), which means most of your volatility is on the upside &mdash; a positive sign. "
    elif adv_sortino < adv_sharpe * 0.8:
        perf_text += f"The Sortino ({adv_sortino:.2f}) being lower than Sharpe ({adv_sharpe:.2f}) suggests meaningful downside volatility. Your losses tend to be outsized relative to gains."

    if adv_alpha > 0.02:
        perf_text += f"<br><br>Alpha of {adv_alpha*100:+.2f}% indicates the portfolio is generating excess returns beyond what the benchmark (SPY) provides for the level of risk. This is the hallmark of good stock selection."
    elif adv_alpha > -0.02:
        perf_text += f"<br><br>Alpha near zero ({adv_alpha*100:+.2f}%) means the portfolio is performing roughly as expected given its risk exposure. Consider whether the added complexity over a simple index fund is justified."
    else:
        perf_text += f"<br><br>Negative alpha ({adv_alpha*100:+.2f}%) means the portfolio is underperforming on a risk-adjusted basis versus SPY. A simple index fund would have delivered better results for the same risk."

    advisor_card("📊", "Risk-Adjusted Performance", perf_text, "#34d399")

    # ── 6. Tail Risk & Downside ──
    tail_text = f"<strong style='color:#f1f5f9 !important;'>VaR (95%): {adv_var_95*100:.2f}%</strong> &bull; "
    tail_text += f"<strong style='color:#f1f5f9 !important;'>CVaR (95%): {adv_cvar_95*100:.2f}%</strong> &bull; "
    tail_text += f"<strong style='color:#f1f5f9 !important;'>Win rate: {adv_win_rate:.1f}%</strong><br><br>"

    tail_text += f"On a typical bad day (95% confidence), you could lose up to <strong style='color:#f87171 !important;'>{adv_var_95*100:.2f}%</strong> of portfolio value. "
    tail_text += f"In extreme scenarios (worst 5% of days), the average loss would be <strong style='color:#f87171 !important;'>{adv_cvar_95*100:.2f}%</strong>. "

    portfolio_val = st.session_state.total_value
    dollar_var = portfolio_val * adv_var_95
    dollar_cvar = portfolio_val * adv_cvar_95
    tail_text += f"<br><br>In dollar terms on a ${portfolio_val:,.0f} portfolio: a bad day could cost ~<strong style='color:#f1f5f9 !important;'>${dollar_var:,.0f}</strong>, "
    tail_text += f"and a worst-case day ~<strong style='color:#f1f5f9 !important;'>${dollar_cvar:,.0f}</strong>. "

    if adv_win_rate > 55:
        tail_text += f"<br><br>With a win rate of {adv_win_rate:.1f}%, the portfolio has more positive days than negative &mdash; a sign of consistent performance."
    elif adv_win_rate > 48:
        tail_text += f"<br><br>Win rate of {adv_win_rate:.1f}% is near 50/50. Profitability depends on the magnitude of gains exceeding losses, not frequency."
    else:
        tail_text += f"<br><br>Win rate below 50% ({adv_win_rate:.1f}%) means more losing days than winning. The portfolio relies on larger gains to offset frequent small losses."

    advisor_card("🛡️", "Tail Risk & Downside Protection", tail_text, "#c084fc")

    # ── 7. Actionable Suggestions ──
    suggestions = []

    if adv_max_weight > 0.35:
        suggestions.append(f"<strong style='color:#f1f5f9 !important;'>Reduce concentration:</strong> {adv_max_weight_ticker} at {adv_max_weight*100:.0f}% is oversized. Consider trimming to under 25% and reallocating to uncorrelated assets.")

    if abs(adv_max_dd) > 0.25:
        suggestions.append("<strong style='color:#f1f5f9 !important;'>Add downside protection:</strong> The portfolio has experienced a significant drawdown. Consider adding defensive positions (bonds, gold, low-beta stocks) or using stop-loss strategies.")

    if adv_sharpe < 0.5:
        suggestions.append("<strong style='color:#f1f5f9 !important;'>Improve risk-adjusted returns:</strong> The Sharpe ratio is below 0.5. Review underperforming holdings and consider replacing them with assets that offer better return per unit of risk.")

    if adv_n_eff < 3 and len(adv_tickers) >= 3:
        suggestions.append("<strong style='color:#f1f5f9 !important;'>Rebalance weights:</strong> Despite holding multiple assets, the effective diversification is low. Rebalancing toward equal weights would improve risk distribution.")

    if adv_beta > 1.3:
        suggestions.append("<strong style='color:#f1f5f9 !important;'>Lower market sensitivity:</strong> A beta above 1.3 means amplified exposure to market downturns. Consider adding low-beta or market-neutral positions.")

    if adv_alpha < -0.02:
        suggestions.append("<strong style='color:#f1f5f9 !important;'>Revisit stock selection:</strong> Negative alpha suggests underperformance vs. benchmark. Consider whether an index fund (SPY/VOO) would serve you better for the core allocation.")

    if high_corr_pairs:
        suggestions.append("<strong style='color:#f1f5f9 !important;'>Reduce correlated holdings:</strong> Some positions move in lockstep. Swap one correlated asset for something in a different sector or asset class (e.g., international, fixed income, commodities).")

    if len(adv_tickers) < 5:
        suggestions.append("<strong style='color:#f1f5f9 !important;'>Increase holdings count:</strong> With fewer than 5 positions, idiosyncratic risk is high. Adding 3&ndash;5 more uncorrelated positions could significantly reduce portfolio-specific risk.")

    if not suggestions:
        suggestions.append("Your portfolio metrics look well-balanced. Continue monitoring and rebalance periodically (quarterly or semi-annually) to maintain your target allocations.")

    sugg_html = "<ol style='margin:0;padding-left:20px;'>"
    for s in suggestions:
        sugg_html += f"<li style='margin-bottom:10px;color:#cbd5e1 !important;'>{s}</li>"
    sugg_html += "</ol>"

    advisor_card("💡", "Actionable Suggestions", sugg_html, "#38bdf8")

    # ── Metrics Summary Table ──
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #f1f5f9 !important; margin-bottom: 20px;'>Metrics Summary</h3>", unsafe_allow_html=True)

    metrics_summary = pd.DataFrame({
        'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                    'Max Drawdown', 'Alpha', 'Beta', 'VaR (95%)', 'CVaR (95%)',
                    'Effective Positions', 'Win Rate'],
        'Value': [f"{adv_ann_return*100:.2f}%", f"{adv_ann_vol*100:.2f}%", f"{adv_sharpe:.2f}",
                  f"{adv_sortino:.2f}", f"{adv_max_dd*100:.2f}%", f"{adv_alpha*100:+.2f}%",
                  f"{adv_beta:.2f}", f"{adv_var_95*100:.2f}%", f"{adv_cvar_95*100:.2f}%",
                  f"{adv_n_eff:.1f} / {len(adv_tickers)}", f"{adv_win_rate:.1f}%"],
        'Assessment': [
            "Strong" if adv_ann_return > 0.10 else "Moderate" if adv_ann_return > 0.04 else "Weak",
            "Low" if adv_ann_vol < 0.15 else "Moderate" if adv_ann_vol < 0.25 else "High",
            "Excellent" if adv_sharpe >= 1 else "Good" if adv_sharpe >= 0.5 else "Poor",
            "Excellent" if adv_sortino >= 1 else "Good" if adv_sortino >= 0.5 else "Poor",
            "Contained" if abs(adv_max_dd) < 0.15 else "Moderate" if abs(adv_max_dd) < 0.30 else "Severe",
            "Outperforming" if adv_alpha > 0.01 else "Neutral" if adv_alpha > -0.01 else "Underperforming",
            "Defensive" if adv_beta < 0.8 else "Market-like" if adv_beta < 1.2 else "Aggressive",
            "Low" if adv_var_95 < 0.015 else "Moderate" if adv_var_95 < 0.025 else "High",
            "Low" if adv_cvar_95 < 0.02 else "Moderate" if adv_cvar_95 < 0.035 else "High",
            "Well Diversified" if adv_n_eff >= len(adv_tickers) * 0.6 else "Concentrated",
            "Strong" if adv_win_rate > 54 else "Average" if adv_win_rate > 48 else "Below Average",
        ]
    })

    st.dataframe(metrics_summary, use_container_width=True, hide_index=True)

    # ── Disclaimer ──
    st.markdown("""
        <div style='background: rgba(248,113,113,0.06); padding: 20px 24px; border-radius: 12px;
                    border: 1px solid rgba(248,113,113,0.15); margin-top: 40px;'>
            <p style='color: #f87171 !important; font-size: 13px; font-weight: 700; margin: 0 0 8px 0;
                      text-transform: uppercase; letter-spacing: .5px;'>
                ⚠️ Important Disclaimer
            </p>
            <p style='color: #cbd5e1 !important; font-size: 13px; line-height: 1.8; margin: 0;'>
                This analysis is generated automatically using quantitative metrics and rule-based logic.
                It is intended <strong style='color:#f1f5f9 !important;'>purely for educational and research purposes</strong> and does
                <strong style='color:#f87171 !important;'>not constitute financial advice</strong>.
                Past performance does not guarantee future results. All investments carry risk, including the
                potential loss of principal. Consult a licensed financial advisor before making any investment decisions.
                The creators of this tool accept no liability for any actions taken based on this analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
    <div style='text-align: center; padding: 20px 20px 16px 20px; 
                margin-top: 30px; border-top: 1px solid rgba(148,163,184,0.08);'>
        <p style='font-size: 11px; color: #475569 !important; margin: 0;'>
            Disclaimer: This tool is for educational purposes only. Not financial advice.
        </p>
    </div>
""", unsafe_allow_html=True)

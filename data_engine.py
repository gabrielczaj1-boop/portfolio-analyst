"""
Data Engine for Portfolio Analysis
Handles data fetching, caching, and all financial calculations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st
import re


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(tickers, years=10):
    """
    Fetch historical price data for given tickers
    
    Args:
        tickers: List of ticker symbols
        years: Number of years of historical data (default 10 for max history)
        
    Returns:
        DataFrame with adjusted close prices
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    data = {}
    failed_tickers = []
    error_messages = {}
    
    for ticker in tickers:
        try:
            # Try yfinance Ticker object first (more reliable)
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date, 
                end=end_date, 
                interval='1d',  # Daily data
                auto_adjust=True
            )
            
            if df.empty:
                # Fallback to download method
                df = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    interval='1d',  # Daily data
                    progress=False,
                    auto_adjust=True
                )
            
            if df.empty:
                failed_tickers.append(ticker)
                error_messages[ticker] = "No data returned"
                continue
            
            # Get Close price (yfinance returns Close not Adj Close when auto_adjust=True)
            if 'Close' in df.columns:
                data[ticker] = df['Close']
            else:
                # Fallback to first column if Close not found
                data[ticker] = df.iloc[:, 0]
                
        except Exception as e:
            error_messages[ticker] = str(e)
            failed_tickers.append(ticker)
    
    if failed_tickers:
        error_details = "\n".join([f"  • {t}: {error_messages.get(t, 'Unknown error')}" for t in failed_tickers])
        raise ValueError(f"Failed to fetch data for:\n{error_details}")
    
    if not data:
        raise ValueError("No data was successfully fetched for any ticker")
    
    prices_df = pd.DataFrame(data)
    
    # Only drop rows where ALL values are NaN
    prices_df = prices_df.dropna(how='all')
    
    # Fill forward any remaining NaN values (holidays, etc.)
    prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
    
    if prices_df.empty:
        raise ValueError("All fetched data contained NaN values after cleaning")
    
    return prices_df


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_current_prices(tickers):
    """
    Fetch current prices for given tickers
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        dict: {ticker: current_price}
    """
    prices = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Try multiple methods to get the price
            try:
                # Method 1: info dictionary
                info = stock.info
                price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            except:
                price = None
            
            if not price:
                # Method 2: history (more reliable)
                try:
                    hist = stock.history(period='5d')
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]
                except:
                    pass
            
            if price and not pd.isna(price):
                prices[ticker] = float(price)
            else:
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"Error fetching price for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        raise ValueError(f"Failed to fetch current price for: {', '.join(failed_tickers)}")
    
    return prices


def calculate_portfolio_weights(holdings_df, current_prices):
    """
    Calculate portfolio weights based on current holdings and prices
    
    Args:
        holdings_df: DataFrame with columns ['Ticker', 'Shares']
        current_prices: dict of {ticker: current_price}
        
    Returns:
        tuple: (weights_array, position_values_dict, total_value)
    """
    position_values = {}
    
    for _, row in holdings_df.iterrows():
        ticker = row['Ticker']
        shares = float(row['Shares'])
        if ticker in current_prices:
            position_values[ticker] = shares * current_prices[ticker]
    
    total_value = sum(position_values.values())
    
    if total_value == 0:
        raise ValueError("Total portfolio value is zero")
    
    # Calculate weights in the same order as holdings_df
    weights = []
    for _, row in holdings_df.iterrows():
        ticker = row['Ticker']
        weight = position_values.get(ticker, 0) / total_value
        weights.append(weight)
    
    return np.array(weights), position_values, total_value


def calculate_portfolio_pnl(holdings_df, current_prices):
    """
    Calculate profit/loss for portfolio if cost basis is provided
    
    Args:
        holdings_df: DataFrame with columns ['Ticker', 'Shares', 'Cost Basis (Optional)']
        current_prices: dict of {ticker: current_price}
        
    Returns:
        DataFrame with P&L details or None if no cost basis
    """
    if 'Cost Basis (Optional)' not in holdings_df.columns:
        return None
    
    pnl_data = []
    
    for _, row in holdings_df.iterrows():
        ticker = row['Ticker']
        shares = float(row['Shares'])
        cost_basis = row.get('Cost Basis (Optional)', None)
        
        if pd.isna(cost_basis) or cost_basis == '' or cost_basis == 0:
            continue
            
        cost_basis = float(cost_basis)
        current_price = current_prices.get(ticker, 0)
        
        if current_price == 0:
            continue
        
        cost_value = shares * cost_basis
        current_value = shares * current_price
        pnl = current_value - cost_value
        pnl_pct = (pnl / cost_value * 100) if cost_value != 0 else 0
        
        pnl_data.append({
            'Ticker': ticker,
            'Shares': shares,
            'Cost Basis': cost_basis,
            'Current Price': current_price,
            'Cost Value': cost_value,
            'Current Value': current_value,
            'P&L': pnl,
            'P&L %': pnl_pct
        })
    
    if not pnl_data:
        return None
    
    return pd.DataFrame(pnl_data)


def calculate_returns(prices_df):
    """Calculate daily returns from price data"""
    return prices_df.pct_change().dropna()


def calculate_portfolio_returns(returns_df, weights):
    """
    Calculate portfolio returns given asset returns and weights
    
    Args:
        returns_df: DataFrame of asset returns
        weights: Array of portfolio weights
        
    Returns:
        Series of portfolio returns
    """
    return (returns_df * weights).sum(axis=1)


def calculate_cumulative_returns(returns_series):
    """Calculate cumulative returns from daily returns"""
    return (1 + returns_series).cumprod()


def calculate_annualized_return(returns_series):
    """Calculate annualized return"""
    total_return = (1 + returns_series).prod() - 1
    n_days = len(returns_series)
    years = n_days / 252  # 252 trading days per year
    return (1 + total_return) ** (1 / years) - 1


def calculate_annualized_volatility(returns_series):
    """Calculate annualized volatility"""
    return returns_series.std() * np.sqrt(252)


def calculate_sharpe_ratio(returns_series, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio
    
    Args:
        returns_series: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)
    """
    excess_returns = returns_series - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns_series.std()


def calculate_sortino_ratio(returns_series, risk_free_rate=0.02):
    """
    Calculate Sortino Ratio (uses downside deviation instead of total volatility)
    """
    excess_returns = returns_series - risk_free_rate / 252
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    
    if downside_std == 0:
        return np.nan
    
    return (calculate_annualized_return(returns_series) - risk_free_rate) / downside_std


def calculate_max_drawdown(returns_series):
    """Calculate maximum drawdown"""
    cumulative = calculate_cumulative_returns(returns_series)
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_alpha_beta(portfolio_returns, benchmark_returns):
    """
    Calculate alpha and beta vs benchmark
    
    Returns:
        tuple: (alpha, beta)
    """
    # Align the series
    aligned_data = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned_data) < 2:
        return 0, 0
    
    covariance = aligned_data.cov().iloc[0, 1]
    benchmark_variance = aligned_data['benchmark'].var()
    
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    portfolio_return = calculate_annualized_return(aligned_data['portfolio'])
    benchmark_return = calculate_annualized_return(aligned_data['benchmark'])
    
    alpha = portfolio_return - beta * benchmark_return
    
    return alpha, beta


def calculate_var(returns_series, confidence=0.95):
    """
    Calculate Value at Risk (VaR) at given confidence level
    
    Args:
        returns_series: Series of returns
        confidence: Confidence level (default 95%)
        
    Returns:
        VaR as a positive number (percentage loss)
    """
    return -np.percentile(returns_series, (1 - confidence) * 100)


def calculate_cvar(returns_series, confidence=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) - expected loss beyond VaR
    
    Args:
        returns_series: Series of returns
        confidence: Confidence level (default 95%)
        
    Returns:
        CVaR as a positive number (percentage loss)
    """
    var = calculate_var(returns_series, confidence)
    return -returns_series[returns_series <= -var].mean()


def calculate_correlation_matrix(returns_df):
    """Calculate correlation matrix of asset returns"""
    return returns_df.corr()


def calculate_risk_contribution(returns_df, weights):
    """
    Calculate risk contribution of each asset
    Based on weight * volatility (simplified approach)
    
    Returns:
        Series with risk contribution per asset
    """
    volatilities = returns_df.std() * np.sqrt(252)
    risk_contributions = weights * volatilities
    # Normalize to percentages
    return risk_contributions / risk_contributions.sum()


def get_monthly_returns(returns_series):
    """
    Convert daily returns to monthly returns matrix (Year x Month)
    
    Returns:
        DataFrame with years as rows and months as columns
    """
    # Convert to DataFrame if Series
    if isinstance(returns_series, pd.Series):
        returns_df = returns_series.to_frame('returns')
    else:
        returns_df = returns_series.copy()
    
    # Calculate monthly returns
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df['year'] = returns_df.index.year
    returns_df['month'] = returns_df.index.month
    
    # Group by year and month, calculate compound return
    monthly = returns_df.groupby(['year', 'month']).apply(
        lambda x: (1 + x.iloc[:, 0]).prod() - 1 if isinstance(x.iloc[:, 0], pd.Series) else (1 + x).prod() - 1
    )
    
    # Pivot to matrix format
    monthly_matrix = monthly.unstack(fill_value=0)
    monthly_matrix.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    return monthly_matrix


def stress_test_covid(prices_df, weights):
    """
    Stress test: How would the portfolio perform during COVID crash (Feb-Mar 2020)?
    
    Returns:
        dict with stress test results
    """
    covid_start = '2020-02-01'
    covid_end = '2020-04-01'
    
    # Filter to COVID period
    covid_prices = prices_df.loc[covid_start:covid_end]
    
    if covid_prices.empty:
        return {
            'available': False,
            'message': 'COVID period data not available in dataset'
        }
    
    covid_returns = calculate_returns(covid_prices)
    portfolio_returns = calculate_portfolio_returns(covid_returns, weights)
    cumulative_return = calculate_cumulative_returns(portfolio_returns).iloc[-1] - 1
    max_drawdown = calculate_max_drawdown(portfolio_returns)
    
    return {
        'available': True,
        'start_date': covid_prices.index[0].strftime('%Y-%m-%d'),
        'end_date': covid_prices.index[-1].strftime('%Y-%m-%d'),
        'total_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'worst_day': portfolio_returns.min(),
        'worst_day_date': portfolio_returns.idxmin().strftime('%Y-%m-%d')
    }


def clean_broker_csv(uploaded_file):
    """
    Parse and clean messy broker CSV exports (supports multiple formats)
    
    Handles:
    - Merrill Lynch / Bank of America
    - Fidelity
    - Charles Schwab
    - TD Ameritrade
    - E*TRADE
    - Robinhood
    - Vanguard
    - Interactive Brokers
    - And most other broker formats
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        DataFrame with columns ['Ticker', 'Shares', 'Cost Basis (Optional)']
        or raises ValueError if parsing fails
    """
    import io
    
    try:
        # Read file content into memory
        content = uploaded_file.read()
        
        # Try different encodings
        text = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text = content.decode(encoding)
                break
            except:
                continue
        
        if text is None:
            text = content.decode('utf-8', errors='ignore')
        
        # Split into lines
        lines = text.strip().split('\n')
        
        # Find the header row - look specifically for "symbol" as a column header
        # This should be at the START of the line or after a comma
        header_row_idx = None
        
        for idx, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if line starts with "symbol" (it's the first column)
            if line_lower.startswith('symbol'):
                header_row_idx = idx
                break
            
            # Check if ",symbol" appears (it's not the first column)
            if ',symbol' in line_lower:
                header_row_idx = idx
                break
            
            # Check for "ticker" as column header
            if line_lower.startswith('ticker') or ',ticker' in line_lower:
                header_row_idx = idx
                break
        
        if header_row_idx is None:
            raise ValueError("Could not find header row. Looking for 'Symbol' or 'Ticker' column header.")
        
        # Parse from header row onwards
        csv_content = '\n'.join(lines[header_row_idx:])
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Clean column names (strip whitespace)
        df.columns = [str(col).strip() for col in df.columns]
        
        # Build column mapping (case-insensitive, flexible matching)
        symbol_col = None
        quantity_col = None
        value_col = None
        price_col = None
        day_change_col = None  # Day's Value Change
        gain_loss_col = None   # Unrealized Gain/Loss
        description_col = None # Description
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Symbol/Ticker column (highest priority matches first)
            if symbol_col is None:
                if col_lower in ['symbol', 'ticker', 'sym']:
                    symbol_col = col
                elif 'symbol' in col_lower or 'ticker' in col_lower:
                    symbol_col = col
            
            # Quantity/Shares column
            if quantity_col is None:
                if col_lower in ['quantity', 'shares', 'qty', 'units']:
                    quantity_col = col
                elif any(x in col_lower for x in ['quantity', 'shares', 'qty', 'units']):
                    quantity_col = col
            
            # Value/Market Value column (avoid "Day's Value Change" etc.)
            if value_col is None:
                if col_lower in ['value', 'market value', 'current value', 'total value']:
                    value_col = col
                elif col_lower == 'value' or (col_lower.endswith('value') and 'change' not in col_lower and 'day' not in col_lower):
                    value_col = col
            
            # Price column (backup)
            if price_col is None:
                if col_lower in ['price', 'last price', 'current price']:
                    price_col = col
                elif col_lower == 'price':
                    price_col = col
            
            # Day's Value Change column
            if day_change_col is None:
                if "day" in col_lower and "value" in col_lower and "change" in col_lower:
                    day_change_col = col
            
            # Unrealized Gain/Loss column
            if gain_loss_col is None:
                if "unrealized" in col_lower or "gain" in col_lower or "loss" in col_lower:
                    if "day" not in col_lower:  # Avoid day's change columns
                        gain_loss_col = col
            
            # Description column
            if description_col is None:
                if col_lower in ['description', 'name', 'security', 'security name']:
                    description_col = col
        
        if symbol_col is None:
            raise ValueError(f"Could not find Symbol/Ticker column. Found: {list(df.columns)}")
        
        # Currency parsing function
        def parse_number(value):
            if pd.isna(value):
                return 0.0
            
            value_str = str(value).strip()
            
            # Skip if it's clearly not a number
            if value_str in ['', '-', '--', 'N/A', 'n/a', 'nan', 'NaN']:
                return 0.0
            
            # Handle parentheses (negative values)
            is_negative = '(' in value_str and ')' in value_str
            if is_negative:
                value_str = value_str.replace('(', '').replace(')', '')
            
            # Handle leading minus sign
            if value_str.startswith('-'):
                is_negative = True
                value_str = value_str[1:]
            
            # Remove currency symbols, commas, spaces, and percent signs
            value_str = re.sub(r'[$,€£¥\s%]', '', value_str)
            
            # Remove any trailing non-numeric characters (like "shares")
            value_str = re.sub(r'[a-zA-Z]+$', '', value_str)
            
            try:
                parsed_value = float(value_str)
                return -parsed_value if is_negative else parsed_value
            except:
                return 0.0
        
        # Clean Symbol column
        df['_symbol_clean'] = df[symbol_col].astype(str).str.strip().str.upper()
        
        # Remove common non-stock rows (exact matches or patterns that indicate non-assets)
        exclude_exact = {
            'BALANCE', 'BALANCES', 'TOTAL', 'SUBTOTAL', 'SUMMARY', 
            'N/A', 'NAN', 'NONE', 'NA', '-', '--', 'CASH', 'MONEY'
        }
        
        # Patterns that indicate non-investment accounts
        exclude_contains = [
            'CASH', 'MONEY', 'BALANCE', 'PENDING', 'MARGIN', 'SWEEP', 
            'SETTLEMENT', 'BLACKROCK', 'LIQUIDITY', 'DIRECT DEPOSIT',
            'MONEY MARKET', 'CORE POSITION', 'FDIC', 'INSURED', 'DEPOSIT'
        ]
        
        # Known money market fund symbols to exclude
        money_market_funds = {
            'SPAXX', 'FDRXX', 'VMFXX', 'SPRXX', 'FZFXX', 'SWVXX',  # Fidelity/Schwab/Vanguard
            'VMRXX', 'VMMXX', 'SNSXX', 'SNVXX', 'SWLXX', 'SWGXX',  # More Vanguard/Schwab
            'TTTXX', 'FTEXX', 'FMOXX', 'FMPXX',  # More Fidelity
            'FCASH', 'CORE', 'MMDA', 'MMA'  # Generic money market
        }
        
        # Filter function - only allow actual stock/ETF tickers
        def is_valid_ticker(symbol):
            if not symbol or len(symbol) == 0:
                return False
            # Most stock tickers are 1-5 characters, ETFs up to 5, some up to 6
            if len(symbol) > 6:
                return False
            # Check exact exclusions
            if symbol in exclude_exact:
                return False
            # Check money market funds
            if symbol in money_market_funds:
                return False
            # Check if symbol contains any exclude patterns
            for pattern in exclude_contains:
                if pattern in symbol:
                    return False
            # Must contain at least one letter
            if not any(c.isalpha() for c in symbol):
                return False
            # Valid tickers are alphanumeric (with possible dots/hyphens for classes)
            # Reject anything with spaces
            if ' ' in symbol:
                return False
            return True
        
        # Apply filter
        valid_mask = df['_symbol_clean'].apply(is_valid_ticker)
        clean_df = df[valid_mask].copy()
        
        if clean_df.empty:
            raise ValueError("No valid stock tickers found in CSV")
        
        # Helper to parse combined value+percent strings like "+$10,012.16 +84.25%"
        def parse_gain_loss(value):
            """Parse strings like '+$10,012.16 +84.25%' into (dollar_amount, percent)"""
            if pd.isna(value):
                return None, None
            
            value_str = str(value).strip()
            if value_str in ['', '-', '--', 'N/A', 'n/a', 'nan', 'NaN', '-- --']:
                return None, None
            
            dollar_amount = None
            percent_amount = None
            
            # Try to find dollar amount (with $ sign)
            import re
            # Match patterns like +$10,012.16 or ($544.00) or -$500.00
            dollar_match = re.search(r'[\+\-]?\$[\d,]+\.?\d*|\([\$\d,]+\.?\d*\)', value_str)
            if dollar_match:
                dollar_str = dollar_match.group()
                dollar_amount = parse_number(dollar_str)
            
            # Match percent like +84.25% or -2.42%
            percent_match = re.search(r'[\+\-]?\d+\.?\d*%', value_str)
            if percent_match:
                percent_str = percent_match.group().replace('%', '')
                try:
                    percent_amount = float(percent_str)
                except:
                    percent_amount = None
            
            return dollar_amount, percent_amount
        
        # Extract data
        result_data = []
        
        for _, row in clean_df.iterrows():
            ticker = row['_symbol_clean']
            
            # Get quantity
            shares = 0.0
            if quantity_col and quantity_col in row:
                shares = parse_number(row[quantity_col])
            
            # Get value
            value = 0.0
            if value_col and value_col in row:
                value = parse_number(row[value_col])
            
            # Get price
            price = 0.0
            if price_col and price_col in row:
                price = parse_number(row[price_col])
            
            # Get description
            description = None
            if description_col and description_col in row:
                desc = str(row[description_col]).strip()
                if desc and desc.lower() not in ['nan', 'none', '']:
                    description = desc
            
            # Get day's value change
            day_change_dollar = None
            if day_change_col and day_change_col in row:
                day_change_dollar = parse_number(row[day_change_col])
            
            # Get unrealized gain/loss (dollar and percent)
            gain_loss_dollar = None
            gain_loss_percent = None
            if gain_loss_col and gain_loss_col in row:
                gain_loss_dollar, gain_loss_percent = parse_gain_loss(row[gain_loss_col])
            
            # Skip if both quantity and value are zero/invalid
            if shares <= 0 and value <= 0:
                continue
            
            # If no shares but have value, we'll calculate shares later
            if shares <= 0 and value > 0:
                shares = 1.0  # Placeholder
            
            result_data.append({
                'Ticker': ticker,
                'Shares': shares,
                'Cost Basis (Optional)': None,
                '_original_value': value if value > 0 else None,
                '_price': price if price > 0 else None,
                '_description': description,
                '_day_change': day_change_dollar,
                '_gain_loss_dollar': gain_loss_dollar,
                '_gain_loss_percent': gain_loss_percent
            })
        
        if not result_data:
            raise ValueError("No valid holdings found after processing")
        
        result_df = pd.DataFrame(result_data)
        
        # Remove duplicates (keep first occurrence)
        result_df = result_df.drop_duplicates(subset=['Ticker'], keep='first')
        
        return result_df.reset_index(drop=True)
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")


def calculate_shares_from_values(holdings_df, current_prices):
    """
    Calculate actual shares based on portfolio values and current prices
    
    Args:
        holdings_df: DataFrame with '_original_value' column
        current_prices: dict of current prices
        
    Returns:
        Updated DataFrame with calculated shares
    """
    if '_original_value' not in holdings_df.columns:
        return holdings_df
    
    updated_df = holdings_df.copy()
    
    for idx, row in updated_df.iterrows():
        ticker = row['Ticker']
        if ticker in current_prices and current_prices[ticker] > 0:
            original_value = row['_original_value']
            updated_df.at[idx, 'Shares'] = original_value / current_prices[ticker]
    
    # Remove the temporary column
    updated_df = updated_df.drop(columns=['_original_value'])
    
    return updated_df

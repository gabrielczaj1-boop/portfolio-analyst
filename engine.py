"""
Transaction Engine for Portfolio Analyzer
-----------------------------------------
Handles transaction-ledger based portfolio reconstruction and
point-in-time portfolio state calculations.

Schema (expected):
    Date    : datetime/date-like
    Ticker  : str
    Type    : 'BUY' or 'SELL'
    Quantity: float (>0)
    Price   : float (>0)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class PortfolioState:
    """Point-in-time snapshot of the portfolio."""

    as_of: pd.Timestamp
    positions: pd.DataFrame
    total_value: float
    total_cost: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cash_balance: float


def _normalize_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and validate the transaction ledger.

    Expected columns: Date, Ticker, Type (BUY/SELL), Quantity, Price.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("Transaction ledger is empty.")

    df = raw_df.copy()

    required_cols = ["Date", "Ticker", "Type", "Quantity", "Price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required transaction columns: {', '.join(missing)}")

    # Normalize columns
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", "Type", "Quantity", "Price"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Only BUY / SELL supported
    df = df[df["Type"].isin(["BUY", "SELL"])]

    if df.empty:
        raise ValueError("No valid BUY/SELL transactions after cleaning.")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _compute_positions_and_cost(
    tx: pd.DataFrame, as_of: pd.Timestamp
) -> (Dict[str, float], Dict[str, float]):
    """
    Running-share, running-cost WACB engine per ticker.

    - Buys add shares and cost.
    - Sells remove shares and reduce cost at current average cost.
    - Resulting cost per ticker is cost of remaining shares only.
    """
    tx_cut = tx[tx["Date"] <= as_of]
    if tx_cut.empty:
        return {}, {}

    positions: Dict[str, float] = {}
    costs: Dict[str, float] = {}

    for _, row in tx_cut.iterrows():
        ticker = row["Ticker"]
        qty = float(row["Quantity"])
        price = float(row["Price"])
        side = row["Type"]

        if ticker not in positions:
            positions[ticker] = 0.0
            costs[ticker] = 0.0

        if side == "BUY":
            positions[ticker] += qty
            costs[ticker] += qty * price
        elif side == "SELL":
            if positions[ticker] <= 0:
                # Selling more than held – ignore for position purposes
                continue
            sell_qty = min(qty, positions[ticker])
            # Remove cost at average cost
            avg_cost = costs[ticker] / positions[ticker] if positions[ticker] > 0 else 0.0
            positions[ticker] -= sell_qty
            costs[ticker] -= avg_cost * sell_qty

    # Filter out flat / negative positions
    clean_positions = {t: q for t, q in positions.items() if q > 0}
    clean_costs = {t: costs[t] for t in clean_positions.keys()}
    return clean_positions, clean_costs


def _fetch_prices_for_date(tickers: List[str], as_of: pd.Timestamp) -> Dict[str, float]:
    """
    Fetch adjusted close prices for given tickers as of a specific date.

    Uses a small window around the target date and forward-fills.
    """
    if not tickers:
        return {}

    end = as_of + timedelta(days=1)
    start = as_of - timedelta(days=10)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError("No historical price data returned for transactions universe.")

    # yfinance returns multi-index columns when multiple tickers are requested
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    close = close.sort_index().ffill().bfill()

    if as_of not in close.index:
        # Use the latest available before as_of
        valid_idx = close.index[close.index <= as_of]
        if len(valid_idx) == 0:
            raise ValueError(f"No price data available on/before {as_of.date()}.")
        target_idx = valid_idx[-1]
    else:
        target_idx = as_of

    prices_at_date = close.loc[target_idx]
    if isinstance(prices_at_date, pd.Series):
        return {t: float(prices_at_date.get(t, np.nan)) for t in tickers}
    else:
        # Single ticker case
        return {tickers[0]: float(prices_at_date)}


def _compute_cash_balance(tx: pd.DataFrame, as_of: pd.Timestamp, initial_cash: float) -> float:
    """
    Compute cash balance from transactions up to as_of.

    BUY  -> cash outflow  (negative)
    SELL -> cash inflow   (positive)
    """
    tx_cut = tx[tx["Date"] <= as_of]
    if tx_cut.empty:
        return float(initial_cash)

    cash_flows = np.where(
        tx_cut["Type"] == "BUY",
        -tx_cut["Quantity"] * tx_cut["Price"],
        tx_cut["Quantity"] * tx_cut["Price"],
    )
    return float(initial_cash + cash_flows.sum())


def get_portfolio_state(
    transactions: pd.DataFrame,
    as_of: Optional[date | datetime | pd.Timestamp] = None,
    initial_cash: float = 0.0,
) -> PortfolioState:
    """
    Reconstruct the portfolio state at a specific point in time.

    Steps:
    - Normalize transaction ledger.
    - Filter all transactions <= as_of.
    - Compute cumulative shares and WACB per ticker.
    - Fetch adjusted historical prices for that date.
    - Compute unrealized P&L and aggregate metrics.
    - Compute cash balance from transaction cash flows.
    """
    tx = _normalize_transactions(transactions)

    if as_of is None:
        as_of_ts = tx["Date"].max()
    else:
        as_of_ts = pd.to_datetime(as_of).normalize()

    positions, costs = _compute_positions_and_cost(tx, as_of_ts)

    if not positions:
        # No open positions at this date
        cash_balance = _compute_cash_balance(tx, as_of_ts, initial_cash)
        empty_df = pd.DataFrame(
            columns=[
                "Ticker",
                "Shares",
                "Price",
                "Market Value",
                "Cost Basis",
                "Cost Value",
                "Unrealized P&L",
                "Unrealized P&L %",
            ]
        )
        return PortfolioState(
            as_of=as_of_ts,
            positions=empty_df,
            total_value=0.0,
            total_cost=0.0,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            cash_balance=cash_balance,
        )

    tickers = list(positions.keys())
    prices = _fetch_prices_for_date(tickers, as_of_ts)

    rows = []
    total_value = 0.0
    total_cost = 0.0
    total_unrealized = 0.0

    for ticker in tickers:
        shares = float(positions[ticker])
        cost_value = float(costs.get(ticker, 0.0))
        cost_basis = cost_value / shares if shares > 0 else 0.0
        price = float(prices.get(ticker, np.nan))

        market_value = shares * price if not np.isnan(price) else 0.0
        unrealized = market_value - cost_value
        unrealized_pct = (unrealized / cost_value * 100.0) if cost_value > 0 else 0.0

        total_value += market_value
        total_cost += cost_value
        total_unrealized += unrealized

        rows.append(
            {
                "Ticker": ticker,
                "Shares": shares,
                "Price": price,
                "Market Value": market_value,
                "Cost Basis": cost_basis,
                "Cost Value": cost_value,
                "Unrealized P&L": unrealized,
                "Unrealized P&L %": unrealized_pct,
            }
        )

    positions_df = pd.DataFrame(rows).sort_values("Market Value", ascending=False)

    unrealized_pct_total = (
        (total_unrealized / total_cost * 100.0) if total_cost > 0 else 0.0
    )
    cash_balance = _compute_cash_balance(tx, as_of_ts, initial_cash)

    return PortfolioState(
        as_of=as_of_ts,
        positions=positions_df,
        total_value=total_value,
        total_cost=total_cost,
        unrealized_pnl=total_unrealized,
        unrealized_pnl_pct=unrealized_pct_total,
        cash_balance=cash_balance,
    )


def load_transactions_csv(uploaded_file) -> pd.DataFrame:
    """
    Simple CSV loader for transaction ledger input from Streamlit's file_uploader.

    Expects columns: Date, Ticker, Type, Quantity, Price.
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    df = pd.read_csv(uploaded_file)
    return _normalize_transactions(df)


# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Quant Finance Skills Module.

Reusable analytics helpers, importable directly inside a notebook cell:

    from quant_skills import RiskAnalytics, AlphaFactors, VectorizedBacktest
"""

import numpy as np
import pandas as pd


class RiskAnalytics:
    """Methods for evaluating quantitative portfolio risk metrics."""

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04, periods: int = 252) -> float:
        """Calculates the annualized Sharpe Ratio."""
        rf_daily = (1 + risk_free_rate) ** (1 / periods) - 1
        excess_returns = returns - rf_daily
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(periods) * (excess_returns.mean() / excess_returns.std())

    @staticmethod
    def max_drawdown(prices: pd.Series) -> float:
        """Calculates Maximum Drawdown (peak-to-trough decline)."""
        cumulative_max = prices.cummax()
        drawdown = (prices - cumulative_max) / cumulative_max
        return drawdown.min()

    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculates Historical Value at Risk (VaR)."""
        return np.percentile(returns.dropna(), (1 - confidence_level) * 100)


class AlphaFactors:
    """Methods for computing quantitative signals."""

    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculates Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculates MACD line and Signal line."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line


class VectorizedBacktest:
    """Simplified vectorized strategy backtester."""

    @staticmethod
    def run_sma_crossover(prices: pd.Series, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
        """Simple Moving Average (SMA) Crossover Backtest."""
        df = pd.DataFrame({"Close": prices})
        df["SMA_Short"] = df["Close"].rolling(window=short_window).mean()
        df["SMA_Long"] = df["Close"].rolling(window=long_window).mean()

        signal = pd.Series(0.0, index=df.index)
        signal.iloc[short_window:] = np.where(
            df["SMA_Short"].iloc[short_window:] > df["SMA_Long"].iloc[short_window:], 1.0, 0.0
        )
        df["Signal"] = signal
        df["Position"] = df["Signal"].shift(1)

        df["Asset_Returns"] = df["Close"].pct_change()
        df["Strategy_Returns"] = df["Asset_Returns"] * df["Position"]
        df["Cumulative_Strategy"] = (1 + df["Strategy_Returns"].fillna(0)).cumprod()
        df["Cumulative_Asset"] = (1 + df["Asset_Returns"].fillna(0)).cumprod()

        return df

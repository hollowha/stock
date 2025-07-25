from math import sqrt

import numpy as np
import pandas as pd

def sharpe_ratio(
    expected_return: float,
    volatility: float,
    risk_free: float = 0.0,
) -> float:
    """
    Computes the sharpe ratio.

    parameter
    ---------
    expected_return: float
        Daily expected return of a portfolio.
    volatility: float
        Daily volatility of a portfolio.
    risk_free: float
        Daily risk free rate.

    return: float
        Returns the annual sharpe ratio.
    """
    return (expected_return - risk_free) / volatility

def sortino_ratio(
    expected_return: float,
    downside_risk: float,
    risk_free: float = 0.0,
) -> float:
    """
    Computes the sortino ratio.

    parameter
    ---------
    expected_return: float
        Daily expected return of a portfolio.
    downside_risk: float
        Daily downside risk of a portfolio.
    risk_free: float
        Daily risk free rate.

    return: float
        Returns the annual sortino ratio.
    """
    return (expected_return - risk_free) / downside_risk

def calmar_ratio(
    expected_return: float, max_drawdown: float
) -> float:
    """
    Computes the calmar ratio.

    parameter
    ---------
    expected_return: float
        Daily expected return of a portfolio.
    max_drawdown: float
        Maximum drawdown of a portfolio.

    return: float
        Returns the annual calmar ratio.
    """
    return expected_return / max_drawdown

def downside_risk(
    daily_return: pd.Series, risk_free: float = 0.0
) -> float:
    """
    Computes the downside risk (standard deviation of negative returns).

    parameter
    ---------
    daily_return: pd.Series
        Daily return of a portfolio.
    risk_free: float
        Daily risk free rate.
    trading_days: float
        Number of trading days in a year.

    return: float
        Returns the annual downside risk.
    """
    return daily_return[daily_return < risk_free].std()

def max_drawdown(daily_return: pd.Series) -> float:
    """
    Computes the maximum drawdown of a portfolio.

    parameter
    ---------
    daily_return: pd.Series
        Daily return of a portfolio.

    return: float
        Returns the maximum drawdown percentage.
    """
    # 1. Compute cummulative return.
    balance = (1.0 + daily_return).cumprod()
    # 2. Compute cummulative maximum.
    peak = balance.cummax()
    # 3. Compute drawdown.
    drawdown = (peak - balance) / peak
    return float(drawdown.max())
import matplotlib.axes
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional
from math import sqrt

from finolio.types import ASSET_TYPE
from finolio.metrics import *

class Asset:
    daily_return: pd.Series
    trading_days: float
    name: Optional[str]
    asset_type: ASSET_TYPE
    risk_free: float
    _expected_return: float
    _max_drawdown: float
    _volatility: float
    
    def __init__(
        self,
        daily_return: pd.Series,
        trading_days: Optional[float] = None,
        risk_free: float = 0.0,
        name: Optional[str] = None,
        asset_type: ASSET_TYPE = ASSET_TYPE.NONE
    ) -> None:
        """
        Parameters
        ----------
        daily_return: pd.Series
            Historical daily returns for the asset.
        name: str, optional
            Name of the asset.
        asset_type: ASSET_TYPE
            Type of the asset.
        """
        self.daily_return = daily_return.astype(np.float64)
        self.risk_free = risk_free
        self.name = name
        self.asset_type = asset_type

        if trading_days is not None:
            self.trading_days = trading_days
        else:
            if self.asset_type == ASSET_TYPE.STOCK:
                self.trading_days = 252
            elif self.asset_type == ASSET_TYPE.MARKET_INDEX:
                self.trading_days = 252
            elif self.asset_type == ASSET_TYPE.CRYPTO:
                self.trading_days = 365.26
            elif self.asset_type == ASSET_TYPE.NONE:
                self.trading_days = 365.26
            else:
                raise TypeError(f"Incorrect type for Asset.asset_type. Expected: finolio.types.ASSET_TYPE. Found: {type(self.asset_type).__name__}.")

        self._compute_metrics()

    def _compute_metrics(self):
        # Annual volatility
        self._volatility = self.daily_return.std() * sqrt(self.trading_days)
        # Annual expected return
        self._expected_return = self.daily_return.mean() * self.trading_days
        # Max drawdown
        self._max_drawdown = max_drawdown(daily_return=self.daily_return)

    @property
    def max_drawdown(self):
        return self._max_drawdown

    @property
    def volatility(self):
        return self._volatility

    @property
    def expected_return(self):
        return self._expected_return

    @property
    def sharpe_ratio(self):
        return sharpe_ratio(self._expected_return, self._volatility, self.risk_free)

    @property
    def sortino_ratio(self):
        daily_risk_free = self.risk_free / self.trading_days
        daily_downside_risk = downside_risk(self.daily_return, daily_risk_free) * sqrt(self.trading_days)
        return sortino_ratio(self._expected_return, daily_downside_risk, self.risk_free)

    @property
    def calmar_ratio(self):
        return calmar_ratio(self._expected_return, self._max_drawdown)

    @property
    def cumulative_return(self):
        """
        Returns the cumulative return of each day.
        """
        return (1.0 + self.daily_return).cumprod()

    def plot_series(
        self,
        ax: matplotlib.axes.Axes = None,
        alpha: float = 1.0,
        show_label: bool = True,
        balance: float = 1,
        fee_ratio: float = 0.0,
    ) -> matplotlib.axes.Axes:
        """
        This function plot the cumulative return on the given axes.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes to plot cumulative returns on.
        alpha: float
            The alpha of the line. Default is 1.0.
        balance: float
            Initial balance.

        return:
            The resulting axes after plotting the cumulative return.
        """
        balances = self.cumulative_return * balance
        fees = balances * fee_ratio
        balances = balances - fees
        label = (self.name if show_label else "")
        ax.plot(balances.index, balances, alpha=alpha, label=label)
        return ax

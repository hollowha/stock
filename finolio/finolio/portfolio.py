import numpy as np
import pandas as pd
import matplotlib
import scipy
from typing import Optional, List
from math import sqrt

from finolio.types import ASSET_TYPE
from finolio.asset import Asset
from finolio.metrics import *

class Portfolio(Asset):
    asset_daily_returns_df: pd.DataFrame
    weights: pd.Series
    weighted_daily_returns_df: pd.Series
    _cov_matrix_df: pd.DataFrame

    def __init__(
        self,
        daily_returns: pd.DataFrame,
        weights: List[float],
        trading_days: float = 252,
        risk_free: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        asset_daily_returns_df: pd.DataFrame
            A 2d dataframe, where each row is the daily percentage change, and column is the asset.
        weights: List[float]
            Percentage of each asset in the portfolio.
        trading_days: float
            Number of trading days in a year.
        risk_free: float
            Annual risk free rate.
        name: str, optional
            Name of the portfolio.
        """
        self.asset_daily_returns_df = daily_returns
        # Covariance matrix of each asset.
        self._cov_matrix_df = self.asset_daily_returns_df.cov() * trading_days
        self.update_weights(weights, update_metrics=False)
        super().__init__(daily_return=self.weighted_daily_returns_df,
                         trading_days=trading_days,
                         risk_free=risk_free,
                         name=name,
                         asset_type=ASSET_TYPE.MARKET_INDEX)

    def update_weights(self, weights: List[float], update_metrics: bool = True):
        """
        Update the weights for each asset.

        Parameters
        ----------
        weights: List[float]
            Percentage of each asset in the portfolio.
        update_metrics: bool
            Update the metrics such as expected_return, volatility, and max_drawdown.
        """
        self.weights = pd.Series(weights)
        # Compute the combined weighted daily return of the full portfolio.
        self.weighted_daily_returns_df = self.asset_daily_returns_df.dot(self.weights.values.reshape(-1, 1)).squeeze()
        if update_metrics:
            super().__init__(daily_return=self.weighted_daily_returns_df,
                             trading_days=self.trading_days,
                             risk_free=self.risk_free,
                             name=self.name,
                             asset_type=self.asset_type)

    def mean_variance_optimization(self) -> List[float]:
        """
        Find the weights that minimize mean-variance in Harry Markowitz's Modern Portfolio Theory.

        return:
            The optimized weights for each asset.
        """
        num_assets = self.asset_daily_returns_df.shape[1]
        # Constraints
        constraints = ({
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        })
        # Bounds
        lower_bound = 0
        upper_bound = 1
        bounds = tuple((lower_bound, upper_bound) for _ in range(num_assets))
        # Equal initial weights.
        initial_weights = np.ones(num_assets) / num_assets
        result = scipy.optimize.minimize(lambda weights: weights.T @ self._cov_matrix @ weights,
                                         initial_weights,
                                         method="SLSQP",
                                         bounds=bounds,
                                         constraints=constraints)
        if result.success:
            return result.x.tolist()
        raise BaseException(result.message)

    @property
    def cov_matrix(self) -> pd.DataFrame:
        return self._cov_matrix_df

    def plot_series(
        self,
        ax: matplotlib.axes.Axes = None,
        plot_child: bool = True,
        balance: float = 1,
        fee_ratio: float = 0.0,
    ) -> matplotlib.axes.Axes:
        """
        This function plot the cumulative return on the given axes.
        Alpha of each asset line is determined by the weights in the portfolio.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes to plot all cumulative returns on.
        plot_child: bool
            Plot each asset in the portfolio or not.
        balance: float
            Initial balance.

        return:
            The resulting axes after plotting the cumulative returns.
        """
        if plot_child:
            for i, asset_name in enumerate(self.asset_daily_returns_df.columns):
                Asset(daily_return=self.asset_daily_returns_df[asset_name],
                      name=asset_name
                ).plot_series(ax=ax, balance=balance, alpha=abs(self.weights[i]))
        super().plot_series(ax=ax, balance=balance, fee_ratio=fee_ratio)
        return ax

    def plot_pie(
        self,
        ax: matplotlib.axes.Axes = None,
    ) -> matplotlib.axes.Axes:
        """
        Plot the pie graph by the weights of each asset.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes to plot all cumulative returns on.

        return:
            The resulting axes after plotting the cumulative returns.
        """
        ax.pie(
            self.weights,
            labels=self.asset_daily_returns_df.columns,
            autopct="%1.1f%%",
            counterclock=False,
            textprops={"fontsize": 8}
        )
        return ax
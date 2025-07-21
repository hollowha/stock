import pandas as pd
import torch
import numpy as np


def load_original_prices_aligned(csv_path: str, asset_list: list) -> torch.Tensor:
    """
    Load original prices from a CSV file, aligned with the given asset list order.
    
    :param csv_path: Path to the CSV file. The first column is 'date' and will be ignored.
    :param asset_list: List of asset names in the order matching the 'data' tensor.
    :return: torch.Tensor of shape [num_assets], aligned with asset_list
    """
    df = pd.read_csv(csv_path)
    
    if df.shape[0] != 1:
        raise ValueError(f"CSV should contain only one row of price data, but got {df.shape[0]} rows.")

    df = df.set_index('date')  # Remove the 'date' column by setting it as the index
    
    try:
        prices = df.loc[:, asset_list].iloc[0].values.astype(float)
    except KeyError as e:
        raise ValueError(f"Some asset names in asset_list are not found in the CSV: {e}")

    return torch.tensor(prices, dtype=torch.float32)



def infer_price_by_index(final_prices: torch.Tensor, data, index: int) -> torch.Tensor:
    """
    Back-calculate the price of each asset at a specific index using the final prices and daily return data.

    Accepts either a torch.Tensor or pandas.DataFrame for data.
    """
    if isinstance(data, pd.DataFrame):
        data = torch.tensor(data.values, dtype=torch.float32)

    if index < 0 or index >= data.shape[0]:
        raise ValueError(f"Index {index} is out of range for data with {data.shape[0]} rows.")

    price = final_prices.clone()
    for t in reversed(range(index, data.shape[0])):
        price = price / (1 + data[t])

    return price
    
def get_final_prices(original_prices, data: torch.Tensor) -> torch.Tensor:
    """
    Calculate final prices by applying daily returns to original prices.

    :param original_prices: torch.Tensor or array-like, shape [num_assets]
    :param data: torch.Tensor, shape [num_days, num_assets], daily returns
    :return: torch.Tensor, final prices after applying all returns
    """
    if isinstance(original_prices, torch.Tensor):
        prices = original_prices.clone().detach()
    else:
        prices = torch.tensor(original_prices, dtype=torch.float32)

    for daily_return in data:
        prices = prices * (1 + daily_return)

    return prices


def calculate_portfolio_value(stocks, new_prices) -> float:
    """
    Calculate the total portfolio value given number of shares and current prices.

    :param stocks: List, array, or tensor of shares per asset (fractional allowed)
    :param new_prices: List, array, or tensor of current prices per asset
    :return: Total portfolio value as float
    """
    stocks = np.array(stocks, dtype=np.float32)
    prices = np.array(new_prices, dtype=np.float32)
    return float(np.sum(stocks * prices))
    
    

def convert_to_stocks(original_balance: float, weights, original_prices) -> np.ndarray:
    """
    Convert investment weights into number of shares (fractional allowed).
    
    :param original_balance: Total investment capital
    :param weights: List, array, or tensor of portfolio weights per asset
    :param original_prices: List, array, or tensor of asset prices on the investment day
    :return: Array of shares (fractional shares allowed) per asset
    """
    weights = np.array(weights, dtype=np.float32)
    prices = np.array(original_prices, dtype=np.float32)
    allocated_cash = original_balance * weights
    stocks = allocated_cash / prices
    return stocks


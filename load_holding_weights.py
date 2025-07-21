import pandas as pd
import torch


def load_holding_weights(csv_path, assets, device="cpu"):
    """
    Reads the holding weights from a CSV file and outputs a tensor sorted according to the provided assets.
    If the CSV contains indices not in the provided assets list, these extra indices will be appended at the end.
    For indices that overlap with assets, the order will follow the order in assets.

    Parameters:
        csv_path: Path to the holding weights CSV file.
        assets: A sorted list of asset names.
        device: The device on which the tensor will be allocated.

    Returns:
        holding_weights: A tensor of shape (1, num_assets + extra) that can be compared element-wise with the model's weights.
    """
    
    # Read the CSV without a header, using the first column as the index.
    df = pd.read_csv(csv_path, header=None, index_col=0)
    df.index = df.index.astype(str)

    # Reindex for those assets present in the provided list (preserving order from assets).
    df_assets = df.reindex(assets, fill_value=0)

    # Identify extra indices that are in the CSV but not in the assets list.
    extra_index = [idx for idx in df.index if idx not in assets]
    df_extra = df.loc[extra_index]

    # Concatenate the two parts: first the assets (in order), then the extra indices.
    df_new = pd.concat([df_assets, df_extra])

    # Convert the weights (first column) to a float array and then to a tensor,
    # adding a batch dimension so that the tensor shape is (1, total_num_assets)
    weights_array = df_new.iloc[:, 0].values.astype(float)
    holding_weights = torch.tensor(weights_array, dtype=torch.float, device=device).unsqueeze(0)

    return holding_weights


def load_prices(csv_path, device="cpu"):
    """
    Loads the latest prices from a CSV file.
    Assumes the CSV format:
        date, stock1, stock2, ...
        OOO , price1, price2, ...

    Parameters:
        csv_path: Path to the price CSV file.
        device: The device on which the tensor will be allocated.

    Returns:
        stock_names: List of stock names.
        prices: Tensor of shape (1, num_stocks).
    """
    df = pd.read_csv(csv_path, header=0)

    # First row are prices
    prices_array = df.iloc[0, 1:].values.astype(float)
    stock_names = df.columns[1:].tolist()

    prices = torch.tensor(prices_array, dtype=torch.float, device=device).unsqueeze(0)

    return stock_names, prices
    
def load_holding(csv_path, device="cpu"):
    """
    Loads holding information from a CSV file.
    Assumes the CSV format (no header):
        stock, holding_shares, average_cost

    Parameters:
        csv_path: Path to the holding CSV file.
        device: The device on which the tensors will be allocated.

    Returns:
        stock_names: List of stock names.
        holdings: Tensor of shape (1, num_stocks) representing share counts.
        avg_costs: Tensor of shape (1, num_stocks) representing average costs.
    """
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['stock', 'holding', 'avg_cost']

    stock_names = df['stock'].astype(str).tolist()
    holdings_array = df['holding'].values.astype(float)
    avg_costs_array = df['avg_cost'].values.astype(float)

    holdings = torch.tensor(holdings_array, dtype=torch.float, device=device).unsqueeze(0)
    avg_costs = torch.tensor(avg_costs_array, dtype=torch.float, device=device).unsqueeze(0)

    return stock_names, holdings, avg_costs

def calculate_total_value(holdings, prices, holding_stocks, price_stocks):
    """
    Calculates the total market value of the holdings.

    Assumes holdings and prices are (1, num_stocks) tensors.
    Stocks must be matched by name.

    Parameters:
        holdings: Tensor of holding share counts (1, num_holding_stocks).
        prices: Tensor of stock prices (1, num_price_stocks).
        holding_stocks: List of stock names for holdings.
        price_stocks: List of stock names for prices.

    Returns:
        total_value: Float, total market value of holdings.
    """
    # Create stock name to price mapping
    price_dict = dict(zip(price_stocks, prices.squeeze(0).tolist()))
    
    total_value = 0.0
    for i, stock in enumerate(holding_stocks):
        if stock not in price_dict:
            raise ValueError(f"Price for stock {stock} not found in price list.")
        stock_price = price_dict[stock]
        total_value += holdings[0, i].item() * stock_price

    return total_value

def filter_underwater_stocks(holdings, avg_costs, prices, holding_stocks, price_stocks):
    """
    Filters out stocks where the current price is lower than the average cost.

    Parameters:
        holdings: Tensor of holding share counts (1, num_holding_stocks).
        avg_costs: Tensor of average costs (1, num_holding_stocks).
        prices: Tensor of stock prices (1, num_price_stocks).
        holding_stocks: List of stock names for holdings.
        price_stocks: List of stock names for prices.

    Returns:
        df_underwater: DataFrame containing ['stock', 'avg_cost', 'current_price'] for underwater stocks.
    """
    import pandas as pd

    # Create mapping from stock name to price
    price_dict = dict(zip(price_stocks, prices.squeeze(0).tolist()))
    
    records = []
    for i, stock in enumerate(holding_stocks):
        avg_cost = avg_costs[0, i].item()
        if stock not in price_dict:
            continue  # ignore if no price
        current_price = price_dict[stock]
        if current_price < avg_cost:
            records.append({
                'stock': stock,
                'avg_cost': avg_cost,
                'current_price': current_price
            })

    df_underwater = pd.DataFrame(records)

    return df_underwater


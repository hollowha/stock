import math, pdb, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as cPickle

from datetime import datetime
from dateutil.relativedelta import relativedelta

import torch
import torch.nn as nn

from argparse import ArgumentParser

from load_config import config
from utils import *

from finolio.asset import Asset
from finolio.portfolio import Portfolio
from finolio.types import *

from train import train_and_test

# parser = ArgumentParser(description='Input parameters for Deep Generative Neural Network')
# parser.add_argument('--noise', default=16, type=int, help='Number of Noise Variables for GNN')
# parser.add_argument('--cnndim', default=2, type=int, help='Size of Latent Dimensions for GNN')
# parser.add_argument('--iter', default=50, type=int, help='Number of Total Iterations for Solver')
# parser.add_argument('--batch', default=1024, type=int, help='Number of Evaluations in an Iteration')
# parser.add_argument('--rseed', default=0, type=int, help='Random Seed for Network Initialization')
# parser.add_argument('--cagr', default=0.1, type=float, help='Target CAGR')
# parser.add_argument('--csv', action='store_true', help='Read data from csv')
# args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])

device = torch.device("cpu")

Dataset = pd.read_csv(config["data"]).set_index("date")
Dataset = drop_columns_with_many_nans(Dataset, threshold=0.2)
Dataset = Dataset.dropna()
ETFs = pd.read_csv(config["etf"]).columns
ETFs = [etf for etf in ETFs if etf in Dataset.columns]

selected = [config["index"]]
# while not selected:
#     selected = input("Enter ETFs to track (comma-separate if multiple equal weighted ETFs) e.g. RYH, VGT\n").split()
#     selected = [etf for etf in selected if etf in Dataset.columns]

# If multiple ETF are selected, we create an index with equally weighted.
selected_returns = torch.Tensor(Dataset[selected].values).to(device)
mean_weights = torch.ones(len(selected)).to(device)
mean_weights /= len(selected)
index = (selected_returns * mean_weights).sum(dim=1)

selected_0050 = ["0050"]
# while not selected:
#     selected = input("Enter ETFs to track (comma-separate if multiple equal weighted ETFs) e.g. RYH, VGT\n").split()
#     selected = [etf for etf in selected if etf in Dataset.columns]

# If multiple ETF are selected, we create an index with equally weighted.
selected_0050_returns = torch.Tensor(Dataset[selected_0050].values).to(device)
mean_0050_weights = torch.ones(len(selected_0050)).to(device)
mean_0050_weights /= len(selected_0050)
index_0050 = (selected_0050_returns * mean_0050_weights).sum(dim=1)


Dataset = Dataset.drop(columns=ETFs)
assets = list(Dataset.columns.values)

def split_range(Dataset: pd.DataFrame, split_date_from: str, split_date_to: str):
    """
    Returns data: torch.tensor, index_: torch.tensor, dates: np.array, before split_date
    """
    previous_date = Dataset.index[Dataset.index < split_date_from].max()
    if pd.isnull(previous_date):
        previous_date = split_date_from
    
    mask = (Dataset.index >= previous_date) & (Dataset.index < split_date_to)
    data = Dataset.loc[mask]
    data = np.array(data)
    data = torch.from_numpy(data).float()

    dates = Dataset.loc[mask].index
    
    
    index_start = len(Dataset.loc[Dataset.index < previous_date])
    index_end = len(Dataset.loc[Dataset.index < split_date_to])
    index_ = index[index_start:index_end]
    index_ = np.array(index_)
    index_ = torch.from_numpy(index_).float()

    assert data.shape[0] == index_.shape[0]
    return data, index_, dates

fig, ax = plt.subplots(1, 1, figsize=(18, 12))

balance = 78000

# Read from config
delay_interval_months = config["delay_interval_months"]
testing_interval_months = config["testing_interval_months"]
fee = config["fee"]
start_trading_weight = config["start_trading_weight"]



# Loops over every two months
start_date = datetime.strptime(Dataset.index[0], "%Y-%m-%d")
end_date = datetime.strptime(Dataset.index[-1], "%Y-%m-%d")
current_date = start_date + relativedelta(months=delay_interval_months)
start_date_str = str(start_date).split()[0]

delay_indices = 0


# Plot 0050 and index first
tmp, _, _ = split_range(Dataset, str(start_date).split()[0], str(start_date + relativedelta(months=delay_interval_months)).split()[0])
print(tmp.shape)
skipped_days = tmp.shape[0]-1

index_0050_asset = Asset(daily_return=pd.Series(index_0050[skipped_days:], index=pd.to_datetime(Dataset.index[skipped_days:])),
                    name="0050",
                    asset_type=ASSET_TYPE.MARKET_INDEX)
index_0050_balance = 100000/index_0050_asset.cumulative_return[0]
ax = index_0050_asset.plot_series(ax=ax, balance=index_0050_balance, alpha=1)

'''
index_asset = Asset(daily_return=pd.Series(index[skipped_days:], index=pd.to_datetime(Dataset.index[skipped_days:])),
                    name=config["index"],
                    asset_type=ASSET_TYPE.MARKET_INDEX)
index_balance = 100000/index_asset.cumulative_return[0]                    
ax = index_asset.plot_series(ax=ax, balance=index_balance, alpha=1)
'''


#while current_date + relativedelta(months=testing_interval_months) <= end_date:
while current_date <= end_date:
    split_from = str(current_date).split()[0]
    split_to = str(current_date + relativedelta(months=testing_interval_months)).split()[0]

    print(f"Train {start_date_str} ~ {split_from} | Test {split_from} ~ {split_to}")

    train_data, train_index, train_dates = split_range(Dataset, start_date_str, split_from)
    test_data, test_index, test_dates = split_range(Dataset, split_from, split_to)
    
    first_test = False

    if delay_indices == 0:
        delay_indices = train_data.shape[0]
        first_test = True
        

    # Handle outliers
    q3 = train_data.abs().quantile(0.75)
    q3 =  q3 + 1.5 * (q3 - train_data.abs().quantile(0.25))
    train_data[train_data > q3] = q3
    train_data[train_data < -q3] = -q3
    test_data[test_data > q3] = q3
    test_data[test_data < -q3] = -q3

    pf = train_and_test(train_data=train_data,
                        train_index=train_index,
                        train_dates=train_dates,
                        test_data=test_data,
                        test_index=test_index,
                        test_dates=test_dates,
                        assets=assets,
                        config=config,
                        device=device,
                        delay_indices=delay_indices,
                        start_trading_weight=start_trading_weight)
                        
    if(first_test):
      balance = 100000/pf.cumulative_return[0]

    # ax = pf.plot_series(ax=ax, plot_child=False, balance=balance, fee_ratio=fee)
    balances = (pf.cumulative_return * balance -100000)*0.78+100000
    fees = balances * fee
    balances = balances - fees
    ax.plot(pd.to_datetime(test_dates), balances, alpha=1, label="Ours")
    
    balance *= pf.cumulative_return[-2]
    balance -= balance * fee
    
 
    current_date += relativedelta(months=testing_interval_months)

"""
# Split data into training and testing.
train_data = Dataset.loc[Dataset.index < config["split_date"]]
test_data = Dataset.loc[Dataset.index >= config["split_date"]]
train_data = np.array(train_data)
train_data = torch.from_numpy(train_data).float()
test_data = np.array(test_data)
test_data = torch.from_numpy(test_data).float()

train_index = index[:train_data.shape[0]]
test_index = index[train_data.shape[0]:]

train_dates = Dataset.loc[Dataset.index < config["split_date"]].index
test_dates = Dataset.loc[Dataset.index >= config["split_date"]].index

assert train_data.shape[0] == train_index.shape[0]
assert test_data.shape[0] == test_index.shape[0]

# Handle outliers
q3 = train_data.abs().quantile(0.75)
q3 =  q3 + 1.5 * (q3 - train_data.abs().quantile(0.25))
train_data[train_data > q3] = q3
train_data[train_data < -q3] = -q3
test_data[test_data > q3] = q3
test_data[test_data < -q3] = -q3

print(train_data.max(), train_data.min())
print(test_data.max(), test_data.min())
"""




title = f"delay interval: {delay_interval_months} months; testing interval: {testing_interval_months} months; fee: {fee * 100:.5f}%"
ax.set_title(title)
ax.set_xlabel("Dates")
ax.set_ylabel("$")
ax.grid(True)
output_file = config["output"]
plt.savefig(output_file, bbox_inches="tight", pad_inches=0)

print(f"Result saved to {output_file}")

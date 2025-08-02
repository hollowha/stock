import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as cPickle
import torch
import torch.nn as nn
from datetime import datetime
from dateutil.relativedelta import relativedelta
from argparse import ArgumentParser
from load_config import config
from utils import *
from finolio.asset import Asset
from finolio.portfolio import Portfolio
from finolio.types import *
from train import train_and_predict



# Set deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config["seed"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
Dataset = pd.read_csv(config["data"]).set_index("date")
Dataset = drop_columns_with_many_nans(Dataset, threshold=0.2)
Dataset = Dataset.dropna()
ETFs = pd.read_csv(config["etf"]).columns
ETFs = [etf for etf in ETFs if etf in Dataset.columns]

selected = [config["index"]]
selected_returns = torch.Tensor(Dataset[selected].values).to(device)
mean_weights = torch.ones(len(selected)).to(device)
mean_weights /= len(selected)
index = (selected_returns * mean_weights).sum(dim=1)

selected_0050 = ["0050"]
selected_0050_returns = torch.Tensor(Dataset[selected_0050].values).to(device)
mean_0050_weights = torch.ones(len(selected_0050)).to(device)
mean_0050_weights /= len(selected_0050)
index_0050 = (selected_0050_returns * mean_0050_weights).sum(dim=1)

Dataset = Dataset.drop(columns=ETFs)
assets = list(Dataset.columns.values)

def split_range(Dataset: pd.DataFrame):
    """
    Returns data: torch.tensor, index_: torch.tensor, dates: np.array, before split_date
    """

    data = np.array(Dataset)
    data = torch.from_numpy(data).float().to(device)

    index_ = np.array(index)
    index_ = torch.from_numpy(index_).float().to(device)

    dates = Dataset.index
    assert data.shape[0] == index_.shape[0]
    return data, index_, dates

train_data, train_index, train_dates = split_range(Dataset)

# Handle outliers
q3 = train_data.abs().quantile(0.75)
q3 =  q3 + 1.5 * (q3 - train_data.abs().quantile(0.25))
train_data[train_data > q3] = q3
train_data[train_data < -q3] = -q3

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
# Train the model and predict
pf = train_and_predict(train_data=train_data,
                  train_dates=train_dates,
                  train_index=train_index,
                  assets=assets,
                  config=config,
                  device=device)
                  
ax = pf.plot_series(ax=ax, balance=100000, plot_child=False)           
                  
index_0050_asset = Asset(daily_return=pd.Series(index_0050, index=pd.to_datetime(train_dates)),
                    name="0050",
                    asset_type=ASSET_TYPE.MARKET_INDEX)
ax = index_0050_asset.plot_series(ax=ax, balance=100000, alpha=1)

index_asset = Asset(daily_return=pd.Series(index, index=pd.to_datetime(train_dates)),
                    name=config["index"],
                    asset_type=ASSET_TYPE.MARKET_INDEX)
ax = index_asset.plot_series(ax=ax, balance=100000, alpha=1)

title = f"Only training"
ax.set_title(title)
ax.set_xlabel("Dates")
ax.set_ylabel("$")
ax.grid(True)
output_file = config["output"]
plt.savefig(output_file, bbox_inches="tight", pad_inches=0)

print(f"Result saved to {output_file}")                  



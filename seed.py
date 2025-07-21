import math
import numpy as np
import pandas as pd
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
from train import train_and_val

'''
# Set deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])
'''

device = torch.device("cpu")

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

Dataset = Dataset.drop(columns=ETFs)
assets = list(Dataset.columns.values)

def split_range(Dataset: pd.DataFrame, split_date_from: str, split_date_to: str):
    """
    Splits the dataset into train/validation based on the date range.
    """
    mask = (Dataset.index >= split_date_from) & (Dataset.index < split_date_to)
    data = Dataset.loc[mask]
    data = np.array(data)
    data = torch.from_numpy(data).float()

    index_start = len(Dataset.loc[Dataset.index < split_date_from])
    index_end = len(Dataset.loc[Dataset.index < split_date_to])
    index_ = index[index_start:index_end]
    index_ = np.array(index_)
    index_ = torch.from_numpy(index_).float()

    dates = Dataset.loc[mask].index
    assert data.shape[0] == index_.shape[0]
    return data, index_, dates

# Calculate split dates
delay_interval_months = config["delay_interval_months"]
testing_interval_months = config["testing_interval_months"]

start_date = datetime.strptime(Dataset.index[0], "%Y-%m-%d")
train_end_date = start_date + relativedelta(months=delay_interval_months)
val_end_date = train_end_date + relativedelta(months=testing_interval_months)

train_start_date = str(start_date).split()[0]
train_end_date = str(train_end_date).split()[0]
val_end_date = str(val_end_date).split()[0]

# Loop over random seeds
num_loops = config["seed"]
validation_results = []

for i in range(num_loops):
    random_seed = np.random.randint(0, 10000)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    print(f"Iteration {i+1}/{num_loops}: Random seed = {random_seed}")
    print(f"Training date range: {train_start_date} to {train_end_date}")
    print(f"Validation date range: {train_end_date} to {val_end_date}")

    train_data, train_index, train_dates = split_range(Dataset, train_start_date, train_end_date)
    val_data, val_index, val_dates = split_range(Dataset, train_end_date, val_end_date)

    # Train and validate the model
    train_loss, val_loss = train_and_val(train_data=train_data,
                  train_index=train_index,
                  train_dates=train_dates,
                  val_data=val_data,
                  val_index=val_index,
                  val_dates=val_dates,
                  assets=assets,
                  config=config,
                  device=device)

    print(f"Train loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")



    # Store the result
    validation_results.append((random_seed, val_loss))


# Sort results by validation loss
validation_results.sort(key=lambda x: x[1])

# Write results to a file
with open("seed_output.txt", "w") as f:
    f.write("Seed, Validation Loss\n")
    for seed, val_loss in validation_results:
        f.write(f"{seed}, {val_loss:.6f}\n")

print("Validation results saved to seed_output.txt")

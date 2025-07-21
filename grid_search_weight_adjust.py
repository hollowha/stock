import math, pdb, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as cPickle

from datetime import datetime
from dateutil.relativedelta import relativedelta

import torch
import torch.nn as nn

import csv

from argparse import ArgumentParser

from load_config import config
from utils import *
from load_holding_weights import *

from finolio.asset import Asset
from finolio.portfolio import Portfolio
from finolio.types import *

# Import the new training function
from train import train_and_test_for_combined_weight, train_and_predict_for_combined_weight, train_and_test_for_combined_weight_Lagrangian
from optimize import *


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")

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




def split_range(Dataset: pd.DataFrame, split_date_from: str, split_date_to: str):
    """
    Returns data: torch.tensor, index_: torch.tensor, dates: np.array, for the given date range.
    """
    mask = (Dataset.index >= split_date_from) & (Dataset.index <= split_date_to)
    
    data = Dataset.loc[mask]
    data = np.array(data)
    data = torch.from_numpy(data).float()

    dates = Dataset.loc[mask].index

    index_start = len(Dataset.loc[Dataset.index < split_date_from])
    index_end = len(Dataset.loc[Dataset.index <= split_date_to])
    index_ = index[index_start:index_end]
    index_ = np.array(index_)
    index_ = torch.from_numpy(index_).float()

    assert data.shape[0] == index_.shape[0]
    return data, index_, dates


# Read configuration parameters
delay_interval_months = config["delay_interval_months"]
testing_interval_days = config["testing_interval_days"]
loss2_interval_days = config["loss2_interval_days"]
fee = config["fee"]
weights_dir = "results"
removed = ["3705", "1513"]


# Loop over each testing period
start_date = datetime.strptime(Dataset.index[0], "%Y-%m-%d")
end_date = datetime.strptime(Dataset.index[-1], "%Y-%m-%d")

# current_date = start_date + relativedelta(months=delay_interval_months)
raw_date = start_date + relativedelta(months=delay_interval_months)
first_day_of_month = raw_date.replace(day=1).strftime("%Y-%m-%d")
if first_day_of_month > Dataset.index[-1]:  
    print(f"Warning: {first_day_of_month} exceeds dataset range. Using last available date: {Dataset.index[-1]}.")
    current_date = Dataset.index[-1]
else:
    current_date = Dataset.index[Dataset.index >= first_day_of_month][0]
current_date_init = str(current_date).split()[0]
current_date_index_init = Dataset.index.get_loc(current_date)

start_date_str = str(start_date).split()[0]


# define your grid
max_change_values   = np.arange(0.30, 0, -0.025)
# max_change_values   = [0.30, 0.275, 0.25, 0.225, 0.20]
# lambda_score_values = np.arange(1_000_000, 1_000_001, 500_000)
lambda_score_values = [10]
results = []

# outer grid search
for max_change in max_change_values:
    for lambda_score in lambda_score_values:

        # ---- re-init for each grid point ----
        previous_pf    = None
        balance        = 100_000
        total_fee      = 0.0
        current_date   = current_date_init   # whatever your initial date is
        current_date_index  = current_date_index_init
        output_dir = f"{max_change:.4f}_{lambda_score:.4f}"

        # if you want to plot every curve, you could create a new fig here:
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        # Plot 0050 and index first
        tmp, _, _ = split_range(Dataset, str(start_date).split()[0], str(current_date).split()[0])
        print(tmp.shape)
        skipped_days = tmp.shape[0]-1
        
        index_0050_asset = Asset(daily_return=pd.Series(index_0050[skipped_days:], index=pd.to_datetime(Dataset.index[skipped_days:])),
                                 name="0050",
                                 asset_type=ASSET_TYPE.MARKET_INDEX)
        index_0050_balance = 100000 / index_0050_asset.cumulative_return[0]
        ax = index_0050_asset.plot_series(ax=ax, balance=index_0050_balance, alpha=1)

        
        # rolling window backtest
        while current_date_index < len(Dataset.index)-1:
            split_from = str(current_date).split()[0]
            split_to_candidate = current_date_index+testing_interval_days
            if split_to_candidate >= len(Dataset.index):
              split_to_index = len(Dataset.index)-1
              split_to = Dataset.index[-1]
            else:
              split_to_index = split_to_candidate
              split_to = Dataset.index[split_to_candidate]
        
            split_to = str(split_to).split()[0]
        
        
            train_data, train_index, train_dates = split_range(Dataset, start_date_str, split_from)
            test_data, test_index, test_dates = split_range(Dataset, split_from, split_to)
            
            print(f"Train {train_dates[0]} ~ {train_dates[-1]} | Test {test_dates[0]} ~ {test_dates[-1]}")
        
            delay_indices = 0
            delay_candidate = current_date_index-loss2_interval_days
            if delay_candidate > 0:
              delay_indices = delay_candidate
        
            
            # Handle outliers
            q3 = train_data.abs().quantile(0.75)
            q3 = q3 + 1.5 * (q3 - train_data.abs().quantile(0.25))
            train_data[train_data > q3] = q3
            train_data[train_data < -q3] = -q3
            # test_data[test_data > q3] = q3
            # test_data[test_data < -q3] = -q3
            
            '''
            pf_original = weights_to_optimized_pf(rebalance_date=test_dates[0],
                                        train_data=train_data,
                                        test_data=test_data,
                                        test_dates=test_dates,
                                        assets=assets,
                                        config=config,
                                        weights_dir=weights_dir,
                                        previous_pf=previous_pf,
                                        removed=removed,
                                        device="cpu",
                                        lambda_penalty=27.5,
                                        initial_gamma=0.875)
            
            pf_original = weights_to_pf(rebalance_date=test_dates[0], 
                                        test_data=test_data,
                                        test_dates=test_dates,
                                        assets=assets,
                                        weights_dir=weights_dir,
                                        device="cpu",
                                        previous_pf=previous_pf,
                                        output_dir='diff')
            '''
            pf_original = weights_to_optimized_pf_abs(rebalance_date=test_dates[0],
                                        train_data=train_data,
                                        test_data=test_data,
                                        test_dates=test_dates,
                                        assets=assets,
                                        config=config,
                                        weights_dir=weights_dir,
                                        output_dir=output_dir,
                                        previous_pf=previous_pf,
                                        removed=removed,
                                        balance=balance,
                                        max_change=max_change,
                                        lambda_score=lambda_score)
                                  
                
            # fee
            if previous_pf is not None:
              weight_diff = pf_original.weights - previous_pf.weights
              transaction_fee = 0.5 * np.sum(np.abs(weight_diff)) * fee * balance
            else:
              transaction_fee = 0
              
            balance = balance - transaction_fee
            total_fee = total_fee + transaction_fee
        
            # Plot portfolio performance
            adjusted_cumulative_return = pf_original.cumulative_return - (pf_original.cumulative_return.iloc[0]-1) # return of the purchase day is NOT counted 
            cumulative_value = adjusted_cumulative_return * balance
            profit = cumulative_value - 100000
            balances = np.where(
              profit > 0,
              profit * 0.78 + 100000,
              cumulative_value
            )
            balances = pd.Series(balances, index=pf_original.cumulative_return.index)
            
            ax.plot(pd.to_datetime(test_dates), balances, alpha=1, label="Ours")
            
        
            balance = cumulative_value[-1]
            previous_pf = pf_original    
            
            current_date_index = split_to_index
            current_date = split_to
        
        
        print("Output figure...")
        
        title = f"delay interval: {delay_interval_months} months; testing interval: {testing_interval_days} days; fee: {fee * 100:.5f}%"
        ax.set_title(title)
        ax.set_xlabel("Dates")
        ax.set_ylabel("$")
        ax.grid(True)
        output_filename = f"{config['output'][:-4]}_{max_change:.2f}_{lambda_score}_fee={total_fee:.4f}.png"
        output_file = os.path.join(output_dir, output_filename)
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
        
        print(f"Result saved to {output_file}")
        




        # Use the entire dataset as training data
        all_train_data = torch.from_numpy(Dataset.values).float()
        all_train_index = index  # 'index' was computed earlier
        all_train_dates = Dataset.index  # entire date index
        
        delay_indices = 0
        delay_candidate = len(Dataset.index)-loss2_interval_days
        if delay_candidate > 0:
          delay_indices = delay_candidate
        
        # Handle outliers (mimicking the while loop code)
        q3 = all_train_data.abs().quantile(0.75)
        q3 = q3 + 1.5 * (q3 - all_train_data.abs().quantile(0.25))
        all_train_data[all_train_data > q3] = q3
        all_train_data[all_train_data < -q3] = -q3
        
        '''
        prediction_weights_to_optimized_pf(
            train_data=all_train_data,
            assets=assets,
            config=config,
            weights_dir=weights_dir,
            removed=removed,
            lambda_penalty=20.0,
            initial_gamma=0.99)
        '''
        prediction_weights_to_optimized_pf_abs(
                                train_data=all_train_data,
                                assets=assets,
                                config=config,
                                weights_dir=weights_dir,
                                output_dir=output_dir,
                                removed=removed,
                                max_change=max_change,
                                lambda_score=lambda_score,
                                )
        
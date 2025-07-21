import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from finolio.asset import Asset
from finolio.portfolio import Portfolio
from finolio.types import *
# from load_config import config
from entmax import entmax15, sparsemax
from math import sqrt  # Import sqrt from math

from model import Generator, loss_fn, loss_combined, loss_combined_holding, loss_combined_holding_Lagrange
from load_holding_weights import *

# from sklearn.linear_model import LinearRegression

# Single seed

def train_and_test(train_data: torch.tensor,
          train_index: torch.tensor,
          train_dates: np.array,
          test_data: torch.tensor,
          test_index: torch.tensor,
          test_dates: np.array,
          assets: np.array,
          config: dict,
          device: torch.device,
          delay_indices: int,
          start_trading_weight: float,
    ) -> Portfolio:
    actor = Generator(noise_dim = config["noise"], output_dim = len(assets)).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)

    best_reward = None

    for epoch in range(config["iterations"]):
        torch.cuda.empty_cache()
        weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))

        #robustness of portfolio candidates against dropping 75% of their weights
        dweights = nn.functional.dropout(dweights, p = 0.75).softmax(dim=1)

        loss1 = loss_combined(dweights, train_data[:delay_indices], train_index[:delay_indices], 1, 0, 0, 0.3, device, True).mean()
        loss2 = loss_combined(dweights, train_data[delay_indices:], train_index[delay_indices:], 1, 0.003, 0, 0.3, device, True).mean()
        loss = loss1 + loss2 * start_trading_weight
        # loss = loss_combined(dweights, train_data, train_index, 1, 0, 0.3, device, True).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()


    weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
    # combined_weights = sparsemax(weights.mean(dim=0), dim=0)
    topk = int(config["batch"] / 4)
    portfolio_weights = sparsemax(weights.clone(), dim = 1)
    portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
    portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim = 1)
    top_k_indices = torch.topk(portfolio_returns, topk, largest = True, sorted = True).indices
    combined_weights = portfolio_weights[top_k_indices].mean(dim = 0)
    
    bw = combined_weights.detach().cpu().numpy()
    bw = pd.DataFrame(bw).set_index([assets])
    bw = bw.loc[~(bw==0).all(axis=1)]
    bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
    bw.to_csv(f"{config['output'][:-4]}_{test_dates[0]}.csv", header=False)
    
    pf = Portfolio(daily_returns=pd.DataFrame(test_data, index=pd.to_datetime(test_dates)),
                   weights=combined_weights.tolist(),
                   name=f"train {train_dates[0]} ~ {train_dates[-1]} | test {test_dates[0]} ~ {test_dates[-1]}")

    return pf
    
def train_and_predict(train_data: torch.tensor,
          train_dates: np.array,
          train_index: torch.tensor,
          assets: np.array,
          config: dict,
          device: torch.device,
    ) -> Portfolio:
    actor = Generator(noise_dim = config["noise"], output_dim = len(assets)).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)

    best_reward = None

    for epoch in range(config["iterations"]):
        torch.cuda.empty_cache()
        weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))

        #robustness of portfolio candidates against dropping 75% of their weights
        dweights = nn.functional.dropout(dweights, p = 0.75).softmax(dim=1)

        loss = loss_combined(dweights, train_data, train_index, 1, 0, 0, 0.3, device, True).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()


    weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
    # combined_weights = sparsemax(weights.mean(dim=0), dim=0)
    topk = int(config["batch"] / 4)
    portfolio_weights = sparsemax(weights.clone(), dim = 1)
    portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
    portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim = 1)
    top_k_indices = torch.topk(portfolio_returns, topk, largest = True, sorted = True).indices
    combined_weights = portfolio_weights[top_k_indices].mean(dim = 0)
    
    bw = combined_weights.detach().cpu().numpy()
    bw = pd.DataFrame(bw).set_index([assets])
    bw = bw.loc[~(bw==0).all(axis=1)]
    bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
    bw.to_csv(f"{config['output'][:-4]}_prediction.csv", header=False)
    
    pf = Portfolio(daily_returns=pd.DataFrame(train_data, index=pd.to_datetime(train_dates)),
                   weights=combined_weights.tolist(),
                   name=f"Training data")
    
    return pf

def train_and_val(train_data: torch.tensor,
                  train_index: torch.tensor,
                  train_dates: np.array,
                  val_data: torch.tensor,
                  val_index: torch.tensor,
                  val_dates: np.array,
                  assets: np.array,
                  config: dict,
                  device: torch.device) -> tuple:
    """
    Train the model on train_data and calculate validation loss on val_data.
    Returns the final train_loss and validation loss.
    """
    actor = Generator(noise_dim=config["noise"], output_dim=len(assets)).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)

    final_train_loss = None

    for epoch in range(config["iterations"]):
        torch.cuda.empty_cache()
        weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))

        # Apply dropout to ensure robustness
        dweights = nn.functional.dropout(dweights, p=0.75).softmax(dim=1)

        # Compute training loss
        train_loss = loss_combined(dweights, train_data, train_index, 1, 0, 0, 0.3, device, True).mean()
        final_train_loss = train_loss.detach().item()  # Update final train loss for the last epoch

        loss = train_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()

    # Use combined weights to calculate validation loss
    weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
    # combined_weights = sparsemax(weights.mean(dim=0), dim=0)
    
    topk = int(config["batch"] / 4)
    portfolio_weights = sparsemax(weights.clone(), dim = 1)
    portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
    portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim = 1)
    top_k_indices = torch.topk(portfolio_returns, topk, largest = True, sorted = True).indices
    combined_weights = portfolio_weights[top_k_indices].mean(dim = 0)
    
    # Calculate validation loss
    val_loss = loss_combined(combined_weights.unsqueeze(0), val_data, val_index, 1, 0, 0, 0, device, False).mean().item()

    return final_train_loss, val_loss
    

# Multiple seeds        


def train_and_test_for_combined_weight(train_data: torch.tensor,
                                       train_index: torch.tensor,
                                       train_dates: np.array,
                                       test_data: torch.tensor,
                                       test_index: torch.tensor,
                                       test_dates: np.array,
                                       assets: np.array,
                                       unlisted_mask: torch.tensor,
                                       config: dict,
                                       device: torch.device,
                                       delay_indices: int,
                                       start_trading_weight_candidates: list,  # MODIFIED
                                       seeds: list,
                                       previous_pf: Portfolio,
                                       ) -> Portfolio:
    """
    Inputs:
      - start_trading_weight_candidates: List of candidate trading weight factors  # MODIFIED
    """
    loss2_weight = config["loss2_weight"]
    reg_weight = config["reg_weight"]
    reg_ratio = config["reg_ratio"]
    diff_weight = config["diff_weight"]
    ratio_SR_return = config["SR_weight"]
    validation_days = config["validation_days"] 
    
    if config["keep_previous_pf"] == 0:
      previous_pf = None
    
    best_combined_weights = None
    best_score = -float('inf')
    best_weight_candidate = None
    
    holding_weights = None
    if previous_pf is not None:
      holding_weights = torch.tensor(previous_pf.weights.values, dtype=torch.float).unsqueeze(0)
    
    
    for start_trading_weight in start_trading_weight_candidates:  # MODIFIED
        combined_weights_all_seeds = []
        print(f"Processing candidate weight: {start_trading_weight}")  # Added print statement
        
        # Loop through each seed for independent training
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            actor = Generator(noise_dim=config["noise"], output_dim=len(assets)).to(device)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)
            
            # Training loop
            for epoch in range(config["iterations"]):
                torch.cuda.empty_cache()
                # weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))
                # dweights = nn.functional.dropout(dweights, p=0.75).softmax(dim=1)
                weights, dlogits = actor(torch.randn((config["batch"], config["noise"])).to(device))
                dlogits = dlogits.masked_fill(unlisted_mask, -1e9)
                dweights = nn.functional.dropout(dlogits, p=0.75).softmax(dim=1)                
                
                if holding_weights is not None:
                  loss1 = loss_combined_holding(dweights, train_data[:delay_indices], train_index[:delay_indices],
                                                  holding_weights, 
                                                  1, 0, reg_weight, reg_ratio, diff_weight, device, True).mean()
                  loss2 = loss_combined_holding(dweights, train_data[delay_indices:], train_index[delay_indices:],
                                                  holding_weights, 
                                                  1, loss2_weight, reg_weight, reg_ratio, diff_weight, device, True).mean()
                else:
                  loss1 = loss_combined(dweights, train_data[:delay_indices], train_index[:delay_indices],
                                      1, 0, reg_weight, reg_ratio, device, True).mean()
                  loss2 = loss_combined(dweights, train_data[delay_indices:], train_index[delay_indices:],
                                      1, loss2_weight, reg_weight, reg_ratio, device, True).mean()
                
                loss = loss1 + loss2 * start_trading_weight
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
            
            # Compute combined weight after training
            weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
            weights = weights.masked_fill(unlisted_mask, -1e9)
            topk = int(config["batch"] / 4)
            portfolio_weights = sparsemax(weights.clone(), dim=1)
            portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
            portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim=1)
            top_k_indices = torch.topk(portfolio_returns, topk, largest=True, sorted=True).indices
            combined_weights = portfolio_weights[top_k_indices].mean(dim=0)
            combined_weights_all_seeds.append(combined_weights)
        
        # Compute final average combined weights for this candidate
        final_avg_combined_weights = torch.stack(combined_weights_all_seeds).mean(dim=0)
        
        # Compute portfolio daily returns for the period after delay_indices
        portfolio_daily_pct = final_avg_combined_weights @ train_data[-validation_days:].T
        # Compute total portfolio return
        portfolio_return = torch.prod(1 + portfolio_daily_pct).detach().item()
        # Compute Sharpe ratio using the candidate's daily returns (assumes sharpe_ratio function is defined)
        SR = sharpe_ratio(portfolio_daily_pct, validation_days, risk_free_rate=0.0)
        # Combine the two metrics with a 3:7 weight (3 for Sharpe ratio, 7 for return)
        candidate_score = ratio_SR_return * SR + (1-ratio_SR_return) * portfolio_return

        # Update best weights if current candidate is better
        if candidate_score > best_score:
            best_score = candidate_score
            best_combined_weights = final_avg_combined_weights
            best_weight_candidate = start_trading_weight
    
    # Previous_pf's return
    max_asset_diff_percent = -1
    if previous_pf is not None:
      previous_pf_daily_pct = holding_weights @ train_data[-validation_days:].T
      previous_pf_return = torch.prod(1 + previous_pf_daily_pct).item()
      previous_pf_SR = sharpe_ratio(previous_pf_daily_pct, validation_days, risk_free_rate=0.0)
      previous_pf_score = ratio_SR_return * previous_pf_SR + (1-ratio_SR_return) * previous_pf_return
    
      if previous_pf_score > best_score:
        bw = previous_pf.weights.copy()
        bw = pd.DataFrame(bw).set_index([assets])
        bw = bw.loc[~(bw == 0).all(axis=1)]
        bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
        bw.to_csv(f"{config['output'][:-4]}_{test_dates[0]}_-1_diff=0.csv", header=False)

        pf = Portfolio(daily_returns=pd.DataFrame(test_data, index=pd.to_datetime(test_dates)),
                   weights=previous_pf.weights,
                   name=f"train {train_dates[0]} ~ {train_dates[-1]} | test {test_dates[0]} ~ {test_dates[-1]}")
        return pf
      else:        
        # Remove batch dimension for holding_weights
        holding_weights_squeezed = holding_weights.squeeze(0)
        num_asset = best_combined_weights.shape[0]
        holding_assets = holding_weights_squeezed[:num_asset]
        diff_ratio = torch.zeros_like(holding_assets)
        nonzero_mask = holding_assets != 0
        diff_ratio[nonzero_mask] = torch.abs((best_combined_weights[nonzero_mask] - holding_assets[nonzero_mask]) / holding_assets[nonzero_mask])
        max_asset_diff_percent = diff_ratio.max().item()
    
    # Save the best weights to CSV  # MODIFIED
    bw = best_combined_weights.detach().cpu().numpy()
    bw = pd.DataFrame(bw).set_index([assets])
    bw = bw.loc[~(bw == 0).all(axis=1)]
    bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
    bw.to_csv(f"{config['output'][:-4]}_{test_dates[0]}_{best_weight_candidate}_diff={max_asset_diff_percent:.4f}.csv", header=False)  # MODIFIED
    
    # Construct final portfolio with the best weight  # MODIFIED
    pf = Portfolio(daily_returns=pd.DataFrame(test_data, index=pd.to_datetime(test_dates)),
                   weights=best_combined_weights.tolist(),
                   name=f"train {train_dates[0]} ~ {train_dates[-1]} | test {test_dates[0]} ~ {test_dates[-1]}")
    return pf


def train_and_test_for_combined_weight_Lagrangian(train_data: torch.tensor,
                                       train_index: torch.tensor,
                                       train_dates: np.array,
                                       test_data: torch.tensor,
                                       test_index: torch.tensor,
                                       test_dates: np.array,
                                       assets: np.array,
                                       config: dict,
                                       device: torch.device,
                                       delay_indices: int,
                                       start_trading_weight_candidates: list,  # MODIFIED
                                       seeds: list,
                                       previous_pf: Portfolio,
                                       ) -> Portfolio:
    """
    Inputs:
      - start_trading_weight_candidates: List of candidate trading weight factors  # MODIFIED
    """
    loss2_weight = config["loss2_weight"]
    reg_weight = config["reg_weight"]
    reg_ratio = config["reg_ratio"]
    diff_weight = config["diff_weight"]
    ratio_SR_return = config["SR_weight"]
    validation_days = config["validation_days"] 
    
    if config["keep_previous_pf"] == 0:
      previous_pf = None
    
    best_combined_weights = None
    best_score = -float('inf')
    best_weight_candidate = None
    
    holding_weights = None
    expanded_holding_assets = None
    if previous_pf is not None:
      holding_weights = torch.tensor(previous_pf.weights.values, dtype=torch.float).unsqueeze(0)
      expanded_holding_assets = holding_weights[:, :len(assets)].expand(config["batch"], -1)  # shape: [batch, num_assets]
    
    
    for start_trading_weight in start_trading_weight_candidates:  # MODIFIED
        combined_weights_all_seeds = []
        print(f"Processing candidate weight: {start_trading_weight}")  # Added print statement
        
        # Loop through each seed for independent training
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            actor = Generator(noise_dim=config["noise"], output_dim=len(assets)).to(device)
            # Define learnable lambda_raw, and always use softplus(lambda_raw) as lambda_diff
            lambda_raw = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))
            optimizer_actor = torch.optim.AdamW(actor.parameters(), lr=1e-3)
            optimizer_lambda = torch.optim.SGD([lambda_raw], lr=1e-2)  # or use Adagrad
            
            # Training loop
            for epoch in range(config["iterations"]):
                torch.cuda.empty_cache()
                weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))
                dweights = nn.functional.dropout(weights, p=0).softmax(dim=1)
                lambda_diff = torch.nn.functional.softplus(lambda_raw)
                
                if holding_weights is not None:
                  loss1 = loss_combined_holding_Lagrange(dweights, train_data[:delay_indices], train_index[:delay_indices],
                                     holding_weights, 
                                     1, 0, reg_weight, reg_ratio, 
                                     lambda_diff=lambda_diff, device=device, train=True).mean()
                  loss2 = loss_combined_holding_Lagrange(dweights, train_data[delay_indices:], train_index[delay_indices:],
                                                           holding_weights, 
                                                           1, loss2_weight, reg_weight, reg_ratio, 
                                                           lambda_diff=lambda_diff, device=device, train=True).mean()
                else:
                  loss1 = loss_combined(dweights, train_data[:delay_indices], train_index[:delay_indices],
                                      1, 0, reg_weight, reg_ratio, device, True).mean()
                  loss2 = loss_combined(dweights, train_data[delay_indices:], train_index[delay_indices:],
                                      1, loss2_weight, reg_weight, reg_ratio, device, True).mean()
                
                loss = loss1 + loss2 * start_trading_weight
                
                if previous_pf is not None:
                  with torch.no_grad():
                    weights_assets = dweights  # shape: [batch, num_assets]
                    diff = torch.abs(weights_assets - expanded_holding_assets)
                    diff_ratio = torch.zeros_like(expanded_holding_assets)
                    mask = expanded_holding_assets != 0
                    diff_ratio[mask] = diff[mask] / expanded_holding_assets[mask]
                    valid_mean = diff_ratio[mask].mean().item() if mask.sum() > 0 else 0.0
                    
                    valid_mean = diff_ratio[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=device)
                    lambda_diff.data += 1e-2 * valid_mean
                    lambda_diff.data.clamp_(min=0.0)
                    
                    #if epoch % 8 == 0:
                    #  print(f"[Epoch {epoch}] mean diff_ratio = {valid_mean:.4f}, max = {diff_ratio.max().item():.4f}, lambda = {lambda_diff.item():.4f}")

                optimizer_actor.zero_grad()
                optimizer_lambda.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer_actor.step()
                optimizer_lambda.step()
            
            # Compute combined weight after training
            weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
            topk = int(config["batch"] / 4)
            portfolio_weights = sparsemax(weights.clone(), dim=1)
            portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
            portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim=1)
            top_k_indices = torch.topk(portfolio_returns, topk, largest=True, sorted=True).indices
            combined_weights = portfolio_weights[top_k_indices].mean(dim=0)
            combined_weights_all_seeds.append(combined_weights)
        
        # Compute final average combined weights for this candidate
        final_avg_combined_weights = torch.stack(combined_weights_all_seeds).mean(dim=0)
        
        # Compute portfolio daily returns for the period after delay_indices
        portfolio_daily_pct = final_avg_combined_weights @ train_data[-validation_days:].T
        # Compute total portfolio return
        portfolio_return = torch.prod(1 + portfolio_daily_pct).item()
        # Compute Sharpe ratio using the candidate's daily returns (assumes sharpe_ratio function is defined)
        SR = sharpe_ratio(portfolio_daily_pct, validation_days, risk_free_rate=0.0)
        # Combine the two metrics with a 3:7 weight (3 for Sharpe ratio, 7 for return)
        candidate_score = ratio_SR_return * SR + (1-ratio_SR_return) * portfolio_return

        # Update best weights if current candidate is better
        if candidate_score > best_score:
            best_score = candidate_score
            best_combined_weights = final_avg_combined_weights
            best_weight_candidate = start_trading_weight
    
    # Previous_pf's return
    max_asset_diff_percent = -1
    if previous_pf is not None:
      previous_pf_daily_pct = holding_weights @ train_data[-validation_days:].T
      previous_pf_return = torch.prod(1 + previous_pf_daily_pct).item()
      previous_pf_SR = sharpe_ratio(previous_pf_daily_pct, validation_days, risk_free_rate=0.0)
      previous_pf_score = ratio_SR_return * previous_pf_SR + (1-ratio_SR_return) * previous_pf_return
    
      if previous_pf_score > best_score:
        bw = previous_pf.weights.copy()
        bw = pd.DataFrame(bw).set_index([assets])
        bw = bw.loc[~(bw == 0).all(axis=1)]
        bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
        bw.to_csv(f"{config['output'][:-4]}_{test_dates[0]}_-1_diff=0.csv", header=False)

        pf = Portfolio(daily_returns=pd.DataFrame(test_data, index=pd.to_datetime(test_dates)),
                   weights=previous_pf.weights,
                   name=f"train {train_dates[0]} ~ {train_dates[-1]} | test {test_dates[0]} ~ {test_dates[-1]}")
        return pf
      else:        
        # Remove batch dimension for holding_weights
        holding_weights_squeezed = holding_weights.squeeze(0)
        num_asset = best_combined_weights.shape[0]
        holding_assets = holding_weights_squeezed[:num_asset]
        diff_ratio = torch.zeros_like(holding_assets)
        nonzero_mask = holding_assets != 0
        diff_ratio[nonzero_mask] = torch.abs((best_combined_weights[nonzero_mask] - holding_assets[nonzero_mask]) / holding_assets[nonzero_mask])
        max_asset_diff_percent = diff_ratio.max().item()
    
    # Save the best weights to CSV  # MODIFIED
    bw = best_combined_weights.detach().cpu().numpy()
    bw = pd.DataFrame(bw).set_index([assets])
    bw = bw.loc[~(bw == 0).all(axis=1)]
    bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
    bw.to_csv(f"{config['output'][:-4]}_{test_dates[0]}_{best_weight_candidate}_diff={max_asset_diff_percent:.4f}.csv", header=False)  # MODIFIED
    
    # Construct final portfolio with the best weight  # MODIFIED
    pf = Portfolio(daily_returns=pd.DataFrame(test_data, index=pd.to_datetime(test_dates)),
                   weights=best_combined_weights.tolist(),
                   name=f"train {train_dates[0]} ~ {train_dates[-1]} | test {test_dates[0]} ~ {test_dates[-1]}")
    return pf


def train_and_predict_for_combined_weight(train_data: torch.tensor,
                                          train_index: torch.tensor,
                                          assets: np.array,
                                          config: dict,
                                          device: torch.device,
                                          delay_indices: int,
                                          start_trading_weight_candidates: list,
                                          seeds: list) -> None:
    """
    Inputs:
      - train_data, train_index: Training data and corresponding index for loss calculation.
      - assets: List of assets.
      - config: Dictionary containing configuration parameters such as "noise", "batch", "iterations", and "output".
      - device: Torch device (CPU/GPU).
      - delay_indices: Delay indices for segmented loss calculation.
      - start_trading_weight_candidates: List of candidate start trading weights.
      - seeds: List of seeds for independent training runs.
    Output:
      - Writes the best performing combined weights to a CSV file.
    """
    loss2_weight = config["loss2_weight"]
    reg_weight = config["reg_weight"]
    reg_ratio = config["reg_ratio"]
    diff_weight = config["diff_weight"]
    ratio_SR_return = config["SR_weight"]
    validation_days = config["validation_days"]
    
    best_combined_weights = None
    best_score = -float('inf')
    best_weight_candidate = None

    # Original training using loss_combined
    for start_trading_weight in start_trading_weight_candidates:
        combined_weights_all_seeds = []
        print(f"Processing candidate weight: {start_trading_weight}")

        # Loop through each seed for independent training
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            actor = Generator(noise_dim=config["noise"], output_dim=len(assets)).to(device)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)

            for epoch in range(config["iterations"]):
                torch.cuda.empty_cache()
                weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))
                dweights = nn.functional.dropout(dweights, p=0.75).softmax(dim=1)

                loss1 = loss_combined(dweights, train_data[:delay_indices], train_index[:delay_indices],
                                      1, 0, reg_weight, reg_ratio, device, True).mean()
                loss2 = loss_combined(dweights, train_data[delay_indices:], train_index[delay_indices:],
                                      1, loss2_weight, reg_weight, reg_ratio, device, True).mean()
                loss = loss1 + loss2 * start_trading_weight

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()

            weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
            topk = int(config["batch"] / 4)
            portfolio_weights = sparsemax(weights.clone(), dim=1)
            portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
            portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim=1)
            top_k_indices = torch.topk(portfolio_returns, topk, largest=True, sorted=True).indices
            combined_weights = portfolio_weights[top_k_indices].mean(dim=0)
            combined_weights_all_seeds.append(combined_weights)

        final_avg_combined_weights = torch.stack(combined_weights_all_seeds).mean(dim=0)


        # Compute portfolio daily returns for the period after delay_indices
        portfolio_daily_pct = final_avg_combined_weights @ train_data[-validation_days:].T
        # Compute total portfolio return
        portfolio_return = torch.prod(1 + portfolio_daily_pct).detach().item()
        # Compute Sharpe ratio using the candidate's daily returns (assumes sharpe_ratio function is defined)
        SR = sharpe_ratio(portfolio_daily_pct, validation_days, risk_free_rate=0.0)
        # Combine the two metrics with a 3:7 weight (3 for Sharpe ratio, 7 for return)
        candidate_score = ratio_SR_return * SR + (1-ratio_SR_return) * portfolio_return

        # Update best weights if current candidate is better
        if candidate_score > best_score:
            best_score = candidate_score
            best_combined_weights = final_avg_combined_weights
            best_weight_candidate = start_trading_weight

    # Save the best combined weights to a CSV file (sorted by absolute weight value)
    bw = best_combined_weights.detach().cpu().numpy()
    bw = pd.DataFrame(bw).set_index([assets])
    bw = bw.loc[~(bw == 0).all(axis=1)]
    bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
    bw.to_csv(f"{config['output'][:-4]}_prediction_{best_weight_candidate}.csv", header=False)
    

    
    # Load holding and price
    
   
    if diff_weight == 0:
      # diff_weight = 0.05
      return
    if config.get("holding"):
        # holding_stocks, holdings, avg_costs = load_holding(config["holding"])
        # price_stocks, prices = load_prices(config["price"])
        # total_value = calculate_total_value(holdings, prices, holding_stocks, price_stocks)
        # df_underwater = filter_underwater_stocks(holdings, avg_costs, prices, holding_stocks, price_stocks)
        holding_weights = load_holding_weights(config["holding"], assets, device=device)

        best_combined_weights = None
        best_score = -float('inf')
        best_weight_candidate = None

        for start_trading_weight in start_trading_weight_candidates:
            combined_weights_all_seeds = []
            print(f"Processing candidate weight (holding): {start_trading_weight}")

            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                actor = Generator(noise_dim=config["noise"], output_dim=len(assets)).to(device)
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)

                for epoch in range(config["iterations"]):
                    torch.cuda.empty_cache()
                    weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))

                    dweights = nn.functional.dropout(dweights, p=0.75).softmax(dim=1)

                    loss1 = loss_combined_holding(dweights, train_data[:delay_indices], train_index[:delay_indices],
                                                  holding_weights, 
                                                  1, 0, reg_weight, reg_ratio, diff_weight, device, True).mean()
                    loss2 = loss_combined_holding(dweights, train_data[delay_indices:], train_index[delay_indices:],
                                                  holding_weights, 
                                                  1, loss2_weight, reg_weight, reg_ratio, diff_weight, device, True).mean()
                    loss = loss1 + loss2 * start_trading_weight

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizer.step()

                weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
                topk = int(config["batch"] / 4)
                portfolio_weights = sparsemax(weights.clone(), dim=1)
                portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
                portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim=1)
                top_k_indices = torch.topk(portfolio_returns, topk, largest=True, sorted=True).indices
                combined_weights = portfolio_weights[top_k_indices].mean(dim=0)
                combined_weights_all_seeds.append(combined_weights)

            final_avg_combined_weights = torch.stack(combined_weights_all_seeds).mean(dim=0)

            portfolio_daily_pct = final_avg_combined_weights @ train_data[-validation_days:].T
            portfolio_return = torch.prod(1 + portfolio_daily_pct).item()
            SR = sharpe_ratio(portfolio_daily_pct, validation_days, risk_free_rate=0.0)
            candidate_score = ratio_SR_return * SR + (1-ratio_SR_return) * portfolio_return

            if candidate_score > best_score:
                best_score = candidate_score
                best_combined_weights = final_avg_combined_weights
                best_weight_candidate = start_trading_weight

        # Remove batch dimension for holding_weights
        holding_weights_squeezed = holding_weights.squeeze(0)
        holding_assets = holding_weights_squeezed[:34]
        holding_extra = holding_weights_squeezed[34:]
        diff_assets = torch.abs(best_combined_weights - holding_assets).sum()
        diff_extra = torch.abs(holding_extra).sum()
        total_diff = (diff_assets + diff_extra).item()
        
        bw = best_combined_weights.detach().cpu().numpy()
        bw = pd.DataFrame(bw).set_index([assets])
        bw = bw.loc[~(bw == 0).all(axis=1)]
        bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
        bw.to_csv(f"{config['output'][:-4]}_holding_diff={total_diff:.4f}_prediction_{best_weight_candidate}.csv", header=False)
        
        
        return


def train_and_multiple_holding_predict_for_combined_weight(train_data: torch.tensor,
                                          train_index: torch.tensor,
                                          assets: np.array,
                                          config: dict,
                                          device: torch.device,
                                          delay_indices: int,
                                          start_trading_weight_candidates: list,
                                          seeds: list) -> None:
    """
    Inputs:
      - train_data, train_index: Training data and corresponding index for loss calculation.
      - assets: List of assets.
      - config: Dictionary containing configuration parameters such as "noise", "batch", "iterations", and "output".
      - device: Torch device (CPU/GPU).
      - delay_indices: Delay indices for segmented loss calculation.
      - start_trading_weight_candidates: List of candidate start trading weights.
      - seeds: List of seeds for independent training runs.
    Output:
      - Writes the best performing combined weights to a CSV file.
    """
    loss2_weight = config["loss2_weight"]
    reg_weight = config["reg_weight"]
    reg_ratio = config["reg_ratio"]
    diff_weight = config["diff_weight"]
    ratio_SR_return = config["SR_weight"]
    validation_days = config["validation_days"]
    
    best_combined_weights = None
    best_score = -float('inf')
    best_weight_candidate = None

    diff_weight_set = [diff_weight]
    if diff_weight == 0:
      diff_weight_set = np.arange(0, 100, 5).tolist()
    if config.get("holding"):
        # holding_stocks, holdings, avg_costs = load_holding(config["holding"])
        # price_stocks, prices = load_prices(config["price"])
        # total_value = calculate_total_value(holdings, prices, holding_stocks, price_stocks)
        # df_underwater = filter_underwater_stocks(holdings, avg_costs, prices, holding_stocks, price_stocks)
        holding_weights = load_holding_weights(config["holding"], assets, device=device)
        
        for current_diff_weight in diff_weight_set:

          best_combined_weights = None
          best_score = -float('inf')
          best_weight_candidate = None
  
          for start_trading_weight in start_trading_weight_candidates:
              combined_weights_all_seeds = []
              print(f"Processing candidate weight (holding): {start_trading_weight}")
  
              for seed in seeds:
                  np.random.seed(seed)
                  torch.manual_seed(seed)
                  torch.cuda.manual_seed(seed)
  
                  actor = Generator(noise_dim=config["noise"], output_dim=len(assets)).to(device)
                  optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)
  
                  for epoch in range(config["iterations"]):
                      torch.cuda.empty_cache()
                      weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))
                      dweights = nn.functional.dropout(dweights, p=0.75).softmax(dim=1)
                      
                      loss1 = loss_combined_holding(dweights, train_data[:delay_indices], train_index[:delay_indices],
                                                    holding_weights, 
                                                    1, 0, reg_weight, reg_ratio, current_diff_weight, device, True).mean()
                      loss2 = loss_combined_holding(dweights, train_data[delay_indices:], train_index[delay_indices:],
                                                    holding_weights, 
                                                    1, loss2_weight, reg_weight, reg_ratio, current_diff_weight, device, True).mean()
                      loss = loss1 + loss2 * start_trading_weight
  
                      optimizer.zero_grad()
                      loss.backward()
                      nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                      optimizer.step()
  
                  weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
                  topk = int(config["batch"] / 4)
                  portfolio_weights = sparsemax(weights.clone(), dim=1)
                  portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
                  portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim=1)
                  top_k_indices = torch.topk(portfolio_returns, topk, largest=True, sorted=True).indices
                  combined_weights = portfolio_weights[top_k_indices].mean(dim=0)
                  combined_weights_all_seeds.append(combined_weights)
  
              final_avg_combined_weights = torch.stack(combined_weights_all_seeds).mean(dim=0)
  
              portfolio_daily_pct = final_avg_combined_weights @ train_data[-validation_days:].T
              portfolio_return = torch.prod(1 + portfolio_daily_pct).item()
              SR = sharpe_ratio(portfolio_daily_pct, validation_days, risk_free_rate=0.0)
              candidate_score = ratio_SR_return * SR + (1-ratio_SR_return) * portfolio_return
  
              if candidate_score > best_score:
                  best_score = candidate_score
                  best_combined_weights = final_avg_combined_weights
                  best_weight_candidate = start_trading_weight
  
          # Remove batch dimension for holding_weights
          holding_weights_squeezed = holding_weights.squeeze(0)
          num_asset = best_combined_weights.shape[0]
          holding_assets = holding_weights_squeezed[:num_asset]
          diff_ratio = torch.zeros_like(holding_assets)
          nonzero_mask = holding_assets != 0
          diff_ratio[nonzero_mask] = torch.abs((best_combined_weights[nonzero_mask] - holding_assets[nonzero_mask]) / holding_assets[nonzero_mask])
          max_asset_diff_percent = diff_ratio.max().item()
          
          # print("Per-asset diff ratio (% change):", diff_ratio.tolist())
          
          
          bw = best_combined_weights.detach().cpu().numpy()
          bw = pd.DataFrame(bw).set_index([assets])
          bw = bw.loc[~(bw == 0).all(axis=1)]
          bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
          bw.to_csv(f"{config['output'][:-4]}_holding_{current_diff_weight:.2f}_diff={max_asset_diff_percent:.4f}_prediction_{best_weight_candidate}.csv", header=False)
          
        return


def train_and_multiple_holding_predict_for_combined_weight_Lagrangian(train_data: torch.tensor,
                                          train_index: torch.tensor,
                                          assets: np.array,
                                          config: dict,
                                          device: torch.device,
                                          delay_indices: int,
                                          start_trading_weight_candidates: list,
                                          seeds: list) -> None:
    """
    Inputs:
      - train_data, train_index: Training data and corresponding index for loss calculation.
      - assets: List of assets.
      - config: Dictionary containing configuration parameters such as "noise", "batch", "iterations", and "output".
      - device: Torch device (CPU/GPU).
      - delay_indices: Delay indices for segmented loss calculation.
      - start_trading_weight_candidates: List of candidate start trading weights.
      - seeds: List of seeds for independent training runs.
    Output:
      - Writes the best performing combined weights to a CSV file.
    """
    loss2_weight = config["loss2_weight"]
    reg_weight = config["reg_weight"]
    reg_ratio = config["reg_ratio"]
    diff_weight = config["diff_weight"]
    ratio_SR_return = config["SR_weight"]
    validation_days = config["validation_days"]
    
    best_combined_weights = None
    best_score = -float('inf')
    best_weight_candidate = None

    if config.get("holding"):
        # holding_stocks, holdings, avg_costs = load_holding(config["holding"])
        # price_stocks, prices = load_prices(config["price"])
        # total_value = calculate_total_value(holdings, prices, holding_stocks, price_stocks)
        # df_underwater = filter_underwater_stocks(holdings, avg_costs, prices, holding_stocks, price_stocks)
        holding_weights = load_holding_weights(config["holding"], assets, device=device)
        expanded_holding_assets = holding_weights[:, :len(assets)].expand(config["batch"], -1)  # shape: [batch, num_assets]
        

        best_combined_weights = None
        best_score = -float('inf')
        best_weight_candidate = None

        for start_trading_weight in start_trading_weight_candidates:
            combined_weights_all_seeds = []
            print(f"Processing candidate weight (holding): {start_trading_weight}")

            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                actor = Generator(noise_dim=config["noise"], output_dim=len(assets)).to(device)
                # Define learnable lambda_raw, and always use softplus(lambda_raw) as lambda_diff
                lambda_raw = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))
                optimizer_actor = torch.optim.AdamW(actor.parameters(), lr=1e-3)
                optimizer_lambda = torch.optim.SGD([lambda_raw], lr=1e-2)  # or use Adagrad

                for epoch in range(config["iterations"]):
                    torch.cuda.empty_cache()
                    weights, dweights = actor(torch.randn((config["batch"], config["noise"])).to(device))
                    dweights = nn.functional.dropout(weights, p=0).softmax(dim=1)
                    lambda_diff = torch.nn.functional.softplus(lambda_raw)
                    
                    # loss1 = loss_combined_holding(dweights, train_data[:delay_indices], train_index[:delay_indices],
                    #                               holding_weights, 
                    #                               1, 0, reg_weight, reg_ratio, current_diff_weight, device, True).mean()
                    # loss2 = loss_combined_holding(dweights, train_data[delay_indices:], train_index[delay_indices:],
                    #                               holding_weights, 
                    #                               1, loss2_weight, reg_weight, reg_ratio, current_diff_weight, device, True).mean()
                    loss1 = loss_combined_holding_Lagrange(dweights, train_data[:delay_indices], train_index[:delay_indices],
                                     holding_weights, 
                                     1, 0, reg_weight, reg_ratio, 
                                     lambda_diff=lambda_diff, device=device, train=True).mean()
                    loss2 = loss_combined_holding_Lagrange(dweights, train_data[delay_indices:], train_index[delay_indices:],
                                                           holding_weights, 
                                                           1, loss2_weight, reg_weight, reg_ratio, 
                                                           lambda_diff=lambda_diff, device=device, train=True).mean()
                    loss = loss1 + loss2 * start_trading_weight
                    
                    with torch.no_grad():
                      weights_assets = dweights  # shape: [batch, num_assets]
                      diff = torch.abs(weights_assets - expanded_holding_assets)
                      diff_ratio = torch.zeros_like(expanded_holding_assets)
                      mask = expanded_holding_assets != 0
                      diff_ratio[mask] = diff[mask] / expanded_holding_assets[mask]
                      valid_mean = diff_ratio[mask].mean().item() if mask.sum() > 0 else 0.0
                      
                      valid_mean = diff_ratio[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=device)
                      lambda_diff.data += 1e-2 * valid_mean
                      lambda_diff.data.clamp_(min=0.0)
                      
                      #if epoch % 8 == 0:
                      #  print(f"[Epoch {epoch}] mean diff_ratio = {valid_mean:.4f}, max = {diff_ratio.max().item():.4f}, lambda = {lambda_diff.item():.4f}")

                    optimizer_actor.zero_grad()
                    optimizer_lambda.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizer_actor.step()
                    optimizer_lambda.step()

                weights, _ = actor(torch.randn((config["batch"], config["noise"])).to(device))
                topk = int(config["batch"] / 4)
                portfolio_weights = sparsemax(weights.clone(), dim=1)
                portfolio_daily_pct = portfolio_weights.matmul(train_data.T)
                portfolio_returns = torch.prod(1 + portfolio_daily_pct, dim=1)
                top_k_indices = torch.topk(portfolio_returns, topk, largest=True, sorted=True).indices
                combined_weights = portfolio_weights[top_k_indices].mean(dim=0)
                combined_weights_all_seeds.append(combined_weights)

            final_avg_combined_weights = torch.stack(combined_weights_all_seeds).mean(dim=0)

            portfolio_daily_pct = final_avg_combined_weights @ train_data[-validation_days:].T
            portfolio_return = torch.prod(1 + portfolio_daily_pct).item()
            SR = sharpe_ratio(portfolio_daily_pct, validation_days, risk_free_rate=0.0)
            candidate_score = ratio_SR_return * SR + (1-ratio_SR_return) * portfolio_return

            if candidate_score > best_score:
                best_score = candidate_score
                best_combined_weights = final_avg_combined_weights
                best_weight_candidate = start_trading_weight

        # Remove batch dimension for holding_weights
        holding_weights_squeezed = holding_weights.squeeze(0)
        num_asset = best_combined_weights.shape[0]
        holding_assets = holding_weights_squeezed[:num_asset]
        diff_ratio = torch.zeros_like(holding_assets)
        nonzero_mask = holding_assets != 0
        diff_ratio[nonzero_mask] = torch.abs((best_combined_weights[nonzero_mask] - holding_assets[nonzero_mask]) / holding_assets[nonzero_mask])
        max_asset_diff_percent = diff_ratio.max().item()
        
        print("Per-asset diff ratio (% change):", diff_ratio.tolist())
        
        
        bw = best_combined_weights.detach().cpu().numpy()
        bw = pd.DataFrame(bw).set_index([assets])
        bw = bw.loc[~(bw == 0).all(axis=1)]
        bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
        bw.to_csv(f"{config['output'][:-4]}_holding_Langrangian_diff={max_asset_diff_percent:.4f}_prediction_{best_weight_candidate}.csv", header=False)
        


def pf_from_combined_weights(combined_weights_list: list,
                             test_data: torch.tensor,
                             test_dates: np.array,
                             assets: np.array,
                             pf_name: str = "Average Combined Portfolio") -> Portfolio:
    """
    Inputs:
      - combined_weights_list: A list of combined_weight tensors, each with shape [num_assets]
      - test_data: Test data tensor
      - test_dates: Dates corresponding to the test data (used for DataFrame index)
      - assets: List of asset names (used for CSV saving and portfolio construction)
      - pf_name: Name of the portfolio (default: "Average Combined Portfolio")
    
    Output:
      - A portfolio created using the average of all input combined_weights
    """
    avg_combined_weights = torch.stack(combined_weights_list).mean(dim=0)
    pf = Portfolio(daily_returns=pd.DataFrame(test_data, index=pd.to_datetime(test_dates)),
                   weights=avg_combined_weights.tolist(),
                   name=pf_name)
    return pf
        

def sharpe_ratio(daily_pct_returns, num_days, risk_free_rate=0.0):
    # Calculate the daily risk-free rate
    daily_risk_free_rate = risk_free_rate / num_days

    # Compute the excess daily returns by subtracting the daily risk-free rate.
    excess_daily_returns = daily_pct_returns - daily_risk_free_rate

    # Calculate the mean of the excess daily returns.
    mean_excess_return = torch.mean(excess_daily_returns)

    # Calculate the standard deviation of the excess daily returns.
    std_excess_return = torch.std(excess_daily_returns)

    # Avoid division by zero by checking if std_excess_return is zero.
    if std_excess_return == 0:
        return 0.0

    # Compute the daily Sharpe ratio.
    daily_sharpe_ratio = mean_excess_return / std_excess_return

    # Annualize the Sharpe ratio by multiplying by sqrt(num_days).
    annual_sharpe_ratio = daily_sharpe_ratio * sqrt(num_days)
    # return float(annual_sharpe_ratio)
    return annual_sharpe_ratio.detach().item()


# Adaptive SR_weight


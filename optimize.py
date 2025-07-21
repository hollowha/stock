import os
import glob
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
from scipy.optimize import minimize

from model import Generator, loss_fn, loss_combined, loss_combined_holding, loss_combined_holding_Lagrange
from load_holding_weights import *

# from sklearn.linear_model import LinearRegression

def weights_to_pf(rebalance_date: str,
                  test_data: torch.tensor,
                  test_dates: np.array,
                  assets: list,
                  weights_dir: str,
                  device: str = "cpu",
                  previous_pf: Portfolio = None,
                  output_dir: str = None) -> Portfolio:
    """
    Locate the weight CSV corresponding to the given date, load and convert it to a tensor,
    and return a Portfolio object based on the provided test data.

    Parameters:
    - rebalance_date: date string in 'YYYY-MM-DD' format used to find the CSV file.
    - test_data: pd.DataFrame of daily returns, indexed by date, columns matching assets list.
    - assets: list of all asset names in test_data columns.
    - weights_dir: directory path containing weight CSV files.
    - device: device for tensor allocation, default 'cpu'.

    Returns:
    - Portfolio object with cumulative_return attribute.
    """
    # Find the CSV file that contains the rebalance_date
    pattern = os.path.join(weights_dir, f"*{rebalance_date}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No weight file found for date {rebalance_date} (pattern: {pattern})")
    csv_path = files[0]

    # Load holding weights as a tensor
    holding_weights = load_holding_weights(csv_path, assets, device=device)
    # Remove batch dimension and select the first len(assets) elements
    weights_tensor = holding_weights.squeeze(0)[:len(assets)]
    weights_list = weights_tensor.tolist()
    
    if previous_pf is not None and output_dir is not None:
        prev_weights = np.array(previous_pf.weights)
        cur_weights  = np.array(weights_list)
        save_diff(
            rebalance_date=rebalance_date,
            assets=assets,
            previous=prev_weights,
            current=cur_weights,
            output_dir=output_dir,
            diff_type="prediction"
        )

    # Create Portfolio object
    pf = Portfolio(daily_returns=pd.DataFrame(test_data, index=pd.to_datetime(test_dates)),
                   weights=weights_list,
                   name=f"test {test_dates[0]} ~ {test_dates[-1]}")

    return pf
    

def save_diff(rebalance_date: str,
              assets: list,
              previous: np.ndarray,
              current: np.ndarray,
              output_dir: str,
              diff_type: str,
              threshold: float = 0.1):

    os.makedirs(output_dir, exist_ok=True)
    
    df_prev = pd.DataFrame({'asset': assets, 'previous': previous})
    df_cur  = pd.DataFrame({'asset': assets, diff_type: current})
   
    df_union = pd.merge(df_prev, df_cur, on='asset', how='outer').fillna(0)
    
    df_union['rate'] = df_union.apply(
        # lambda r: (r[diff_type] - r['previous']) / r['previous']
        lambda r: r[diff_type] - r['previous']
        if r['previous'] != 0 else np.nan,
        axis=1
    )
    
    mask = df_union['previous'].abs() >= threshold
    max_diff = df_union.loc[mask, 'rate'].abs().max(skipna=True)
    
    fname = f"{rebalance_date}_{diff_type}_diff_maxDiff={max_diff:.4f}.csv"
    outpath = os.path.join(output_dir, fname)
    df_union.to_csv(outpath, index=False)
    return outpath  


def weights_to_optimized_pf(rebalance_date: str,
                            train_data: torch.Tensor,
                            test_data: torch.Tensor,
                            test_dates: np.ndarray,
                            assets: list,
                            config: dict,
                            weights_dir: str,
                            previous_pf: Portfolio,
                            removed: list,
                            device: str = "cpu",
                            lambda_penalty: float = 20.0,
                            min_lambda: float = 0.0001,
                            initial_gamma: float = 0.9,
                            min_gamma: float = 0.0001,
                            max_iter: int = 5000)->Portfolio:
    """
    Optimize predicted weights with turnover penalty and enforce a minimum score relative to predictions.

    Adds a hard constraint: optimized score >= gamma * prediction_score.
    Returns a Portfolio built from optimized weights, and saves CSVs.
    """
    # configuration
    ratio_SR_return = config["SR_weight"]
    validation_days = config["validation_days"]

    # locate prediction CSV and load predicted weights
    pattern = os.path.join(weights_dir, f"*{rebalance_date}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No prediction file for date {rebalance_date}: {pattern}")
    csv_path = files[0]
    pred_tensor = load_holding_weights(csv_path, assets, device=device)
    prediction = pred_tensor.squeeze(0)[:len(assets)].cpu().numpy()

    # compute prediction score
    sub_train = train_data[-validation_days:, :]  # shape (D, N)
    # portfolio daily pct: prediction @ sub_train^T
    w_pred = torch.tensor(prediction, dtype=torch.float, device=device)
    portfolio_daily_pct_pred = w_pred @ sub_train.T  # shape (D,)
    portfolio_return_pred = torch.prod(1 + portfolio_daily_pct_pred).item()
    SR_pred = sharpe_ratio(portfolio_daily_pct_pred, validation_days, risk_free_rate=0.0)
    pred_score = ratio_SR_return * SR_pred + (1 - ratio_SR_return) * portfolio_return_pred

    # handle no previous_pf: just save predictions
    if previous_pf is None:
        weights_list = prediction.tolist()
        df_w = pd.DataFrame({'weight': weights_list}, index=assets)
        df_w = df_w[df_w['weight'] != 0.0]
        df_w = df_w.reindex(df_w['weight'].abs().sort_values(ascending=False).index)
        reb_date = pd.to_datetime(test_dates[0]).strftime('%Y-%m-%d')
        outfile = f"{config['output'][:-4]}_{reb_date}_optimized.csv"
        df_w.to_csv(outfile, index=False, header=False)
        # print(f"Optimized weights saved to {outfile}")
        returns_df = pd.DataFrame(test_data.cpu().numpy(), index=pd.to_datetime(test_dates), columns=assets)
        return Portfolio(daily_returns=returns_df, weights=weights_list, name=f"Optimized {reb_date}")

    # prepare previous holdings and masks
    holding = np.array(previous_pf.weights, dtype=float)
    removed_mask = np.array([a in removed for a in assets], dtype=bool)

    # set up bounds, initial guess, penalty mask
    bounds, x0, penalized_mask = [], [], []
    for i in range(len(assets)):
        if removed_mask[i]:
            bounds.append((0.0, 0.0)); x0.append(0.0); penalized_mask.append(False)
        elif holding[i] == 0.0:
            bounds.append((0.0, 1.0)); x0.append(prediction[i]); penalized_mask.append(False)
        else:
            bounds.append((0.0, 1.0)); x0.append(prediction[i]); penalized_mask.append(True)
    x0 = np.array(x0, dtype=float)

    # objective with turnover penalty
    def objective(x, lam):
        dev = np.sum((x - prediction) ** 2)
        pen = 0.0
        for j in range(len(x)):
            if penalized_mask[j]:
                delta = abs(x[j] - holding[j])
                thr = 0.5 * holding[j]
                pen += delta if delta < thr else delta**2
        return dev + lam * pen

    # score function for optimized weights
    def score_fn(x):
        x_t = torch.tensor(x, dtype=torch.float, device=device)
        pct = x_t @ sub_train.T
        ret = torch.prod(1 + pct).item()
        sr = sharpe_ratio(pct, validation_days, risk_free_rate=0.0)
        return ratio_SR_return * sr + (1 - ratio_SR_return) * ret

    # constraints: sum(x)==1 and score_fn(x) >= gamma * pred_score
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: score_fn(x) - initial_gamma * pred_score}
    ]

    # solve with retry on iteration limit
    lam = lambda_penalty
    gamma = initial_gamma
    while True:
        result = minimize(
            fun=objective,
            x0=x0,
            args=(lam,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol':1e-9, 'maxiter':max_iter}
        )
        if result.success:
            break

        msg = result.message.lower()

        # 1) iteration limit, line-search, or LSQ
        if (('iteration limit' in msg)
            or ('positive directional derivative' in msg)
            or ('more than 3*n iterations' in msg)) \
           and lam > min_lambda:

            old_lam = lam
            lam *= 2.0/3.0
            print(f"Issue '{msg.split(':')[0]}', reducing lambda from {old_lam} to {lam}")
            continue

        # 2) inequality constraints
        if 'inequality constraints incompatible' in msg and gamma > min_gamma:
            old_gamma = gamma
            gamma *= 0.9
            print(f"Issue '{msg.split(':')[0]}', reducing gamma from {old_gamma} to {gamma}")
            constraints[1] = {
                'type': 'ineq',
                'fun': lambda x: score_fn(x) - gamma * pred_score
            }
            continue

        raise RuntimeError(f"Optimization failed at lambda={lam}, gamma={gamma}: {result.message}")
    optimized = result.x

    # save optimized weights
    df_opt = pd.DataFrame({'asset': assets, 'optimized': optimized})
    '''
    df_opt_nonzero = df_opt[df_opt['optimized'] != 0.0]
    df_opt_sorted = df_opt_nonzero.reindex(df_opt_nonzero['optimized'].abs().sort_values(ascending=False).index)
    opt_file = f"{config['output'][:-4]}_{rebalance_date}_optimized.csv"
    df_opt_sorted.to_csv(opt_file, index=False, header=False)
    # print(f"Optimized weights saved to {opt_file}")
    '''

    # save change-rate CSV
    df_prev = pd.DataFrame({'asset': assets, 'previous': holding})
    df_union = pd.merge(df_prev, df_opt, on='asset', how='outer').fillna(0)
    df_union['rate'] = df_union.apply(lambda row: (row['optimized'] - row['previous']) / row['previous'] if row['previous'] != 0 else 'n/a', axis=1)
    mask_ok = df_union['previous'].abs() >= 0.02
    numeric_rates = pd.to_numeric(df_union.loc[mask_ok, 'rate'], errors='coerce').abs()
    maxDiff = numeric_rates.max(skipna=True)
    diff_file = f"{config['output'][:-4]}_{rebalance_date}_gamma={gamma:.4f}_maxDiff={maxDiff:.4f}_lambda={lam:.4f}.csv"
    df_union.to_csv(diff_file, index=False)
    # print(f"Change rates saved to {diff_file}")

    # build and return Portfolio
    returns_df = pd.DataFrame(test_data.cpu().numpy(), index=pd.to_datetime(test_dates), columns=assets)
    return Portfolio(daily_returns=returns_df, weights=optimized.tolist(), name=f"Optimized {rebalance_date}")


def weights_to_optimized_pf_abs(rebalance_date: str,
                                train_data: torch.Tensor,
                                test_data: torch.Tensor,
                                test_dates: np.ndarray,
                                assets: list,
                                config: dict,
                                weights_dir: str,
                                output_dir: str,
                                previous_pf: Portfolio,
                                removed: list,
                                balance: float,                                
                                device: str = "cpu",
                                max_change: float = 0.1,
                                lambda_score: float = 5.0,
                                max_iter: int = 5000,
                                ) -> Portfolio:


    # Load setting
    validation_days = config["validation_days"]
    ratio_SR_return = config["SR_weight"]

    # load prediction
    pattern = os.path.join(weights_dir, f"*{rebalance_date}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No prediction file for date {rebalance_date}: {pattern}")
    pred_tensor = load_holding_weights(files[0], assets, device=device)
    prediction = pred_tensor.squeeze(0)[:len(assets)].cpu().numpy()

    # compute prediction's score
    sub_train = train_data[-validation_days:, :]
    w_pred = torch.tensor(prediction, dtype=torch.float, device=device)
    pct_pred = w_pred @ sub_train.T
    ret_pred = torch.prod(1 + pct_pred).item()
    sr_pred = sharpe_ratio(pct_pred, validation_days, risk_free_rate=0.0)
    pred_score = ratio_SR_return * sr_pred + (1 - ratio_SR_return) * ret_pred
    pred_score *= balance

    # if no previous_pf, directly output prediction
    weights_list = prediction.tolist()
    df_w = pd.DataFrame({'weight': weights_list}, index=assets)
    df_w = df_w[df_w['weight'] != 0.0]
    df_w = df_w.reindex(df_w['weight'].abs().sort_values(ascending=False).index)
    reb_date = pd.to_datetime(test_dates[0]).strftime('%Y-%m-%d')
    outfile = f"{config['output'][:-4]}_{reb_date}_optimized.csv"
    # df_w.to_csv(outfile, index=False, header=False)
    returns_df = pd.DataFrame(test_data.cpu().numpy(),
                              index=pd.to_datetime(test_dates),
                              columns=assets)
    pred_pf = Portfolio(daily_returns=returns_df,
                     weights=weights_list,
                     name=f"Optimized {reb_date}")
                     
    if previous_pf is None:
        return pred_pf

    # Prepare previous holdings
    holding = np.array(previous_pf.weights, dtype=float)
    # removed_mask = np.array([a in removed for a in assets], dtype=bool)
    pred_mask = prediction != 0
    hold_mask = holding    != 0
    removed_mask = (np.array([a in removed for a in assets], dtype=bool) | (~pred_mask & ~hold_mask))
    
    # score function
    def score_fn(x):
        xt = torch.tensor(x, dtype=torch.float, device=device)
        pct = xt @ sub_train.T
        ret = torch.prod(1 + pct).item()
        sr = sharpe_ratio(pct, validation_days, risk_free_rate=0.0)
        return ratio_SR_return * sr + (1 - ratio_SR_return) * ret

    # objective: distance + score penalty
    def objective(x):
        dev = np.sum((x - prediction)**2)
        sc = score_fn(x)
        sc *= balance
        pen = max(0.0, pred_score - sc)
        return dev + lambda_score * pen

    # sum(x)=1 constraint
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)


    # Initialize current lambda_score and trial counter
    ls_current = lambda_score
    trial = 0

    while True:
        # 1) Relax max_change until bounds become feasible
        while True:
            lb = ub = 0.0
            for i, h in enumerate(holding):
                if removed_mask[i]:
                    low, high = 0.0, 0.0
                else:
                    if h != 0.0:
                        low = max(0.0, h - max_change)
                        high = min(1.0, h + max_change)
                    else:
                        low, high = 0.0, 0.1
                lb += low
                ub += high
            # if sum(lower bounds) <= 1 <= sum(upper bounds), feasible
            if lb <= 1.0 <= ub:
                break
            # otherwise increase max_change by 0.01
            max_change += 0.01
            trial += 1
            print(f"[Debug] Increased max_change to {max_change:.2f} (sum_low={lb:.4f}, sum_up={ub:.4f})")
            if trial > 100:
                raise RuntimeError(
                    f"Could not find a feasible max_change (last tried {max_change:.2f})"
                )

        # 2) Build bounds and initial guess x0 based on current max_change
        bounds, x0 = [], []
        for i, pred in enumerate(prediction):
            if removed_mask[i]:
                bounds.append((0.0, 0.0))
                x0.append(0.0)
            else:
                h = holding[i]
                if h != 0.0:
                    low = max(0.0, h - max_change)
                    high = min(1.0, h + max_change)
                else:
                    low, high = 0.0, 0.1
                bounds.append((low, high))
                x0.append(np.clip(pred, low, high))
        x0 = np.array(x0, dtype=float)

        # 3) Define objective using current lambda_score
        def objective(x):
            # squared deviation from prediction
            dev = np.sum((x - prediction)**2)
            # compute score and penalty if below pred_score
            sc = score_fn(x)
            pen = max(0.0, pred_score - sc)
            return dev + ls_current * pen

        # 4) Attempt optimization with SLSQP
        result = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},),
            options={'ftol': 1e-7, 'maxiter': max_iter}
        )

        if result.success:
            optimized = result.x
            break

        msg = result.message.lower()
        # If line search failure, reduce lambda_score by 25%
        if 'positive directional derivative' in msg:
            old_ls = ls_current
            ls_current *= 0.75
            print(f"[Debug] '{msg.split(':')[0]}', lambda_score: {old_ls:.4f} -> {ls_current:.4f}")
            continue

        # If bounds incompatible, increase max_change by 0.01
        if 'inequality constraints incompatible' or 'singular matrix' in msg:
            old_mc = max_change
            max_change += 0.01
            if max_change > 1:
                return pred_pf
            print(f"[Debug] '{msg.split(':')[0]}', max_change: {old_mc:.2f} -> {max_change:.2f}")
            continue

        # Otherwise, raise error
        raise RuntimeError(f"Optimization failed: {result.message}")



    # Save optimized weights CSV
    os.makedirs(output_dir, exist_ok=True)
    
    df_opt = pd.DataFrame({'asset': assets, 'optimized': optimized})
    opt_filename = f"{config['output'][:-4]}_{rebalance_date}_optimized.csv"
    df_opt_nonzero = df_opt[df_opt['optimized'] != 0.0]
    df_opt_sorted = df_opt_nonzero.reindex(
        df_opt_nonzero['optimized'].abs().sort_values(ascending=False).index)
    opt_file = os.path.join(output_dir, opt_filename)
    df_opt_sorted.to_csv(opt_file, index=False, header=False)

    scoreDiff = pred_score - score_fn(optimized)*balance
    
    # Save change-rate CSV
    df_prev = pd.DataFrame({'asset': assets, 'previous': holding})
    df_union = pd.merge(df_prev, df_opt, on='asset', how='outer').fillna(0)
    df_union['abs_change'] = df_union.apply(
        lambda r: abs(r['optimized'] - r['previous'])
                  if r['previous'] != 0 else np.nan,
        axis=1
    )
    abs_changes = df_union.loc[df_union['previous'].abs() >= 0.02, 'abs_change']
    maxDiff = abs_changes.max(skipna=True)

    diff_filename = f"{rebalance_date}_maxChange={max_change:.2f}_maxDiff={maxDiff:.4f}_scoreDiff={scoreDiff:.0f}.csv"
    diff_file = os.path.join(output_dir, diff_filename)
    df_union.to_csv(diff_file, index=False)

    # Return Portfolio
    returns_df = pd.DataFrame(test_data.cpu().numpy(),
                              index=pd.to_datetime(test_dates),
                              columns=assets)
    return Portfolio(daily_returns=returns_df,
                     weights=optimized.tolist(),
                     name=f"Optimized {rebalance_date}")


def prediction_weights_to_optimized_pf_abs(
                                train_data: torch.Tensor,
                                assets: list,
                                config: dict,
                                weights_dir: str,
                                output_dir: str,
                                removed: list,
                                balance: float = 100000,                                
                                device: str = "cpu",
                                max_change: float = 0.1,
                                lambda_score: float = 5.0,
                                max_iter: int = 5000,
                                ):


    # Load setting
    validation_days = config["validation_days"]
    ratio_SR_return = config["SR_weight"]

    # locate prediction CSV and load predicted weights
    rebalance_date = "prediction"
    pattern = os.path.join(weights_dir, f"*{rebalance_date}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No prediction file for date {rebalance_date}: {pattern}")
    csv_path = files[0]
    pred_tensor = load_holding_weights(csv_path, assets, device=device)
    prediction = pred_tensor.squeeze(0)[:len(assets)].cpu().numpy()

    # compute prediction score
    sub_train = train_data[-validation_days:, :]  # shape (D, N)
    # portfolio daily pct: prediction @ sub_train^T
    w_pred = torch.tensor(prediction, dtype=torch.float, device=device)
    portfolio_daily_pct_pred = w_pred @ sub_train.T  # shape (D,)
    portfolio_return_pred = torch.prod(1 + portfolio_daily_pct_pred).item()
    SR_pred = sharpe_ratio(portfolio_daily_pct_pred, validation_days, risk_free_rate=0.0)
    pred_score = ratio_SR_return * SR_pred + (1 - ratio_SR_return) * portfolio_return_pred
    pred_score *= balance

    # prepare previous holdings and masks
    holding_tensor = load_holding_weights(config["holding"], assets, device=device)
    holding = holding_tensor.squeeze(0)[:len(assets)].cpu().numpy()
    # holding = np.array(holding_weights, dtype=float)
    # removed_mask = np.array([a in removed for a in assets], dtype=bool)
    pred_mask = prediction != 0
    hold_mask = holding    != 0
    removed_mask = (np.array([a in removed for a in assets], dtype=bool) | (~pred_mask & ~hold_mask))
        
    # score function
    def score_fn(x):
        xt = torch.tensor(x, dtype=torch.float, device=device)
        pct = xt @ sub_train.T
        ret = torch.prod(1 + pct).item()
        sr = sharpe_ratio(pct, validation_days, risk_free_rate=0.0)
        return ratio_SR_return * sr + (1 - ratio_SR_return) * ret

    # objective: distance + score penalty
    def objective(x):
        dev = np.sum((x - prediction)**2)
        sc = score_fn(x)
        sc *= balance
        pen = max(0.0, pred_score - sc)
        return dev + lambda_score * pen

    # sum(x)=1 constraint
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)


    # Initialize current lambda_score and trial counter
    ls_current = lambda_score
    trial = 0

    while True:
        # 1) Relax max_change until bounds become feasible
        while True:
            lb = ub = 0.0
            for i, h in enumerate(holding):
                if removed_mask[i]:
                    low, high = 0.0, 0.0
                else:
                    if h != 0.0:
                        low = max(0.0, h - max_change)
                        high = min(1.0, h + max_change)
                    else:
                        low, high = 0.0, 0.1
                lb += low
                ub += high
            # if sum(lower bounds) <= 1 <= sum(upper bounds), feasible
            if lb <= 1.0 <= ub:
                break
            # otherwise increase max_change by 0.01
            max_change += 0.01
            trial += 1
            print(f"[Debug] Increased max_change to {max_change:.2f} (sum_low={lb:.4f}, sum_up={ub:.4f})")
            if trial > 100:
                raise RuntimeError(
                    f"Could not find a feasible max_change (last tried {max_change:.2f})"
                )

        # 2) Build bounds and initial guess x0 based on current max_change
        bounds, x0 = [], []
        for i, pred in enumerate(prediction):
            if removed_mask[i]:
                bounds.append((0.0, 0.0))
                x0.append(0.0)
            else:
                h = holding[i]
                if h != 0.0:
                    low = max(0.0, h - max_change)
                    high = min(1.0, h + max_change)
                else:
                    low, high = 0.0, 0.1
                bounds.append((low, high))
                x0.append(np.clip(pred, low, high))
        x0 = np.array(x0, dtype=float)

        # 3) Define objective using current lambda_score
        def objective(x):
            # squared deviation from prediction
            dev = np.sum((x - prediction)**2)
            # compute score and penalty if below pred_score
            sc = score_fn(x)
            pen = max(0.0, pred_score - sc)
            return dev + ls_current * pen

        # 4) Attempt optimization with SLSQP
        result = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},),
            options={'ftol': 1e-7, 'maxiter': max_iter}
        )

        if result.success:
            optimized = result.x
            break

        msg = result.message.lower()
        # If line search failure, reduce lambda_score by 25%
        if 'positive directional derivative' in msg:
            old_ls = ls_current
            ls_current *= 0.75
            print(f"[Debug] '{msg.split(':')[0]}', lambda_score: {old_ls:.4f} -> {ls_current:.4f}")
            continue

        # If bounds incompatible, increase max_change by 0.01
        if 'inequality constraints incompatible' or 'singular matrix' in msg:
            old_mc = max_change
            max_change += 0.01
            if max_change > 1:
                return pred_pf
            print(f"[Debug] '{msg.split(':')[0]}', max_change: {old_mc:.2f} -> {max_change:.2f}")
            continue

        # Otherwise, raise error
        raise RuntimeError(f"Optimization failed: {result.message}")



    # Save optimized weights CSV
    os.makedirs(output_dir, exist_ok=True)
    
    df_opt = pd.DataFrame({'asset': assets, 'optimized': optimized})
    opt_filename = f"{config['output'][:-4]}_{rebalance_date}_optimized.csv"
    df_opt_nonzero = df_opt[df_opt['optimized'] != 0.0]
    df_opt_sorted = df_opt_nonzero.reindex(
        df_opt_nonzero['optimized'].abs().sort_values(ascending=False).index)
    opt_file = os.path.join(output_dir, opt_filename)
    df_opt_sorted.to_csv(opt_file, index=False, header=False)

    scoreDiff = pred_score - score_fn(optimized)*balance
    
    # Save change-rate CSV
    df_prev = pd.DataFrame({'asset': assets, 'previous': holding})
    df_union = pd.merge(df_prev, df_opt, on='asset', how='outer').fillna(0)
    df_union['abs_change'] = df_union.apply(
        lambda r: abs(r['optimized'] - r['previous'])
                  if r['previous'] != 0 else np.nan,
        axis=1
    )
    abs_changes = df_union.loc[df_union['previous'].abs() >= 0.02, 'abs_change']
    maxDiff = abs_changes.max(skipna=True)

    diff_filename = f"{rebalance_date}_maxChange={max_change:.2f}_maxDiff={maxDiff:.4f}_scoreDiff={scoreDiff:.0f}.csv"
    diff_file = os.path.join(output_dir, diff_filename)
    df_union.to_csv(diff_file, index=False)


'''                     
def weights_to_optimized_pf_abs_slack(rebalance_date: str,
                                       train_data: torch.Tensor,
                                       test_data: torch.Tensor,
                                       test_dates: np.ndarray,
                                       assets: list,
                                       config: dict,
                                       weights_dir: str,
                                       output_dir: str,
                                       previous_pf: Portfolio,
                                       removed: list,
                                       device: str = "cpu",
                                       max_change: float = 0.1,
                                       lambda_score: float = 5.0,
                                       lambda_slack: float = 1.0,  # new
                                       max_iter: int = 5000
                                       ) -> Portfolio:
    import os
    from scipy.optimize import minimize

    validation_days = config["validation_days"]
    ratio_SR_return = config["SR_weight"]

    pattern = os.path.join(weights_dir, f"*{rebalance_date}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No prediction file for date {rebalance_date}")
    prediction = load_holding_weights(files[0], assets, device=device).squeeze(0)[:len(assets)].cpu().numpy()

    sub_train = train_data[-validation_days:, :]
    w_pred = torch.tensor(prediction, dtype=torch.float, device=device)
    pct_pred = w_pred @ sub_train.T
    ret_pred = torch.prod(1 + pct_pred).item()
    sr_pred = sharpe_ratio(pct_pred, validation_days, risk_free_rate=0.0)
    pred_score = ratio_SR_return * sr_pred + (1 - ratio_SR_return) * ret_pred

    if previous_pf is None:
        weights_list = prediction.tolist()
        returns_df = pd.DataFrame(test_data.cpu().numpy(), index=pd.to_datetime(test_dates), columns=assets)
        return Portfolio(daily_returns=returns_df, weights=weights_list, name=f"Optimized {rebalance_date}")

    holding = np.array(previous_pf.weights, dtype=float)
    removed_mask = np.array([a in removed for a in assets], dtype=bool)
    n = len(assets)

    def score_fn(x):
        xt = torch.tensor(x, dtype=torch.float, device=device)
        pct = xt @ sub_train.T
        ret = torch.prod(1 + pct).item()
        sr = sharpe_ratio(pct, validation_days, risk_free_rate=0.0)
        return ratio_SR_return * sr + (1 - ratio_SR_return) * ret

    def objective(full_x):
        x = full_x[:n]
        s = full_x[n:]
        dev = np.sum((x - prediction)**2)
        slack_penalty = np.sum(s)
        sc = score_fn(x)
        score_penalty = max(0.0, pred_score - sc)
        return dev + lambda_score * score_penalty + lambda_slack * slack_penalty

    # constraints: sum(x) = 1
    def sum_constraint(full_x):
        return np.sum(full_x[:n]) - 1.0

    # constraints: |x_i - h_i| - max_change - s_i ? 0
    ineq_constraints = []
    for i in range(n):
        if removed_mask[i]:
            continue
        def ineq_fn_factory(i):
            return lambda full_x, i=i: max_change + full_x[n + i] - abs(full_x[i] - holding[i])
        ineq_constraints.append({'type': 'ineq', 'fun': ineq_fn_factory(i)})

    constraints = [{'type': 'eq', 'fun': sum_constraint}] + ineq_constraints

    bounds = []
    for i in range(n):
        if removed_mask[i]:
            bounds.append((0.0, 0.0))  # x_i
        else:
            bounds.append((0.0, 1.0))  # x_i
    bounds += [(0.0, None)] * n  # slack bounds for s_i

    x0_x = np.clip(prediction, 0.0, 1.0)
    x0_x[removed_mask] = 0.0
    x0_s = np.zeros(n)
    x0 = np.concatenate([x0_x, x0_s])

    result = minimize(fun=objective,
                      x0=x0,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'ftol': 1e-7, 'maxiter': max_iter})

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    optimized = result.x[:n]
    slack_used = result.x[n:]

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df_opt = pd.DataFrame({'asset': assets, 'optimized': optimized, 'slack': slack_used})

    df_prev = pd.DataFrame({'asset': assets, 'previous': holding})
    df_union = pd.merge(df_prev, df_opt, on='asset', how='outer').fillna(0)
    df_union['abs_change'] = df_union.apply(
        lambda r: abs(r['optimized'] - r['previous']) if r['previous'] != 0 else np.nan, axis=1)
    abs_changes = df_union.loc[df_union['previous'].abs() >= 0.02, 'abs_change']
    maxDiff = abs_changes.max(skipna=True)

    diff_filename = (
        f"{config['output'][:-4]}_{rebalance_date}"
        f"_maxChange={max_change:.2f}_maxDiff={maxDiff:.4f}.csv"
    )
    diff_file = os.path.join(output_dir, diff_filename)
    df_union.to_csv(diff_file, index=False)

    returns_df = pd.DataFrame(test_data.cpu().numpy(),
                              index=pd.to_datetime(test_dates),
                              columns=assets)
    return Portfolio(daily_returns=returns_df,
                     weights=optimized.tolist(),
                     name=f"Optimized {rebalance_date}")
'''                     


def prediction_weights_to_optimized_pf(train_data: torch.Tensor,
                                        assets: list,
                                        config: dict,
                                        weights_dir: str,
                                        removed: list,
                                        device: str = "cpu",
                                        lambda_penalty: float = 20.0,
                                        min_lambda: float = 0.0001,
                                        initial_gamma: float = 0.9,
                                        min_gamma: float = 0.0001,
                                        max_iter: int = 5000):
    """
    Optimize predicted weights with turnover penalty and enforce a minimum score relative to predictions.

    Adds a hard constraint: optimized score >= gamma * prediction_score.
    Returns a Portfolio built from optimized weights, and saves CSVs.
    """
    # configuration
    ratio_SR_return = config["SR_weight"]
    validation_days = config["validation_days"]

    # locate prediction CSV and load predicted weights
    rebalance_date = "prediction"
    pattern = os.path.join(weights_dir, f"*{rebalance_date}*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No prediction file for date {rebalance_date}: {pattern}")
    csv_path = files[0]
    pred_tensor = load_holding_weights(csv_path, assets, device=device)
    prediction = pred_tensor.squeeze(0)[:len(assets)].cpu().numpy()

    # compute prediction score
    sub_train = train_data[-validation_days:, :]  # shape (D, N)
    # portfolio daily pct: prediction @ sub_train^T
    w_pred = torch.tensor(prediction, dtype=torch.float, device=device)
    portfolio_daily_pct_pred = w_pred @ sub_train.T  # shape (D,)
    portfolio_return_pred = torch.prod(1 + portfolio_daily_pct_pred).item()
    SR_pred = sharpe_ratio(portfolio_daily_pct_pred, validation_days, risk_free_rate=0.0)
    pred_score = ratio_SR_return * SR_pred + (1 - ratio_SR_return) * portfolio_return_pred

    # prepare previous holdings and masks
    holding_tensor = load_holding_weights(config["holding"], assets, device=device)
    holding = holding_tensor.squeeze(0)[:len(assets)].cpu().numpy()
    # holding = np.array(holding_weights, dtype=float)
    removed_mask = np.array([a in removed for a in assets], dtype=bool)

    # set up bounds, initial guess, penalty mask
    bounds, x0, penalized_mask = [], [], []
    for i in range(len(assets)):
        if removed_mask[i]:
            bounds.append((0.0, 0.0)); x0.append(0.0); penalized_mask.append(False)
        elif holding[i] == 0.0:
            bounds.append((0.0, 1.0)); x0.append(prediction[i]); penalized_mask.append(False)
        else:
            bounds.append((0.0, 1.0)); x0.append(prediction[i]); penalized_mask.append(True)
    x0 = np.array(x0, dtype=float)

    # objective with turnover penalty
    def objective(x, lam):
        dev = np.sum((x - prediction) ** 2)
        pen = 0.0
        for j in range(len(x)):
            if penalized_mask[j]:
                delta = abs(x[j] - holding[j])
                thr = 0.5 * holding[j]
                pen += delta if delta < thr else delta**2
        return dev + lam * pen

    # score function for optimized weights
    def score_fn(x):
        x_t = torch.tensor(x, dtype=torch.float, device=device)
        pct = x_t @ sub_train.T
        ret = torch.prod(1 + pct).item()
        sr = sharpe_ratio(pct, validation_days, risk_free_rate=0.0)
        return ratio_SR_return * sr + (1 - ratio_SR_return) * ret

    # constraints: sum(x)==1 and score_fn(x) >= gamma * pred_score
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: score_fn(x) - initial_gamma * pred_score}
    ]

    # solve with retry on iteration limit
    lam = lambda_penalty
    gamma = initial_gamma
    while True:
        result = minimize(
            fun=objective,
            x0=x0,
            args=(lam,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol':1e-9, 'maxiter':max_iter}
        )
        if result.success:
            break

        msg = result.message.lower()

        # 1) iteration limit, line-search, or LSQ
        if (('iteration limit' in msg)
            or ('positive directional derivative' in msg)
            or ('more than 3*n iterations' in msg)) \
           and lam > min_lambda:

            old_lam = lam
            lam *= 2.0/3.0
            print(f"Issue '{msg.split(':')[0]}', reducing lambda from {old_lam} to {lam}")
            continue

        # 2) inequality constraints
        if 'inequality constraints incompatible' in msg and gamma > min_gamma:
            old_gamma = gamma
            gamma *= 0.9
            print(f"Issue '{msg.split(':')[0]}', reducing gamma from {old_gamma} to {gamma}")
            constraints[1] = {
                'type': 'ineq',
                'fun': lambda x: score_fn(x) - gamma * pred_score
            }
            continue

        raise RuntimeError(f"Optimization failed at lambda={lam}, gamma={gamma}: {result.message}")
    optimized = result.x

    # save optimized weights
    df_opt = pd.DataFrame({'asset': assets, 'optimized': optimized})
    '''
    df_opt_nonzero = df_opt[df_opt['optimized'] != 0.0]
    df_opt_sorted = df_opt_nonzero.reindex(df_opt_nonzero['optimized'].abs().sort_values(ascending=False).index)
    opt_file = f"{config['output'][:-4]}_{rebalance_date}_optimized.csv"
    df_opt_sorted.to_csv(opt_file, index=False, header=False)
    # print(f"Optimized weights saved to {opt_file}")
    '''

    # save change-rate CSV
    df_prev = pd.DataFrame({'asset': assets, 'previous': holding})
    df_union = pd.merge(df_prev, df_opt, on='asset', how='outer').fillna(0)
    df_union['rate'] = df_union.apply(lambda row: (row['optimized'] - row['previous']) / row['previous'] if row['previous'] != 0 else 'n/a', axis=1)
    mask_ok = df_union['previous'].abs() >= 0.02
    numeric_rates = pd.to_numeric(df_union.loc[mask_ok, 'rate'], errors='coerce').abs()
    maxDiff = numeric_rates.max(skipna=True)
    diff_file = f"{config['output'][:-4]}_{rebalance_date}_gamma={gamma:.4f}_maxDiff={maxDiff:.4f}_lambda={lam:.4f}.csv"
    df_union.to_csv(diff_file, index=False)
    print(f"Change rates saved to {diff_file}")


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
    return float(annual_sharpe_ratio)


# Adaptive SR_weight


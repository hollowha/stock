import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from load_config import config

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = 5/3)
        if hasattr(m, 'bias') and m.bias is not None: m.bias.data.zero_()

class LSTMModule(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 1, num_layers = 2):
        super(LSTMModule, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.h = torch.zeros(num_layers, 1, hidden_size, requires_grad=True)
        self.c = torch.zeros(num_layers, 1, hidden_size, requires_grad=True)
    def forward(self, x):
        self.rnn.flatten_parameters()
        out, (h_end, c_end) = self.rnn(x, (self.h, self.c))
        self.h.data = h_end.data
        self.c.data = c_end.data
        return out[:,-1, :].flatten()

class Extractor(nn.Module):
    def __init__(self, latent_dim, ks = 5):
        super(Extractor, self).__init__()
        self.conv = nn.Conv1d(config["noise"], latent_dim,
            bias = False, kernel_size = ks, padding = (ks // 2) + 1)
        self.conv.weight.data.normal_(0, 0.01)
        self.activation = nn.Sequential(nn.BatchNorm1d(
            latent_dim, track_running_stats = False), nn.Mish())
        self.gap = nn.AvgPool1d(kernel_size = config["batch"], padding = 1)
        self.rnn = LSTMModule(hidden_size = latent_dim)
    def forward(self, x):
        y = x.unsqueeze(0).permute(0, 2, 1)
        y = self.rnn(self.gap(self.activation(self.conv(y))))
        return torch.cat([x, y.repeat(config["batch"], 1)], dim = 1)

class Generator(nn.Module):
    def __init__(self, noise_dim = 0, output_dim = 0):
        super(Generator, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Tanh()]
        self.model = nn.Sequential(
            *block(noise_dim+config["cnndim"], 512), *block(512, config["batch"]), nn.Linear(config["batch"], output_dim))
        init_weights(self)
        self.extract = Extractor(config["cnndim"])
        self.std_weight = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        mu = self.model(self.extract(x))
        return mu, mu + (self.std_weight * torch.randn_like(mu))

def corr(X, eps=1e-8):
    D = X.shape[-1]
    std = torch.std(X, dim = -1).unsqueeze(-1)
    mean = torch.mean(X, dim = -1).unsqueeze(-1)
    X = (X - mean) / (std + eps)
    return 1/(D-1) * X @ X.transpose(-1, -2)


def loss_fn(weights, data, index, device = "cpu", train=False):
    diff = weights.matmul(data.T) - index
    # if not train: return diff.clamp(max=0.0).pow(2).mean(dim=1)
    if not train:
        # For validation, return simplified loss_1 (no time weighting, no max_corr)
        total_diff = (-diff).sum(dim=1) * 20  # Simplified total difference
        diff_exp = torch.exp(total_diff)  # Exponentially weighted difference
        print(f"Validation total_diff = {total_diff.mean()}, exp = {diff_exp.mean()}")
        return diff_exp

    #weight the training returns by recency in performance calculation
    ww = torch.arange(1, diff.shape[1]+1).pow(0.5).to(device)
    ww /= ww.sum()

    #the performance is calculated from randomly selected 25% returns
    diff = nn.functional.dropout(diff, p = 0.75)

    #minimize the maximum correlation in-between portfolio candidates
    corr_max = corr(diff).fill_diagonal_(0.0).max(dim = 1)[0]
    # print(corr_max)
    return (diff.clamp(max = 0.0).pow(2) * ww).sum(dim = 1) * corr_max


def loss_combined(weights, data, index, weight_origin=1, weight_1=0, weight_entropy=0, k_ratio=1, device="cpu", train=False):
    """
    Combines loss_orig and loss_1 for training, and returns simplified loss_1 for validation.
    """
    diff = weights.matmul(data.T) - index

    if not train:
        # For validation, return simplified loss_1 (no time weighting, no max_corr)
        error = diff.clamp(max=0.0).pow(2).mean(dim=1)
        total_diff = (-diff).sum(dim=1) * 20  # Simplified total difference
        diff_exp = torch.exp(total_diff)  # Exponentially weighted difference
        print(f"Validation total_diff = {total_diff.mean()}, exp = {diff_exp.mean()}")
        return weight_origin * error + weight_1 * diff_exp

    # Apply dropout for robustness in training
    diff = nn.functional.dropout(diff, p=0.75)

    # Weight the training returns by recency in performance calculation
    ww = torch.arange(1, diff.shape[1] + 1).pow(0.5).to(device)

    # Calculate the maximum correlation between sub-portfolios
    corr_max = corr(diff).fill_diagonal_(0.0).max(dim=1)[0] + 1

    # loss_orig: Weighted squared error
    loss_orig = (diff.clamp(max=0.0).pow(2) * (ww / ww.sum())).sum(dim=1)

    # loss_1: Exponentially weighted total difference with correlation
    total_diff = (-diff * (ww / ww.sum())).sum(dim=1) * 20
    diff_exp = torch.exp(total_diff)

    # Log debug information
    # print(f"loss_orig = {loss_orig.mean()}, total_diff = {total_diff.mean()}, exp = {diff_exp.mean()}, corr = {corr_max.mean()}")

    # Combine losses and scale with corr_max
    combined_loss = (weight_origin * loss_orig + weight_1 * diff_exp) * corr_max
    
    # Top-k entropy regularization
    k = max(1, int(weights.shape[1] * k_ratio))  # Ensure at least 1 dimension is selected
    topk_values, _ = torch.topk(weights, k, dim=1)  # Get top-k largest weights
    
    # Compute how much top-k contributes to total weights
    topk_ratio = topk_values.sum(dim=1) / weights.sum(dim=1)  # Normalize by total weights
    
    # Compute dynamic weight_entropy
    weight_entropy_dynamic = weight_entropy * (1 - topk_ratio) / 0.3  # Adjust scale
    weight_entropy_dynamic = torch.clamp(weight_entropy_dynamic, 0, weight_entropy)
    
    # Compute entropy loss with temperature scaling
    temperature = 1.0 + (1.0 - topk_ratio).unsqueeze(1) * 2.0  # Adjust temperature dynamically
    weights_norm = torch.softmax(weights / temperature, dim=1)  # Softmax with adaptive temperature
    
    # Entropy regularization loss
    entropy_loss = -(weights_norm * torch.log(weights_norm + 1e-8)).sum(dim=1)

    
    # Add entropy-based regularization (higher entropy means more evenly spread weights)
    return combined_loss - weight_entropy_dynamic * entropy_loss  # Subtracting because we want to maximize entropy
    


def loss_combined_holding(weights, data, index, holding_weights,
                          weight_origin=1, weight_1=0, weight_entropy=0, k_ratio=1, 
                          weight_diff=1, device="cpu", train=False):
    """
    Combines loss_orig and loss_1 and adds an extra penalty based on the difference
    between the model's weights and the holding_weights. This version reorders the model's weights
    using torch operations to preserve the gradient.
    """
    # Compute a permutation index that maps assets_sorted to the corresponding positions in assets_original.
    # perm = [assets_original.index(a) for a in assets_sorted]
    # perm = torch.tensor(perm, dtype=torch.long, device=weights.device)
    
    # Reorder the weights tensor using the permutation index.
    # weights_ordered = weights[:, perm]
    
    # Compute diff: (weights_ordered * data.T) - index
    diff = weights.matmul(data.T) - index
    
    if not train:
        error = diff.clamp(max=0.0).pow(2).mean(dim=1)
        total_diff = (-diff).sum(dim=1) * 20
        diff_exp = torch.exp(total_diff)
        print(f"Validation total_diff = {total_diff.mean()}, exp = {diff_exp.mean()}")
        return weight_origin * error + weight_1 * diff_exp

    # Apply dropout for training robustness.
    diff = nn.functional.dropout(diff, p=0.75)
    ww = torch.arange(1, diff.shape[1] + 1).pow(0.5).to(device)
    
    # Compute maximum correlation between sub-portfolios (assumes corr function is defined elsewhere).
    corr_max = corr(diff).fill_diagonal_(0.0).max(dim=1)[0] + 1
    loss_orig = (diff.clamp(max=0.0).pow(2) * (ww / ww.sum())).sum(dim=1)
    total_diff = (-diff * (ww / ww.sum())).sum(dim=1) * 20
    diff_exp = torch.exp(total_diff)
    combined_loss = (weight_origin * loss_orig + weight_1 * diff_exp) * corr_max

    # print(f"loss_orig = {loss_orig}, diff_exp = {diff_exp}, combined_loss = {combined_loss}")

    # Top-k entropy regularization.
    k = max(1, int(weights.shape[1] * k_ratio))
    topk_values, _ = torch.topk(weights, k, dim=1)
    topk_ratio = topk_values.sum(dim=1) / weights.sum(dim=1)
    weight_entropy_dynamic = weight_entropy * (1 - topk_ratio) / 0.3
    weight_entropy_dynamic = torch.clamp(weight_entropy_dynamic, 0, weight_entropy)
    temperature = 1.0 + (1.0 - topk_ratio).unsqueeze(1) * 2.0
    weights_norm = torch.softmax(weights / temperature, dim=1)
    entropy_loss = -(weights_norm * torch.log(weights_norm + 1e-8)).sum(dim=1)

    # Compute element-wise difference between weights_ordered and holding_weights.
    batch_size, num_assets = weights.shape
    '''
    # overall diff
    diff_assets = torch.abs(weights - holding_weights[:, :num_assets]).sum(dim=1)
    diff_extras = torch.abs(holding_weights[:, num_assets:]).sum(dim=1)
    total_diff = diff_assets + diff_extras  # shape: [batch_size]
        
    margin = 0.5
    # alpha  = 1.0    # Suggest: 0.5~2
    # exp_slope = 10.0   # Suggest: 10~20
    # penalty_inside = 0.5 * alpha * total_diff**2
    # penalty_outside = torch.exp(exp_slope * (total_diff - margin)) - 1.0
    # holding_penalty = torch.where(total_diff <= margin, penalty_inside, penalty_outside)
    beta = 10.0  
    penalty = torch.nn.functional.softplus(beta * (total_diff - margin)) / beta
    holding_penalty = penalty
    '''
    # per diff
    margin = 0.5
    beta = 10.0
    
    holding_assets = holding_weights[:, :num_assets]
    # holding_extras = holding_weights[:, num_assets:]
    weights_assets = weights  # shape: [batch_size, num_assets]
    
    diff_assets = torch.abs(weights_assets - holding_assets)
    allowed_range = margin * holding_assets.abs()
    allowed_range = torch.where(holding_assets == 0, torch.full_like(allowed_range, float('inf')), allowed_range)    
    violations_assets = diff_assets - allowed_range
    violations_assets = torch.where(violations_assets > 0, violations_assets, torch.zeros_like(violations_assets))    
    penalty_assets = torch.nn.functional.softplus(beta * violations_assets) / beta


    '''
    target_extras = torch.zeros_like(holding_extras)
    diff_extras = torch.abs(target_extras - holding_extras)
    mask_extras = (holding_extras != 0)
    violations_extras = diff_extras - margin
    violations_extras = torch.where(mask_extras & (violations_extras > 0), violations_extras, torch.zeros_like(violations_extras))
    penalty_extras = torch.nn.functional.softplus(beta * violations_extras) / beta
    '''
    
    # holding_penalty = penalty_assets.sum(dim=1) + penalty_extras.sum(dim=1)
    holding_penalty = penalty_assets.sum(dim=1)

    final_loss = combined_loss + weight_diff * holding_penalty - weight_entropy_dynamic * entropy_loss
    
    
    return final_loss
    
def loss_combined_holding_Lagrange(weights, data, index, holding_weights,
                          weight_origin=1, weight_1=0, weight_entropy=0, k_ratio=1, 
                          device="cpu", train=False, lambda_diff=None):
    """
    Combines loss_orig and loss_1 and adds a Lagrangian-style penalty based on the difference
    between the model's weights and the holding_weights. Lambda is learned during training.
    """
    if not train:
        diff = weights.matmul(data.T) - index
        error = diff.clamp(max=0.0).pow(2).mean(dim=1)
        total_diff = (-diff).sum(dim=1) * 20
        diff_exp = torch.exp(total_diff)
        print(f"Validation total_diff = {total_diff.mean()}, exp = {diff_exp.mean()}")
        return weight_origin * error + weight_1 * diff_exp

    # Apply dropout to the prediction error
    diff = nn.functional.dropout(weights.matmul(data.T) - index, p=0.75)
    ww = torch.arange(1, diff.shape[1] + 1).pow(0.5).to(device)

    # Compute base prediction loss
    corr_max = corr(diff).fill_diagonal_(0.0).max(dim=1)[0] + 1
    loss_orig = (diff.clamp(max=0.0).pow(2) * (ww / ww.sum())).sum(dim=1)
    total_diff = (-diff * (ww / ww.sum())).sum(dim=1) * 20
    diff_exp = torch.exp(total_diff)
    combined_loss = (weight_origin * loss_orig + weight_1 * diff_exp) * corr_max

    # Entropy regularization
    k = max(1, int(weights.shape[1] * k_ratio))
    topk_values, _ = torch.topk(weights, k, dim=1)
    topk_ratio = topk_values.sum(dim=1) / weights.sum(dim=1)
    weight_entropy_dynamic = weight_entropy * (1 - topk_ratio) / 0.3
    weight_entropy_dynamic = torch.clamp(weight_entropy_dynamic, 0, weight_entropy)
    temperature = 1.0 + (1.0 - topk_ratio).unsqueeze(1) * 2.0
    weights_norm = torch.softmax(weights / temperature, dim=1)
    entropy_loss = -(weights_norm * torch.log(weights_norm + 1e-8)).sum(dim=1)

    # Diff penalty: use ratio-based violation (Lagrangian version)
    batch_size, num_assets = weights.shape
    holding_assets = holding_weights[:, :num_assets]
    weights_assets = weights
    
    # Compute ratio difference
    diff_assets = torch.abs(weights_assets - holding_assets)
    raw_diff_ratio = diff_assets / (holding_assets.abs() + 1e-8)
    mask = holding_assets != 0
    diff_ratio = torch.where(mask, raw_diff_ratio, torch.zeros_like(raw_diff_ratio))

    # Initialize penalty as all zeros
    penalty_assets = torch.zeros_like(diff_ratio)
    
    # Light penalty: (diff_ratio - 0.2)^2 in [0.2, 0.5]
    light_mask = (diff_ratio > 0.2) & (diff_ratio <= 0.5)
    penalty_assets[light_mask] = (diff_ratio[light_mask] - 0.2) ** 2
    
    # Heavy penalty: amplify (diff_ratio - 0.5)^2 * £\ in (> 0.5)
    heavy_mask = diff_ratio > 0.5
    penalty_assets[heavy_mask] = ((diff_ratio[heavy_mask] - 0.5) ** 2) * 4

    # Apply Lagrangian penalty
    penalty_assets = lambda_diff * penalty_assets
    holding_penalty = penalty_assets.sum(dim=1)
    
    # Final loss: min_w max_lambda L(w, £f)
    final_loss = combined_loss - weight_entropy_dynamic * entropy_loss + holding_penalty

    return final_loss
   


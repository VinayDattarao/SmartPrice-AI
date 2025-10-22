import torch
import torch.nn as nn

class HybridSMAPELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha  # Weight for SMAPE component
        self.beta = beta   # Weight for MSE component
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # SMAPE loss component
        abs_diff = torch.abs(pred - target)
        abs_sum = torch.abs(pred) + torch.abs(target) + self.epsilon
        smape = 200 * torch.mean(abs_diff / abs_sum)
        
        # MSE loss component with relative scaling
        mse = torch.mean((pred - target) ** 2)
        scale = torch.mean(target ** 2) + self.epsilon
        scaled_mse = mse / scale
        
        # Combine losses with dynamic weighting
        dynamic_weight = torch.sigmoid(smape / 100.0)  # Higher SMAPE = more focus on SMAPE
        total_loss = dynamic_weight * smape + (1 - dynamic_weight) * scaled_mse
        
        return total_loss
        
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, epsilon=1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # Calculate relative error (normalized)
        rel_error = torch.abs(pred - target) / (torch.abs(target) + self.epsilon)
        
        # Calculate base SMAPE loss
        abs_diff = torch.abs(pred - target)
        abs_sum = torch.abs(pred) + torch.abs(target) + self.epsilon
        smape = 200 * (abs_diff / abs_sum)
        
        # Calculate focal weights
        focal_weight = (rel_error + self.epsilon) ** self.gamma
        
        # Calculate MSE with relative scaling
        mse = (pred - target) ** 2
        scale = target ** 2 + self.epsilon
        scaled_mse = mse / scale
        
        # Combine losses with focal weights
        total_loss = torch.mean(focal_weight * (self.alpha * smape + (1 - self.alpha) * scaled_mse))
        
        return total_loss
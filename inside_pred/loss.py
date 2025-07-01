import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # BCE with logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # p_t is the probability for the target class
        p_t = torch.sigmoid(inputs)
        p_t = targets * p_t + (1 - targets) * (1 - p_t)
        # Focal loss scaling factor
        loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

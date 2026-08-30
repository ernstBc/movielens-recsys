import torch
from torch import nn



class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        # Create a binary mask where target is not zero (observed data)
        mask = (target != 0).float()

        # Calculate the squared error 
        squared_errors = (pred - target) ** 2

        # Apply a mask to ignore zero values 
        masked_errors = squared_errors * mask
        loss_sum = masked_errors.sum()

        # Average of non zeroes values (+ epsilon to avoid division by zero)
        total_observed = mask.sum()
        loss = loss_sum / (total_observed + 1e-8)

        return loss
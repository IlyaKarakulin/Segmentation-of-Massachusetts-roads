import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseRobustDiceLoss(nn.Module):
    """
    Noise-robust dice loss
    NR-Dice = Σ|tp,q - sp,q|^γ / (Σt²p,q + Σs²p,q + ε)
    """
    def __init__(self, gamma=1.5, epsilon=1e-5):
        super(NoiseRobustDiceLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        
    def forward(self, predictions, targets):

        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        numerator = torch.sum(torch.abs(predictions - targets) ** self.gamma)
        
        denominator = (torch.sum(predictions ** 2) + 
                      torch.sum(targets ** 2) + 
                      self.epsilon)
        
        nr_dice_loss = numerator / denominator
        
        return nr_dice_loss


class MSSDMPALoss(nn.Module):
    """    
    L = LBCE + LNR-Dice
    """
    def __init__(self, gamma=1.5, epsilon=1e-5, 
                 bce_weight=1.0, dice_weight=1.0):
        super(MSSDMPALoss, self).__init__()
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.nr_dice_loss = NoiseRobustDiceLoss(gamma=gamma, epsilon=epsilon)
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, outputs, targets):
        total_loss = 0.0

        for pred_map in outputs:
            # print(pred_map.size(2), pred_map.size(3))
            # print(targets.size())

            target_map = F.interpolate(targets, size=(pred_map.size(2), pred_map.size(3)), mode='nearest')
            ms_bce = self.bce_loss(pred_map, target_map)
            ms_dice = self.nr_dice_loss(pred_map, target_map)

            ms_loss = self.bce_weight * ms_bce + self.dice_weight * ms_dice
            total_loss += ms_loss

        return total_loss

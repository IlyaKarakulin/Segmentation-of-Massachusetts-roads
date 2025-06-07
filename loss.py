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
    Multi-Scale Supervised Loss
    Ltotal = Σ(i=1 to L) L(mi, ŷi) + L(mout, ŷ)
    """
    def __init__(self, gamma=1.5, epsilon=1e-5, 
                 bce_weight=1.0, dice_weight=1.0,
                 final_weight=1.0, multiscale_weight=1.0):
        super(MSSDMPALoss, self).__init__()
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.nr_dice_loss = NoiseRobustDiceLoss(gamma=gamma, epsilon=epsilon)
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.final_weight = final_weight
        self.multiscale_weight = multiscale_weight
    
    def forward(self, outputs, targets):
        final_output = outputs[0]  # mout
        probability_maps = outputs[1:]  # [m1, m2, m3, m4]
        
        total_loss = 0.0
        
        final_bce = self.bce_loss(final_output, targets)

        final_sigmoid = torch.sigmoid(final_output) 
        final_dice = self.nr_dice_loss(final_sigmoid, targets)
        final_loss = self.bce_weight * final_bce + self.dice_weight * final_dice
        
        total_loss += self.final_weight * final_loss

        
        multiscale_loss = 0.0
        for i, pred_map in enumerate(probability_maps):
            target_size = (pred_map.size(2), pred_map.size(3))
            target_map = F.interpolate(targets, size=target_size, mode='nearest')
            
            ms_bce = self.bce_loss(pred_map, target_map)
            
            ms_sigmoid = torch.sigmoid(pred_map)
            ms_dice = self.nr_dice_loss(ms_sigmoid, target_map)
            ms_loss = self.bce_weight * ms_bce + self.dice_weight * ms_dice
            
            multiscale_loss += ms_loss

        
        total_loss += self.multiscale_weight * multiscale_loss
        
        return total_loss
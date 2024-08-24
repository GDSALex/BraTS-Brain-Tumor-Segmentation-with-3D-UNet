import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """
    Combined loss function integrating Dice loss, Cross-Entropy loss, and Focal loss.
    
    Args:
        weight (Tensor, optional): A manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`.
        size_average (bool, optional): Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. 
        alpha (float, optional): Weighting factor for Focal Loss. Default is 0.8.
        gamma (float, optional): Focusing parameter for Focal Loss. Default is 2.
        smooth (float, optional): Smoothing factor for Dice Loss. Default is 1.
    """
    def __init__(self, weight=None, size_average=True, alpha=1, gamma=2, smooth=1):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.focal_loss = FocalLoss(alpha, gamma)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        ce = self.cross_entropy_loss(inputs, targets)
        return dice + ce + focal

class DiceLoss(nn.Module):
    """
    Dice loss function to measure overlap between predicted and target masks.
    
    Args:
        smooth (float, optional): Smoothing factor to prevent division by zero. Default is 1.
    """
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = F.softmax(prediction, dim=1)
        num_classes = prediction.size(1)

        dice_loss = 0
        for i in range(num_classes):
            pred_flat = prediction[:, i].contiguous().view(-1)
            target_flat = (target == i).float().contiguous().view(-1)
            intersection = torch.sum(pred_flat * target_flat)
            dice_loss += 1 - ((2. * intersection + self.smooth) / (torch.sum(pred_flat) + torch.sum(target_flat) + self.smooth))

        return dice_loss / num_classes

class FocalLoss(nn.Module):
    """
    Focal loss function to address class imbalance by focusing on hard-to-classify examples.
    
    Args:
        alpha (float, optional): Weighting factor for the focal loss. Default is 0.8.
        gamma (float, optional): Focusing parameter for the focal loss. Default is 2.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

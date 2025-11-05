import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Loss = 1 - Dice Coefficient
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid if not already applied
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W) - logits
            target: Ground truth (B, 1, H, W) - binary 0 or 1
        """
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Flatten
        pred_prob = pred_prob.view(-1)
        target = target.view(-1)
        
        # Calculate focal loss
        bce_loss = F.binary_cross_entropy(pred_prob, target, reduction='none')
        
        # p_t is the probability of the true class
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal loss formula
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss
    Useful for handling class imbalance
    
    Args:
        pos_weight: Weight for positive class (default: None, calculated from data)
    """
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, 1, H, W) - logits
            target: Ground truth (B, 1, H, W) - binary 0 or 1
        """
        # If pos_weight not provided, calculate it from the batch
        if self.pos_weight is None:
            num_pos = target.sum()
            num_neg = target.numel() - num_pos
            if num_pos > 0:
                pos_weight = num_neg / num_pos
            else:
                pos_weight = 1.0
        else:
            pos_weight = self.pos_weight
        
        # Use BCEWithLogitsLoss which applies sigmoid internally
        loss = F.binary_cross_entropy_with_logits(
            pred, target, 
            pos_weight=torch.tensor(pos_weight, device=pred.device)
        )
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combination of BCE and Dice Loss
    Often works better than either alone
    
    Args:
        alpha: Weight for BCE loss (default: 0.5)
        beta: Weight for Dice loss (default: 0.5)
    """
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.alpha * bce_loss + self.beta * dice_loss


def get_loss_function(loss_name, **kwargs):
    """
    Get loss function by name
    
    Args:
        loss_name: Name of loss function ('bce', 'focal', 'weighted_bce', 'dice', 'combined')
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'weighted_bce':
        return WeightedBCELoss(**kwargs)
    elif loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# Test the losses
if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 4
    height, width = 256, 256
    
    pred = torch.randn(batch_size, 1, height, width)  # Logits
    target = (torch.rand(batch_size, 1, height, width) > 0.5).float()
    
    # Test all losses
    losses = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Focal': FocalLoss(),
        'Weighted BCE': WeightedBCELoss(),
        'Dice': DiceLoss(),
        'Combined': CombinedLoss()
    }
    
    print("\n=== Loss Values ===")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target)
        print(f"{name}: {loss_value.item():.4f}")
    
    # Test with perfect prediction
    pred_perfect = target * 10  # High confidence correct predictions
    
    print("\n=== Perfect Prediction ===")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred_perfect, target)
        print(f"{name}: {loss_value.item():.4f}")
    
    # Test get_loss_function
    print("\n=== Testing get_loss_function ===")
    for loss_name in ['bce', 'focal', 'weighted_bce', 'dice', 'combined']:
        loss_fn = get_loss_function(loss_name)
        print(f"{loss_name}: {type(loss_fn).__name__}")
    
    print("\n✓ Loss functions test completed!")
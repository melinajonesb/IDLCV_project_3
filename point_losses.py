import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy computed only at clicked pixel locations.

    Loss is computed as:
        L = - (1/N) * Σ [y * log(p) + (1-y) * log(1-p)]
    where the sum is only over clicked pixels (N = number of clicks).
    
    Args:
        reduction: 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_logits, click_mask, click_labels):
        """
        Args:
            pred_logits: (B, 1, H, W) - raw network output (before sigmoid)
            click_mask: (B, 1, H, W) - binary mask (1 where clicks are, 0 elsewhere)
            click_labels: (B, 1, H, W) - ground truth at clicks (0 or 1)
        
        Returns:
            Loss value (scalar)
        """
        # Apply sigmoid to logits to get probabilities
        pred_prob = torch.sigmoid(pred_logits)

        # Binary Cross Entropy at each pixel
        bce = -(
            click_labels * torch.log(pred_prob + 1e-8) +
                (1 - click_labels) * torch.log(1 - pred_prob + 1e-8)
        )

        # Apply click mask (only compute loss at clicked locations)
        bce_masked = bce * click_mask

        # Count number of clicks for normalization
        num_clicks = click_mask.sum() + 1e-8  # avoid division by zero

        if self.reduction == 'mean':
            loss = bce_masked.sum() / num_clicks
        elif self.reduction == 'sum':
            loss = bce_masked.sum()
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'")
        
        return loss
    
# Test loss functions
if __name__ == "__main__":
    """Test PartialCrossEntropyLoss"""
    print("\n=== Testing PartialCrossEntropyLoss ===")
    
    loss_fn = PartialCrossEntropyLoss()
    
    # Create dummy data
    B, H, W = 2, 16, 16
    pred = torch.randn(B, 1, H, W)
    
    # Create click mask (4 clicks)
    click_mask = torch.zeros(B, 1, H, W)
    click_mask[0, 0, 5, 10] = 1
    click_mask[0, 0, 8, 15] = 1
    click_mask[0, 0, 4, 12] = 1
    click_mask[1, 0, 7, 7] = 1
    
    # Create labels
    click_labels = torch.zeros_like(click_mask)
    click_labels[0, 0, 5, 10] = 1  # Positive click
    click_labels[0, 0, 8, 15] = 1
    # Negative click are already 0 , so no need to set explicitly    
    # Compute loss
    loss = loss_fn(pred, click_mask, click_labels)
    
    #print(click_mask)
    print(f"Loss: {loss.item():.4f}")
    print(f"Clicks: {click_mask.sum().item()}")
    assert loss.item() > 0
    print("✓ PartialCrossEntropyLoss works!")
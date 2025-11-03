import torch


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice Coefficient (F1 Score for segmentation)
    
    Dice = 2*|X∩Y| / (|X|+|Y|)
    
    Args:
        pred: Predicted segmentation (B, 1, H, W) or (B, H, W)
        target: Ground truth segmentation (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient (scalar)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou(pred, target, smooth=1e-6):
    """
    Intersection over Union (Jaccard Index)
    
    IoU = |X∩Y| / |X∪Y|
    
    Args:
        pred: Predicted segmentation (B, 1, H, W) or (B, H, W)
        target: Ground truth segmentation (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score (scalar)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou_score = (intersection + smooth) / (union + smooth)
    
    return iou_score.item()


def accuracy(pred, target):
    """
    Pixel-wise accuracy
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        pred: Predicted segmentation (B, 1, H, W) or (B, H, W)
        target: Ground truth segmentation (B, 1, H, W) or (B, H, W)
    
    Returns:
        Accuracy (scalar)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    correct = (pred == target).sum()
    total = target.numel()
    
    acc = correct.float() / total
    
    return acc.item()


def sensitivity(pred, target, smooth=1e-6):
    """
    Sensitivity (Recall, True Positive Rate)
    
    Sensitivity = TP / (TP + FN)
    
    Measures how well the model detects positive class (vessels/lesions)
    
    Args:
        pred: Predicted segmentation (B, 1, H, W) or (B, H, W)
        target: Ground truth segmentation (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Sensitivity (scalar)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # True Positives
    tp = (pred * target).sum()
    
    # False Negatives (missed positives)
    fn = ((1 - pred) * target).sum()
    
    sens = (tp + smooth) / (tp + fn + smooth)
    
    return sens.item()


def specificity(pred, target, smooth=1e-6):
    """
    Specificity (True Negative Rate)
    
    Specificity = TN / (TN + FP)
    
    Measures how well the model identifies negative class (background)
    
    Args:
        pred: Predicted segmentation (B, 1, H, W) or (B, H, W)
        target: Ground truth segmentation (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Specificity (scalar)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # True Negatives
    tn = ((1 - pred) * (1 - target)).sum()
    
    # False Positives (false alarms)
    fp = (pred * (1 - target)).sum()
    
    spec = (tn + smooth) / (tn + fp + smooth)
    
    return spec.item()


def compute_all_metrics(pred, target, threshold=0.5):
    """
    Compute all segmentation metrics at once
    
    Args:
        pred: Predicted segmentation logits or probabilities (B, 1, H, W) or (B, H, W)
        target: Ground truth segmentation (B, 1, H, W) or (B, H, W)
        threshold: Threshold for converting predictions to binary (default: 0.5)
    
    Returns:
        Dictionary of metrics
    """
    # Convert predictions to binary
    pred_binary = (pred > threshold).float()
    
    # Ensure target is binary
    target_binary = (target > 0.5).float()
    
    metrics = {
        'dice': dice_coefficient(pred_binary, target_binary),
        'iou': iou(pred_binary, target_binary),
        'accuracy': accuracy(pred_binary, target_binary),
        'sensitivity': sensitivity(pred_binary, target_binary),
        'specificity': specificity(pred_binary, target_binary)
    }
    
    return metrics


# Test the metrics
if __name__ == "__main__":
    print("Testing metrics...")
    
    # Create dummy data
    batch_size = 4
    height, width = 256, 256
    
    # Perfect prediction
    pred = torch.ones(batch_size, 1, height, width)
    target = torch.ones(batch_size, 1, height, width)
    
    print("\n=== Perfect Prediction ===")
    metrics = compute_all_metrics(pred, target)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Random prediction
    pred = torch.rand(batch_size, 1, height, width)
    target = (torch.rand(batch_size, 1, height, width) > 0.5).float()
    
    print("\n=== Random Prediction ===")
    metrics = compute_all_metrics(pred, target)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Mostly correct prediction
    pred = target.clone()
    # Add some noise
    noise_mask = torch.rand_like(pred) < 0.1  # 10% noise
    pred[noise_mask] = 1 - pred[noise_mask]
    
    print("\n=== 90% Correct Prediction ===")
    metrics = compute_all_metrics(pred, target)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print("\n✓ Metrics test completed!")
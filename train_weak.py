"""
Training script for weakly supervised segmentation using point clicks.

Key differences from train.py:
- Uses WeakPH2Dataset instead of PH2Dataset
- Uses PartialCrossEntropyLoss instead of full mask losses
- Evaluates on full masks but trains on clicks only
"""
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

from dataset.weak_dataloader import create_weak_dataloaders
from models.U_net import UNet
from models.encoder_decoder import EncoderDecoder
from point_losses import PartialCrossEntropyLoss
from metrics import compute_all_metrics


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch using ONLY point clicks.
    The full mask is never used during training!
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        click_mask = batch['click_mask'].to(device)
        click_labels = batch['click_labels'].to(device)
        # full_mask is NOT used in training!
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # (B, 1, H, W) logits
        
        # Calculate loss ONLY at click locations
        loss = criterion(outputs, click_mask, click_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return {'loss': avg_loss}


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    - Loss computed on clicks (weak supervision)
    - Metrics computed on FULL masks (to measure actual performance)
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': []
    }
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            click_mask = batch['click_mask'].to(device)
            click_labels = batch['click_labels'].to(device)
            full_mask = batch['full_mask'].to(device)
            
            # Forward pass
            outputs = model(images)  # (B, 1, H, W) logits
            
            # Loss on clicks
            loss = criterion(outputs, click_mask, click_labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Metrics on FULL mask
            outputs_prob = torch.sigmoid(outputs)
            batch_metrics = compute_all_metrics(outputs_prob, full_mask)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Average
    avg_loss = total_loss / num_batches
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = avg_loss
    
    return avg_metrics


def save_checkpoint(model, optimizer, epoch, metrics, args, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args)
    }
    torch.save(checkpoint, filename)


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create WEAK dataloaders
    print(f"\nCreating weak supervision dataloaders...")
    train_loader, val_loader, test_loader = create_weak_dataloaders(
        dataset_name=args.dataset,
        n_pos_clicks=args.n_pos_clicks,
        n_neg_clicks=args.n_neg_clicks,
        click_strategy=args.click_strategy,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model.lower() == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model.lower() == 'encoder_decoder':
        model = EncoderDecoder(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create point-supervised loss
    print(f"\nUsing PartialCrossEntropyLoss (point supervision)...")
    criterion = PartialCrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        'results',
        f"{args.dataset}_{args.model}_pos{args.n_pos_clicks}_neg{args.n_neg_clicks}_{args.click_strategy}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    best_dice = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_accuracy': [],
        'val_sensitivity': [],
        'val_specificity': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['dice'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Metrics on Full Masks:")
        print(f"  Dice: {val_metrics['dice']:.4f}")
        print(f"  IoU: {val_metrics['iou']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {val_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {val_metrics['specificity']:.4f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        for key in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
            history[f'val_{key}'].append(val_metrics[key])
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_checkpoint = os.path.join(results_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, args, best_checkpoint)
            print(f"✓ New best model! Dice: {best_dice:.4f}")
    
    # Save history
    with open(os.path.join(results_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    
    # Test on test set
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results (on Full Masks):")
    print(f"  Dice: {test_metrics['dice']:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    
    # Save final results
    results = {
        'args': vars(args),
        'best_val_dice': best_dice,
        'test_metrics': test_metrics,
        'history': history
    }
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    return results_dir, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train segmentation model with weak supervision (point clicks)'
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='PH2',
                       help='Dataset to use (default: PH2)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers (default: 4)')
    
    # Weak supervision arguments
    parser.add_argument('--n_pos_clicks', type=int, default=5,
                       help='Number of positive clicks (default: 5)')
    parser.add_argument('--n_neg_clicks', type=int, default=5,
                       help='Number of negative clicks (default: 5)')
    parser.add_argument('--click_strategy', type=str, default='random',
                       choices=['random', 'centroid', 'boundary'],
                       help='Click sampling strategy (default: random)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'encoder_decoder'],
                       help='Model architecture (default: unet)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("Weak Supervision Training Configuration:")
    print("-" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("=" * 80)
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()
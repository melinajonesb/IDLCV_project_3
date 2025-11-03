import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime

from dataset.dataloader import create_data_loaders
from models.encoder_decoder import EncoderDecoder
from models.U_net import UNet
from losses import get_loss_function
from metrics import compute_all_metrics


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': []
    }
    
    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            outputs_prob = torch.sigmoid(outputs)
            batch_metrics = compute_all_metrics(outputs_prob, masks)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Average metrics
    avg_loss = total_loss / len(train_loader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = avg_loss
    
    return avg_metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': []
    }
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate metrics
            outputs_prob = torch.sigmoid(outputs)
            batch_metrics = compute_all_metrics(outputs_prob, masks)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Average metrics
    avg_loss = total_loss / len(val_loader)
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
    print(f"Checkpoint saved: {filename}")


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # Create data loaders
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=args.dataset,
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
    
    # Create loss function
    print(f"\nUsing {args.loss} loss...")
    criterion = get_loss_function(args.loss)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        'results', 
        f"{args.dataset}_{args.model}_{args.loss}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    best_dice = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': []
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
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_checkpoint = os.path.join(results_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, args, best_checkpoint)
            print(f"✓ New best model! Dice: {best_dice:.4f}")
        
        # Save latest checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(results_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, args, checkpoint_path)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    
    # Test on test set
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Dice: {test_metrics['dice']:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    
    # Save final results
    results_file = os.path.join(results_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Loss: {args.loss}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"\nBest validation Dice: {best_dice:.4f}\n")
        f.write(f"\nTest Results:\n")
        for key, value in test_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\nResults saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='PH2', 
                        choices=['PH2', 'DRIVE'],
                        help='Dataset to use (default: PH2)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'encoder_decoder'],
                        help='Model architecture (default: unet)')
    
    # Training arguments
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'focal', 'weighted_bce', 'dice', 'combined'],
                        help='Loss function (default: bce)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration:")
    print("-" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("=" * 80)
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()
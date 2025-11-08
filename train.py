import os, sys
USE_TQDM = sys.stdout.isatty()  # False på LSF
os.environ["PYTHONUNBUFFERED"] = "1"  

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

from dataset.dataloader import create_data_loaders
from models.encoder_decoder import EncoderDecoder
from models.U_net import UNet
from losses import get_loss_function
from metrics import compute_all_metrics

# --- Early stopping ---
class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4, mode='max'):
        assert mode in ['min', 'max']
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
    
    def improved(self, current):
        if self.mode == "max":
            return current > self.best + self.min_delta
        else:
            return current < self.best - self.min_delta
    
    def step(self, current):
        if self.improved(current):
            self.best = current
            self.counter = 0
            return False  # Not early stopping
        else:
            self.counter += 1
            return self.counter >= self.patience

# test
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_metrics = {
        'dice': [], 'iou': [], 'accuracy': [], 'sensitivity': [], 'specificity': []
    }

    pbar = tqdm(train_loader, desc='Training', disable=not USE_TQDM)
    printed = False
    for batch in pbar:
        # To support both (images, masks) and (images, masks, fov)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            images, masks, fov = batch
        else:
            images, masks = batch
            fov = None

        images = images.to(device)
        masks = masks.to(device)
        if fov is not None:
            fov = fov.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)

        # Mask with FOV if available (for DRIVE dataset)
        if fov is not None:
            outputs = outputs * fov
            masks   = masks * fov

        # Loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Metrics
        with torch.no_grad():
            outputs_prob = torch.sigmoid(outputs)
            batch_metrics = compute_all_metrics(outputs_prob, masks)
            for k in all_metrics:
                all_metrics[k].append(batch_metrics[k])

        if not printed:
            print("shapes -> images:", tuple(images.shape),
                  "masks:", tuple(masks.shape),
                  "fov:", None if fov is None else tuple(fov.shape))
            printed = True

        if USE_TQDM:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    avg_metrics['loss'] = avg_loss
    return avg_metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_metrics = {
        'dice': [], 'iou': [], 'accuracy': [], 'sensitivity': [], 'specificity': []
    }

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', disable=not USE_TQDM)
        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, masks, fov = batch
            else:
                images, masks = batch
                fov = None

            images = images.to(device)
            masks = masks.to(device)
            if fov is not None:
                fov = fov.to(device)

            outputs = model(images)
            if fov is not None:
                outputs = outputs * fov
                masks   = masks * fov

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            outputs_prob = torch.sigmoid(outputs)
            batch_metrics = compute_all_metrics(outputs_prob, masks)
            for k in all_metrics:
                all_metrics[k].append(batch_metrics[k])

            if USE_TQDM:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(val_loader)
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    history_path = os.path.join(results_dir, 'history.json')
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    best_dice = 0.0
    history = {
        'dataset': args.dataset,
        'model': args.model,
        'loss_name': args.loss,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': [],
        'best_val_dice': None,
        'best_epoch': None,
        'test_metrics': None
    }

    # tracking best training dice
    best_train_dice = 0.0

    # Early stopper
    early_stopper = EarlyStopper(
        patience=args.early_stop_patience, 
        min_delta = args.early_stop_min_delta,
        mode='max')
    
    PLOT_FREQ = args.plot_freq
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        best_train_dice = max(best_train_dice, float(train_metrics['dice']))
        
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
            history['best_val_dice'] = float(best_dice)
            history['best_epoch'] = epoch + 1
            best_checkpoint = os.path.join(results_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, args, best_checkpoint)
            print(f"✓ New best model! Dice: {best_dice:.4f}")

        if (epoch + 1) % PLOT_FREQ == 0 or (epoch + 1) == args.epochs:
            fig, axs = plt.subplots(1, 3, figsize=(12, 3))
            axs[0].plot(history['train_loss'], label='train_loss')
            axs[0].plot(history['val_loss'], label='val_loss')
            axs[0].set_title('Loss')
            axs[0].legend()

            axs[1].plot(history['train_dice'], label='train_dice')
            axs[1].plot(history['val_dice'], label='val_dice')
            axs[1].set_title('Dice')
            axs[1].legend()

            axs[2].plot(history['train_iou'], label='train_iou')
            axs[2].plot(history['val_iou'], label='val_iou')
            axs[2].set_title('IoU')
            axs[2].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'progress_epoch_{epoch+1}.png'))
            plt.close()
            print(f"Saved progress plot for epoch {epoch+1}")

        # Early stopping check
        if early_stopper.step(val_metrics['dice']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # Save latest checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(results_dir, f'checkpoint_epoch_{epoch+1}.pth')
            #save_checkpoint(model, optimizer, epoch, val_metrics, args, checkpoint_path)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")

    # --- Load best model for testing ---
    best_ckpt_path = os.path.join(results_dir, 'best_model.pth')
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best model from epoch {ckpt['epoch']+1.}")
    
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
   
    # save test i history and dump to JSON
    history["test_metrics"] = {k: float(v) for k, v in test_metrics.items()}
    with open(history_path, "w") as f:
        json.dump(history, f, indent =2)
    print(f"Saved per-epoch history to {history_path}")

    csv_path = os.path.join(results_dir, 'history.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,train_dice,val_dice,train_iou,val_iou\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['val_loss'][i]:.6f},"
                    f"{history['train_dice'][i]:.6f},{history['val_dice'][i]:.6f},"
                    f"{history['train_iou'][i]:.6f},{history['val_iou'][i]:.6f}\n")
    print(f"Saved CSV to: {csv_path}")

    #plots for this run
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss curves')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curves.png')); plt.close()

    plt.figure()
    plt.plot(history['train_dice'], label='train_dice')
    plt.plot(history['val_dice'], label='val_dice')
    plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.title('Dice curves')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dice_curves.png')); plt.close()

    plt.figure()
    plt.plot(history['train_iou'], label='train_iou')
    plt.plot(history['val_iou'], label='val_iou')
    plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.title('IoU curves')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'iou_curves.png')); plt.close()
    print(f"Saved plots to: {results_dir}")

    summary_path = "results/ALL_EXPERIMENTS_SUMMARY.csv"
    header = ("dataset,model,loss,"
              "best_train_dice,best_val_dice,"
              "test_dice,test_iou,test_accuracy,test_sensitivity,test_specificity\n")

    summary_row = (
        f"{args.dataset},{args.model},{args.loss},"
        f"{best_train_dice:.4f},{history['best_val_dice']:.4f},"
        f"{test_metrics['dice']:.4f},{test_metrics['iou']:.4f},"
        f"{test_metrics['accuracy']:.4f},"
        f"{test_metrics['sensitivity']:.4f},"
        f"{test_metrics['specificity']:.4f}\n"
    )
    os.makedirs("results", exist_ok=True)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a") as f:
        if write_header:
            f.write(header)
        f.write(summary_row)
    print(f"[SUMMARY] {summary_row.strip()}")



def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')

    parser.add_argument('--early_stop_patience', type=int, default=7,
                    help='Early stopping patience on val metric (default: 7)')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='Minimum improvement to reset patience (default: 1e-4)')
    parser.add_argument('--plot_freq', type=int, default=5,
                        help='Save progress plots every N epochs (default: 5)')
    
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
"""
Weak Supervision DataLoader
Wraps existing PH2Dataset and adds click generation on-the-fly
"""
import torch
from torch.utils.data import Dataset
from dataset.make_clicks import ClickSimulatorPH2


class WeakPH2Dataset(Dataset):
    """
    Wrapper around PH2Dataset that generates clicks on-the-fly.
    
    During training, the model only sees click locations + labels.
    The full mask is kept for evaluation only.
    
    Args:
        base_dataset: The base PH2Dataset (from dataloader.py)
        n_pos_clicks: Number of positive clicks per image
        n_neg_clicks: Number of negative clicks per image
        click_strategy: 'random', 'centroid', or 'boundary'
        min_dist: Minimum distance between clicks (pixels)
    """
    def __init__(self, base_dataset, n_pos_clicks=5, n_neg_clicks=5, 
                 click_strategy='random', min_dist=10, seed=42):
        self.base_dataset = base_dataset
        self.n_pos_clicks = n_pos_clicks
        self.n_neg_clicks = n_neg_clicks
        self.click_simulator = ClickSimulatorPH2(
            strategy=click_strategy,
            min_dist=min_dist,
            seed=seed
        )
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get image and full mask from base dataset
        image, full_mask = self.base_dataset[idx]
        
        # Generate clicks from full mask
        pos_points, neg_points = self.click_simulator.sample_clicks(
            full_mask, 
            num_pos=self.n_pos_clicks,
            num_neg=self.n_neg_clicks
        )
        
        # Create click mask and labels for loss computation
        click_mask, click_labels = self.click_simulator.to_loss_tensors(
            full_mask, pos_points, neg_points
        )
        
        return {
            'image': image,              # (3, H, W)
            'click_mask': click_mask,    # (1, 1, H, W) - where clicks are
            'click_labels': click_labels,# (1, 1, H, W) - labels at clicks  
            'full_mask': full_mask       # (1, H, W) - for evaluation ONLY!
        }


def create_weak_dataloaders(dataset_name='PH2', n_pos_clicks=5, n_neg_clicks=5,
                           click_strategy='random', batch_size=8, img_size=256,
                           num_workers=4, train_split=0.7, val_split=0.15):
    """
    Create weak supervision dataloaders.
    
    Returns train, val, test loaders where each batch returns:
    - image: (B, 3, H, W)
    - click_mask: (B, 1, H, W) - binary mask of click locations
    - click_labels: (B, 1, H, W) - labels at click locations
    - full_mask: (B, 1, H, W) - ground truth for evaluation
    """
    from dataset.dataloader import create_data_loaders
    
    # Get base dataloaders
    train_base, val_base, test_base = create_data_loaders(
        dataset_name=dataset_name,
        batch_size=1,  # We'll handle batching after wrapping
        img_size=img_size,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    # Wrap with weak supervision
    train_weak = WeakPH2Dataset(
        train_base.dataset,
        n_pos_clicks=n_pos_clicks,
        n_neg_clicks=n_neg_clicks,
        click_strategy=click_strategy
    )
    
    val_weak = WeakPH2Dataset(
        val_base.dataset,
        n_pos_clicks=n_pos_clicks,
        n_neg_clicks=n_neg_clicks,
        click_strategy=click_strategy
    )
    
    test_weak = WeakPH2Dataset(
        test_base.dataset,
        n_pos_clicks=n_pos_clicks,
        n_neg_clicks=n_neg_clicks,
        click_strategy=click_strategy
    )
    
    # Custom collate function for dictionary batches
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        click_masks = torch.cat([item['click_mask'] for item in batch])
        click_labels = torch.cat([item['click_labels'] for item in batch])
        full_masks = torch.stack([item['full_mask'] for item in batch])
        
        return {
            'image': images,
            'click_mask': click_masks,
            'click_labels': click_labels,
            'full_mask': full_masks
        }
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_weak, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_weak, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_weak, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"\n✓ Weak supervision dataloaders created:")
    print(f"  Positive clicks: {n_pos_clicks}")
    print(f"  Negative clicks: {n_neg_clicks}")
    print(f"  Strategy: {click_strategy}")
    
    return train_loader, val_loader, test_loader


# Test the weak dataloader
if __name__ == "__main__":
    print("Testing weak supervision dataloader...")
    
    train_loader, val_loader, test_loader = create_weak_dataloaders(
        dataset_name='PH2',
        n_pos_clicks=3,
        n_neg_clicks=3,
        click_strategy='random',
        batch_size=2,
        num_workers=0
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Click masks: {batch['click_mask'].shape}")
    print(f"  Click labels: {batch['click_labels'].shape}")
    print(f"  Full masks: {batch['full_mask'].shape}")
    
    print(f"\nClick statistics:")
    print(f"  Total clicks: {batch['click_mask'].sum().item()}")
    print(f"  Positive clicks: {batch['click_labels'].sum().item()}")
    print(f"  Negative clicks: {batch['click_mask'].sum().item() - batch['click_labels'].sum().item()}")
    
    print("\n✓ Weak dataloader test passed!")
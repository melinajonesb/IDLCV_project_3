import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class PH2Dataset(Dataset):
    """
    Dataset class for PH2 skin lesion segmentaion
    Structure:
    PH2_Dataset_images/
        ├── IMD002/
        │   ├── IMD002_Dermoscopic_Image/IMD435.bmp
        │   └── IMD002_lesion/IMD435_lesion.bmp
        ├── IMD003/
        └── ...
    """
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get all folders (each contains one image)
        self.folders = sorted([d for d in glob.glob(os.path.join(root_dir, 'IMD*')) 
                              if os.path.isdir(d)])
        
        self.image_paths = []
        self.mask_paths = []
        
        for folder in self.folders:
            folder_name = os.path.basename(folder)
            
            # Image path
            img_path = os.path.join(folder, f'{folder_name}_Dermoscopic_Image', f'{folder_name}.bmp')
            
            # Mask path (lesion mask)
            mask_path = os.path.join(folder, f'{folder_name}_lesion', f'{folder_name}_lesion.bmp')
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
        
        print(f"Found {len(self.image_paths)} images in PH2 dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask


class DRIVEDataset(Dataset):
    """
    Dataset class for DRIVE retinal vessel segmentation
    Structure:
    DRIVE/
        └── training/
            ├── images/XX_training.tif
            └── 1st_manual/XX_manual1.gif (vessel masks)
    
    Note: We only use the training set (20 images) since test set doesn't have vessel masks.
    The training set will be split into train/val/test in create_data_loaders().
    """
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Only use training set (has vessel masks)
        self.image_dir = os.path.join(root_dir, 'training', 'images')
        self.mask_dir = os.path.join(root_dir, 'training', '1st_manual')
        
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*_training.tif')))
        self.mask_paths = []
        
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            # Extract number: 21_training.tif -> 21
            img_num = img_name.split('_')[0]
            mask_name = f'{img_num}_manual1.gif'
            mask_path = os.path.join(self.mask_dir, mask_name)
            self.mask_paths.append(mask_path)
        
        print(f"Found {len(self.image_paths)} images in DRIVE dataset (training set only)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask


def get_transforms(img_size=256, augment=False):
    """
    Get image and mask transforms
    
    Args:
        img_size: Target image size (will resize to img_size x img_size)
        augment: Whether to apply data augmentation
    
    Returns:
        image_transform, mask_transform
    """
    if augment:
        # Training transforms with augmentation
        image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ToTensor()
        ])
    else:
        # Validation/test transforms (no augmentation)
        image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    return image_transform, mask_transform


def create_data_loaders(dataset_name, batch_size=8, img_size=256, num_workers=4, 
                        train_split=0.7, val_split=0.15):
    """
    Create train, validation, and test data loaders
    
    Args:
        dataset_name: 'PH2' or 'DRIVE'
        batch_size: Batch size for data loaders
        img_size: Image size for resizing
        num_workers: Number of workers for data loading
        train_split: Proportion of data for training (default: 0.7)
        val_split: Proportion of data for validation (default: 0.15)
        Note: test_split = 1 - train_split - val_split (default: 0.15)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Base path for datasets
    base_path = '/dtu/datasets1/02516'
    
    if dataset_name.upper() == 'PH2':
        # Get transforms
        train_img_tf, train_mask_tf = get_transforms(img_size, augment=True)
        val_img_tf, val_mask_tf = get_transforms(img_size, augment=False)
        
        # Load full dataset
        full_dataset = PH2Dataset(
            root_dir=os.path.join(base_path, 'PH2_Dataset_images'),
            transform=train_img_tf,
            mask_transform=train_mask_tf
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Update transforms for val and test
        val_dataset.dataset.transform = val_img_tf
        val_dataset.dataset.mask_transform = val_mask_tf
        test_dataset.dataset.transform = val_img_tf
        test_dataset.dataset.mask_transform = val_mask_tf
        
    elif dataset_name.upper() == 'DRIVE':
        # Get transforms
        train_img_tf, train_mask_tf = get_transforms(img_size, augment=True)
        val_img_tf, val_mask_tf = get_transforms(img_size, augment=False)
        
        # Load full dataset (only training set has vessel masks)
        full_dataset = DRIVEDataset(
            root_dir=os.path.join(base_path, 'DRIVE'),
            transform=train_img_tf,
            mask_transform=train_mask_tf
        )
        
        # Split into train/val/test
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update transforms for val and test
        val_dataset.dataset.transform = val_img_tf
        val_dataset.dataset.mask_transform = val_mask_tf
        test_dataset.dataset.transform = val_img_tf
        test_dataset.dataset.mask_transform = val_mask_tf
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'PH2' or 'DRIVE'")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset: {dataset_name}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


# Test the data loaders
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PH2 dataset...")
    print("=" * 60)
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            'PH2', batch_size=4, img_size=256, num_workers=0
        )
        
        # Get a batch
        images, masks = next(iter(train_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask values: {masks.unique()}")
        print("✓ PH2 dataset loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading PH2: {e}\n")
    
    print("=" * 60)
    print("Testing DRIVE dataset...")
    print("=" * 60)
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            'DRIVE', batch_size=4, img_size=256, num_workers=0
        )
        
        # Get a batch
        images, masks = next(iter(train_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask values: {masks.unique()}")
        print("✓ DRIVE dataset loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading DRIVE: {e}")
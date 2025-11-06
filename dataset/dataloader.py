import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import random

# Joint augmentation for image and vessel_mask (ensure they are transformed the same way)
class JointAugment:
    def __init__(self, p_flip=0.5, degrees=20):
        self.p_flip = p_flip
        self.degrees = degrees
    
    def __call__(self, img, vessel_mask, fov_mask=None):
        # Random horizontal flip
        if random.random() < self.p_flip:
            img = F.hflip(img)
            vessel_mask = F.hflip(vessel_mask)
            if fov_mask is not None:
                fov_mask = F.hflip(fov_mask)
        
        # Random vertical flip
        if random.random() < self.p_flip:
            img = F.vflip(img)
            vessel_mask = F.vflip(vessel_mask)
            if fov_mask is not None:
                fov_mask = F.vflip(fov_mask)
        
        # Random rotation
        angle = random.uniform(-self.degrees, self.degrees)
        img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        vessel_mask = F.rotate(vessel_mask, angle, interpolation=InterpolationMode.NEAREST)
        if fov_mask is not None:
            fov_mask = F.rotate(fov_mask, angle, interpolation=InterpolationMode.NEAREST)
          

        return (img, vessel_mask, fov_mask) if fov_mask is not None else (img, vessel_mask)
    

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
    def __init__(self, root_dir, transform=None, mask_transform=None, joint_augment=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.joint_augment = joint_augment
        
        # Get all folders (each contains one image)
        self.folders = sorted([d for d in glob.glob(os.path.join(root_dir, 'IMD*')) 
                              if os.path.isdir(d)])
        
        self.image_paths = []
        self.mask_paths = []
        
        for folder in self.folders:
            folder_name = os.path.basename(folder)
            
            # Image path
            img_path = os.path.join(folder, f'{folder_name}_Dermoscopic_Image', f'{folder_name}.bmp')
            
            # Mask path (lesion vessel_mask)
            vessel_mask_path = os.path.join(folder, f'{folder_name}_lesion', f'{folder_name}_lesion.bmp')
            
            if os.path.exists(img_path) and os.path.exists(vessel_mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(vessel_mask_path)
        
        print(f"Found {len(self.image_paths)} images in PH2 dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Load vessel_mask
        vessel_mask = Image.open(self.mask_paths[idx]).convert('L')  # Grayscale

        # Synchronized augmentation (before ToTensor)
        if self.joint_augment is not None:
            image, vessel_mask = self.joint_augment(image, vessel_mask)
        
        # Apply transforms (individually)
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            vessel_mask = self.mask_transform(vessel_mask)
        
        # Convert vessel_mask to binary (0 or 1)
        vessel_mask = (vessel_mask > 0.5).float()
        
        return image, vessel_mask


class DRIVEDataset(Dataset):
    """
    Dataset class for DRIVE retinal vessel segmentation
    Structure:
    DRIVE/
        └── training/
            ├── images/XX_training.tif
            └── 1st_manual/XX_manual1.gif (vessel masks)
            └── vessel_mask/XX_training_mask.gif (FOV masks)
    
    Note: We only use the training set (20 images) since test set doesn't have vessel masks.
    The training set will be split into train/val/test in create_data_loaders().
    The FOV masks are applied during evaluation to ignore background.
    """
    def __init__(self, root_dir, transform=None, mask_transform=None, joint_augment=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.joint_augment = joint_augment
        
        # Only use training set (has vessel masks)
        self.image_dir = os.path.join(root_dir, 'training', 'images')
        self.vessel_mask_dir = os.path.join(root_dir, 'training', '1st_manual') 
        self.fov_mask_dir = os.path.join(root_dir, 'training', 'mask')
        
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*_training.tif')))
        self.vessel_mask_paths = []
        self.fov_mask_paths = []
        
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            # Extract number: 21_training.tif -> 21
            img_num = img_name.split('_')[0]
            vessel_mask_name = f'{img_num}_manual1.gif'
            fov_mask_name = f'{img_num}_training_mask.gif'
            self.vessel_mask_paths.append(os.path.join(self.vessel_mask_dir, vessel_mask_name))
            self.fov_mask_paths.append(os.path.join(self.fov_mask_dir, fov_mask_name))
        
        print(f"Found {len(self.image_paths)} images in DRIVE dataset (training set only)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Load vessel mask
        vessel_mask = Image.open(self.vessel_mask_paths[idx]).convert('L')

        # Load FOV mask
        fov_mask = Image.open(self.fov_mask_paths[idx]).convert('L')

        # Synchronized augmentation (before ToTensor)
        if self.joint_augment is not None:
            image, vessel_mask = self.joint_augment(image, vessel_mask)

        # Apply transforms (individually)
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            vessel_mask = self.mask_transform(vessel_mask)
            fov_mask = self.mask_transform(fov_mask)
        
        # Convert vessel_mask to binary (0 or 1)
        vessel_mask = (vessel_mask > 0.5).float()
        fov_mask = (fov_mask > 0.5).float()
        
        return image, vessel_mask, fov_mask


def get_transforms(img_size=256, augment=False):
    """
    Get image and vessel_mask transforms
    
    Args:
        img_size: Target image size (will resize to img_size x img_size)
        augment: Whether to apply data augmentation (uses joint augmentations if true)
        masks are always resized with nearest
    
    Returns:
        image_transform, mask_transform
    """
    if augment:
        # Training transforms with augmentation
        joint_augment = JointAugment(p_flip=0.5, degrees=20)

        image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    else:
        joint_augment = None

        # Validation/test transforms (no augmentation)
        image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    return image_transform, mask_transform, joint_augment


# split helper

def _split_indices(n, train_split, val_split, seed=42):
    """Returns train, val, test indices for a dataset of size n with given seed"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    n_train = int(train_split * n)
    n_val = int(val_split * n)
    n_test = n - n_train - n_val
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

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
        train_img_tf, train_mask_tf, train_joint_tf = get_transforms(img_size, augment=True)
        val_img_tf, val_mask_tf, _ = get_transforms(img_size, augment=False)

        # Compute indices
        base = PH2Dataset(root_dir=os.path.join(base_path, 'PH2_Dataset_images'))
        train_idx, val_idx, test_idx = _split_indices(len(base), train_split, val_split)


        # Separate datasets
        ds_train = PH2Dataset(os.path.join(base_path, 'PH2_Dataset_images'),
                              transform=train_img_tf,
                              mask_transform=train_mask_tf,
                              joint_augment=train_joint_tf)
        ds_val = PH2Dataset(os.path.join(base_path, 'PH2_Dataset_images'),
                            transform=val_img_tf,
                            mask_transform=val_mask_tf)
        ds_test = PH2Dataset(os.path.join(base_path, 'PH2_Dataset_images'),
                             transform=val_img_tf,
                             mask_transform=val_mask_tf)

        train_dataset = Subset(ds_train, train_idx)
        val_dataset = Subset(ds_val, val_idx)
        test_dataset = Subset(ds_test, test_idx)

    elif dataset_name.upper() == 'DRIVE':
        # Get transforms
        train_img_tf, train_mask_tf, train_joint_tf = get_transforms(img_size, augment=True)
        val_img_tf, val_mask_tf, _ = get_transforms(img_size, augment=False)
        
        # Compute indices
        base = DRIVEDataset(root_dir=os.path.join(base_path, 'DRIVE'))
        train_idx, val_idx, test_idx = _split_indices(len(base), train_split, val_split)

        # Separate datasets
        ds_train = DRIVEDataset(os.path.join(base_path, 'DRIVE'),
                               transform=train_img_tf,
                               mask_transform=train_mask_tf,
                               joint_augment=train_joint_tf)
        ds_val = DRIVEDataset(os.path.join(base_path, 'DRIVE'),
                             transform=val_img_tf,
                             mask_transform=val_mask_tf)
        ds_test = DRIVEDataset(os.path.join(base_path, 'DRIVE'),
                              transform=val_img_tf,
                              mask_transform=val_mask_tf)
        
        train_dataset = Subset(ds_train, train_idx)
        val_dataset = Subset(ds_val, val_idx)
        test_dataset = Subset(ds_test, test_idx)
    
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
        images, masks, fovs = next(iter(train_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask values: {masks.unique()}")
        print(f"FOV values: {fovs.unique()}")
        print("✓ DRIVE dataset loaded successfully!")

    except Exception as e:
        print(f"✗ Error loading DRIVE: {e}")
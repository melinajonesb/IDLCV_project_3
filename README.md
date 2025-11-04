# Introduction to Deep Learning in Computer Vision - Project 3 - Image Segmentation

Medical image segmentation on PH2 (skin lesions) and DRIVE (retinal vessels) datasets.

## Structure

```
.
├── README.md
├── dataset/
│   ├── __init__.py
│   └── dataloader.py       # Data loaders for PH2 and DRIVE
├── models/
│   ├── __init__.py
│   ├── encoder_decoder.py  # Simple encoder-decoder
│   └── U_net.py            # U-Net implementation
├── losses.py               # CE, Focal, Weighted CE
├── metrics.py              # Dice, IoU, Accuracy, Sensitivity, Specificity
├── train.py                # Training and evaluation script
└── results/                # Save outputs here
```

## Datasets

- **Location**: `/dtu/datasets1/02516`
- **PH2**: Skin lesion segmentation
- **DRIVE**: Retinal vessel segmentation (use vessel masks, NOT field-of-view masks)

## Tasks

### Task 1: Data Loading
- [ ] Complete `dataset/dataloader.py` for both PH2 and DRIVE
- [ ] Create train/val/test splits
- [ ] Add data augmentation (flips, rotations, scaling)
- [ ] Handle different image resolutions

### Task 2: Simple CNN
- [ ] Complete `models/encoder_decoder.py`
- [ ] Implement `metrics.py` (Dice, IoU, Accuracy, Sensitivity, Specificity)
- [ ] Implement `losses.py` (Cross Entropy, Focal Loss, Weighted CE)
- [ ] Implement `train.py` with training loop and evaluation
- [ ] Train and evaluate on both datasets

### Task 3: U-Net
- [ ] Complete `models/U_net.py`
- [ ] Train and evaluate on both datasets
- [ ] Compare with encoder-decoder

### Task 4: Ablation Study
- [ ] Compare different loss functions

## Quick Start

```bash
# Training
python train.py --dataset PH2 --model unet --loss focal
```
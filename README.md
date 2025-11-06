# Introduction to Deep Learning in Computer Vision - Project 3 - Image Segmentation

Medical image segmentation on PH2 (skin lesions) and DRIVE (retinal vessels) datasets.

## Project Structure

```
.
├── README.md
├── dataset/
│   ├── __init__.py
│   └── dataloader.py           # Full supervision data loaders
│   ├── make_clicks.py          # Click sampling strategies
│   └── weak_dataloader.py      # Weak supervision data loader
├── models/
│   ├── __init__.py
│   ├── encoder_decoder.py      # Simple encoder-decoder
│   └── U_net.py                # U-Net with skip connections
├── losses.py                   # Full supervision losses
├── point_losses.py             # Weak supervision losses
├── metrics.py                  # Evaluation metrics
├── train.py                    # Full supervision training
├── train_weak.py               # Weak supervision training
├── weak_ablation_study.py      # Automated ablation experiments
└── results/                    # Save models and results
```

## Datasets

- **Location**: `/dtu/datasets1/02516`
- **PH2**: Skin lesion segmentation
- **DRIVE**: Retinal vessel segmentation

## Part 1: Fully Supervised Segmentation

### Implementation

**Architectures**: U-Net, Encoder-Decoder  
**Loss Functions**: BCE, Focal, Weighted BCE, Dice, Combined  
**Metrics**: Dice, IoU, Accuracy, Sensitivity, Specificity

### Results - PH2 Dataset

| Architecture | Loss Function | Test Dice (%) | Test IoU (%) | Sensitivity (%) | Specificity (%) |
|--------------|---------------|---------------|--------------|-----------------|-----------------|
| U-Net | Weighted BCE | **93.84** | **88.41** | **93.40** | 97.69 |
| U-Net | Dice | 93.65 | 88.08 | 93.12 | 97.59 |
| U-Net | Combined | 93.48 | 87.76 | 92.57 | 97.75 |
| U-Net | BCE | 93.31 | 87.49 | 90.35 | 98.72 |
| Encoder-Decoder | Dice | 92.98 | 86.90 | 92.04 | 97.66 |

**Best configuration**: U-Net + Weighted BCE (93.84% Dice)

### Results - DRIVE Dataset

| Architecture | Loss Function | Test Dice (%) | Test IoU (%) | Sensitivity (%) | Specificity (%) |
|--------------|---------------|---------------|--------------|-----------------|-----------------|
| U-Net | Combined | **60.40** | **43.27** | 60.72 | 96.70 |
| U-Net | BCE | 56.14 | 39.02 | 50.01 | 97.69 |
| U-Net | Dice | 54.52 | 37.47 | 75.96 | 91.59 |
| U-Net | Weighted BCE | 52.64 | 35.72 | **83.15** | 89.13 |

**Best configuration**: U-Net + Combined (60.40% Dice)

---

## Part 2: Weakly Supervised Segmentation

### Method

Training with point clicks instead of full segmentation masks:
- **Click sampling strategies**: Random, centroid-based, boundary-based
- **Loss function**: PartialCrossEntropyLoss (BCE only at clicked pixels)
- **Training**: Loss computed on clicks, metrics evaluated on full masks

### Results - PH2 Dataset

**Fully supervised baseline**: 91.79% Dice

**Weak supervision performance**:

| Clicks | Strategy | Test Dice (%) | % of Baseline |
|--------|----------|---------------|---------------|
| 2 (1+1) | Random | **90.61** | **98.7%** |
| 4 (2+2) | Random | 90.45 | 98.5% |
| 6 (3+3) | Random | 88.65 | 96.6% |
| 8 (4+4) | Random | 90.73 | 98.8% |
| 10 (5+5) | Random | 90.42 | 98.5% |
| 20 (10+10) | Random | 89.51 | 97.5% |
| 40 (20+20) | Random | **91.46** | **99.6%** |

**Strategy comparison (10 clicks)**:

| Strategy | Test Dice (%) |
|----------|---------------|
| Random | 90.42 |
| Centroid | 85.46 |
| Boundary | 65.77 |

**Key findings**:
- 2 clicks achieve 98.7% of fully supervised performance
- 40 clicks achieve 99.6% of fully supervised performance  
- Random sampling outperforms centroid and boundary strategies
- Boundary strategy fails (65.77%) due to ambiguous edge supervision

---

## Usage

### Full Supervision
```bash
python train.py --dataset PH2 --model unet --loss weighted_bce --epochs 50
```

### Weak Supervision
```bash
# Single experiment
python train_weak.py --n_pos_clicks 5 --n_neg_clicks 5 --epochs 50

# Full ablation study
python weak_ablation_study.py full
```

---

## Next Steps

1. **Advanced point losses**: Implement Dice-based and Focal variants
2. **Non-equal click ratios**: Test asymmetric configurations (5+1, 10+1, 1+20)
3. **Bounding box supervision**: Iterative pseudo-label refinement
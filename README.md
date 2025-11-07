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

| Architecture      | Loss Function | Test Dice (%) | Test IoU (%) | Sensitivity (%) | Specificity (%) |
|-------------------|---------------|---------------|--------------|-----------------|-----------------|
| **U-Net**         | **Weighted BCE** | **93.07**  | **87.05**    | **94.88**       | **96.52** |
| U-Net             | Focal           | 88.55.      | 79.56        | 89.53           | 94.68 |
| U-Net             | BCE             | 92.69       | 86.39        | 94.04           | 96.43 |
| Encoder-Decoder   | Weighted BCE    | 92.14       | 85.43        | 92.61           | 96.61 |
| Encoder-Decoder   | Focal           | 89.79       | 81.52        | 85.15           | 98.13 |
| Encoder-Decoder   | BCE             | 90.75       | 83.08        | 89.83.          | 96.27 |
 
**Best configuration (PH2):** U-Net + Weighted BCE (Dice = 93.07 %)


### Results - DRIVE Dataset

| Architecture      | Loss Function   | Test Dice (%) | Test IoU (%) | Sensitivity (%) | Specificity (%) |
|-------------------|-----------------|---------------|--------------|-----------------|-----------------|
| **U-Net**         | **Weighted BCE** | **55.54**    | **38.44**    | **67.40**       | **93.19** |
| U-Net             | Focal            | 6.62         | 3.42         | 5.58            | 94.31 |
| U-Net             | BCE              | 21.55        | 12.07        | 100.00 †        | 34.17 |
| Encoder-Decoder   | Weighted BCE     | 21.55        | 12.07        | 100.00 †        | 34.17 |
| Encoder-Decoder   | Focal            | 0.00         | 0.00         | 0.00            | 100.00 |
| Encoder-Decoder   | BCE              | 21.55        | 12.07        | 100.00 †        | 34.17 |

**Best configuration (DRIVE):** U-Net + Weighted BCE (Dice = 55.54 %)  

† *Extremely high sensitivity and low specificity indicate instability (model predicted mostly foreground).*
Encoder-Decoder + Focal failed to learn — predicted only background (Dice = 0) probably.
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
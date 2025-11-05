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
- [x] Complete `dataset/dataloader.py` for both PH2 and DRIVE
- [x] Create train/val/test splits

### Task 2: Simple Encoder-Decoder
- [x] Complete `models/encoder_decoder.py`
- [x] Implement `metrics.py` (Dice, IoU, Accuracy, Sensitivity, Specificity)
- [x] Implement `losses.py` (Cross Entropy, Focal Loss, Weighted CE)
- [x] Implement `train.py` with training loop and evaluation
- [x] Train and evaluate on both datasets

### Task 3: U-Net
- [x] Complete `models/U_net.py`
- [x] Train and evaluate on both datasets
- [x] Compare with encoder-decoder

### Task 4: Ablation Study
- [x] Compare different loss functions

## Quick Start

```bash
# Training
python train.py --dataset PH2 --model unet --loss focal
```

# Complete Experimental Results

## PH2 Dataset (Skin Lesions - 200 images)

| Architecture | Loss Function | Test Dice (%) | Test IoU (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|--------------|---------------|---------------|--------------|--------------|-----------------|-----------------|
| **U-Net** | **Weighted BCE** | **93.84** | **88.41** | **96.59** | **93.40** | 97.69 |
| **U-Net** | Dice | 93.65 | 88.08 | 96.39 | 93.12 | 97.59 |
| **U-Net** | Combined | 93.48 | 87.76 | 96.39 | 92.57 | 97.75 |
| **U-Net** | BCE | 93.31 | 87.49 | 96.49 | 90.35 | 98.72 |
| Encoder-Decoder | Dice | 92.98 | 86.90 | 96.12 | 92.04 | 97.66 |
| Encoder-Decoder | Combined | 92.85 | 86.65 | 96.15 | 90.77 | 98.10 |
| Encoder-Decoder | BCE | 92.65 | 86.32 | 96.08 | 90.03 | **98.26** |
| Encoder-Decoder | Focal | 92.31 | 85.73 | 95.98 | 87.85 | **99.02** |
| Encoder-Decoder | Weighted BCE | 92.31 | 85.73 | 95.73 | 91.45 | 97.31 |
| **U-Net** | Focal | 90.73 | 83.09 | 95.01 | 85.80 | 98.43 |

**Best Model:** U-Net + Weighted BCE  
**Highest Dice:** 93.84%  
**Highest Sensitivity:** 93.40% (U-Net + Weighted BCE)  
**Highest Specificity:** 99.02% (Encoder-Decoder + Focal)

---

## DRIVE Dataset (Retinal Vessels - 20 images)

| Architecture | Loss Function | Test Dice (%) | Test IoU (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|--------------|---------------|---------------|--------------|--------------|-----------------|-----------------|
| **U-Net** | **Combined** | **60.40** | **43.27** | 93.97 | 60.72 | 96.70 |
| **U-Net** | BCE | 56.14 | 39.02 | **94.08** | 50.01 | 97.69 |
| **U-Net** | Dice | 54.52 | 37.47 | 90.41 | 75.96 | 91.59 |
| **U-Net** | Weighted BCE | 52.64 | 35.72 | 88.67 | **83.15** | 89.13 |
| Encoder-Decoder | Combined | 26.88 | 15.53 | 88.20 | 28.66 | 93.07 |
| Encoder-Decoder | Dice | 26.78 | 15.46 | 80.32 | 47.54 | 83.00 |
| Encoder-Decoder | Weighted BCE | 23.43 | 13.27 | 69.89 | 60.86 | 70.63 |
| **U-Net** | Focal | 5.74 | 2.96 | 92.65 | 2.96 | **99.99** |
| Encoder-Decoder | BCE | 2.04 | 1.03 | 92.43 | 1.04 | 99.91 |
| Encoder-Decoder | Focal | 0.00 | 0.00 | 92.43 | 0.00 | **100.00** |

**Best Model:** U-Net + Combined  
**Highest Dice:** 60.40%  
**Highest Sensitivity:** 83.15% (U-Net + Weighted BCE)  
**Highest Specificity:** 100.00% (Encoder-Decoder + Focal - but useless!)

---

## Key Observations

### Architecture Performance

| Dataset | U-Net Avg Dice | Encoder-Decoder Avg Dice | Gap |
|---------|----------------|--------------------------|-----|
| **PH2** | 92.80% | 92.62% | +0.18% |
| **DRIVE** | 45.89% | 15.83% | **+30.06%** |

### Loss Function Rankings

**PH2 (U-Net):**
1. Weighted BCE: 93.84%
2. Dice: 93.65%
3. Combined: 93.48%
4. BCE: 93.31%
5. Focal: 90.73%

**DRIVE (U-Net):**
1. Combined: 60.40%
2. BCE: 56.14%
3. Dice: 54.52%
4. Weighted BCE: 52.64%
5. Focal: 5.74%

### Critical Findings

- **U-Net essential for small datasets:** +30% improvement on DRIVE
- **Encoder-Decoder fails on DRIVE:** Best result only 26.88% (not clinically viable)
- **Focal Loss fails both datasets:** Too aggressive for mild class imbalance
- **Weighted BCE best for sensitivity:** 93.40% (PH2), 83.15% (DRIVE)
- **Combined Loss best overall for hard tasks:** 60.40% on DRIVE
- **Trade-off exists:** High sensitivity → Lower specificity

### Success Rate (Dice > 50%)

- **U-Net:** 9/10 experiments (90%)
- **Encoder-Decoder:** 5/10 experiments (50%)

---

## Recommendations

**Use U-Net** for medical segmentation (especially with limited data)  
**Use Combined Loss** for balanced performance  
**Use Weighted BCE** when detecting lesions/vessels is critical  
**Avoid Focal Loss** for mild class imbalance  
**Avoid Encoder-Decoder** for challenging small datasets
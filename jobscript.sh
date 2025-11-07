#!/bin/sh
### -------- queue -----------------------------
#BSUB -q c02516

### -------- GPU -------------------------------
#BSUB -gpu "num=1:mode=exclusive_process"

### -------- job name --------------------------
#BSUB -J segm_all_runs

### -------- cores & memory --------------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"

### -------- wall-clock time -------------------
#BSUB -W 12:00

### -------- output/error logs -----------------
#BSUB -o logs/run_%J.log
#BSUB -e logs/run_%J.log

### -------- environment -----------------------
# Aktiver miljø (endre sti om nødvendig)
source ~/venv_1/bin/activate

# sikre mapper, backend og ubufferet output
mkdir -p logs results
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# --------- eksperimentparametere --------------
DATASETS=("PH2" "DRIVE")
MODELS=("unet" "encoder_decoder")
LOSSES=("bce" "weighted_bce" "dice" "combined" "focal")

# --------- hovedløkken ------------------------
echo "=== Starting all experiments ==="
for D in "${DATASETS[@]}"; do
  for M in "${MODELS[@]}"; do
    for L in "${LOSSES[@]}"; do
      echo ">>> Running: dataset=$D  model=$M  loss=$L"
      python -u train.py \
        --dataset "$D" \
        --model "$M" \
        --loss "$L" \
        --epochs 50 \
        --batch_size 8 \
        --num_workers 4 
      echo ">>> Done: $D  $M  $L"
      echo "-----------------------------------------"
    done
  done
done

echo "=== All experiments finished ==="
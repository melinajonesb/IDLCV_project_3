from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Konfigurasjon — endre her om stier endres senere
# ------------------------------------------------------------
SUMMARY_CSV = Path("/zhome/c9/a/223944/project_3/IDLCV_project_3/results/ALL_EXPERIMENTS_SUMMARY.csv")
OUTDIR = Path("/zhome/c9/a/223944/project_3/IDLCV_project_3/results")

# ------------------------------------------------------------
# Funksjoner
# ------------------------------------------------------------
METRICS = ["test_dice", "test_iou", "test_accuracy", "test_sensitivity", "test_specificity"]

def load_summary():
    """Laster inn all_experiments_summary.csv"""
    df = pd.read_csv(SUMMARY_CSV)
    numeric_cols = [
        "best_train_dice", "best_val_dice", "test_dice", "test_iou",
        "test_accuracy", "test_sensitivity", "test_specificity"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def select_best_per_model(df):
    """Velg beste rad per dataset og modell (basert på test_dice)."""
    df_sorted = df.sort_values(["dataset", "model", "test_dice"], ascending=[True, True, False])
    return df_sorted.groupby(["dataset", "model"], as_index=False).first()

def percentify(x):
    return float(x) * 100 if pd.notnull(x) else np.nan

# ------------------------------------------------------------
# Plot 1: Beste resultater per modell (U-Net vs Encoder-Decoder)
# ------------------------------------------------------------
def plot_best_per_model(df):
    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset]
        models = sub["model"].unique()
        if len(models) == 0:
            continue

        plt.figure(figsize=(9, 5))
        x = np.arange(len(METRICS))
        width = 0.35 if len(models) == 2 else 0.8 / max(1, len(models))

        for i, (_, row) in enumerate(sub.iterrows()):
            values = [percentify(row[m]) for m in METRICS]
            plt.bar(x + i * width, values, width=width, label=f"{row['model']} ({row['loss']})")

        plt.xticks(x + (len(models)-1)*width/2, [m.replace("test_", "").upper() for m in METRICS])
        plt.ylabel("Score (%)")
        plt.title(f"Best per model — {dataset}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        outpath = OUTDIR / f"best_per_model_{dataset}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {outpath}")

# ------------------------------------------------------------
# Plot 2: U-Net — sammenligning av loss-funksjoner
# ------------------------------------------------------------
def plot_unet_loss_comparison(df):
    for dataset in sorted(df["dataset"].unique()):
        ds = df[(df["dataset"] == dataset) & (df["model"].str.lower() == "unet")]
        if ds.empty:
            continue

        ds = ds.sort_values("test_dice", ascending=False)
        losses = ds["loss"].tolist()

        # Barplot for hver metrikk
        for metric in METRICS:
            plt.figure(figsize=(9, 5))
            vals = [percentify(ds[ds["loss"] == loss][metric].values[0]) for loss in losses]
            plt.bar(np.arange(len(losses)), vals)
            plt.xticks(range(len(losses)), losses)
            plt.ylabel("Score (%)")
            plt.title(f"U-Net — {metric.replace('test_', '').upper()} by loss — {dataset}")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            outpath = OUTDIR / f"unet_{metric}_by_loss_{dataset}.png"
            plt.savefig(outpath, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: {outpath}")

        # Lineplot: Train/Val/Test Dice per loss
        x = np.arange(len(losses))
        train = [ds[ds["loss"] == loss]["best_train_dice"].values[0] for loss in losses]
        val = [ds[ds["loss"] == loss]["best_val_dice"].values[0] for loss in losses]
        test = [ds[ds["loss"] == loss]["test_dice"].values[0] for loss in losses]

        plt.figure(figsize=(9, 5))
        plt.plot(x, [v*100 for v in train], marker="o", label="Train Dice")
        plt.plot(x, [v*100 for v in val], marker="o", label="Val Dice")
        plt.plot(x, [v*100 for v in test], marker="o", label="Test Dice")
        plt.xticks(x, losses)
        plt.ylabel("Dice (%)")
        plt.title(f"U-Net — Dice (Train/Val/Test) by loss — {dataset}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = OUTDIR / f"unet_dice_train_val_test_{dataset}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {outpath}")

# ------------------------------------------------------------
# Eksporter tabeller (CSV)
# ------------------------------------------------------------
def export_best_tables(df):
    best = select_best_per_model(df)
    best.to_csv(OUTDIR / "best_per_model.csv", index=False)
    print(f"✓ Saved: {OUTDIR / 'best_per_model.csv'}")

    for dataset in sorted(df["dataset"].unique()):
        ds = df[(df["dataset"] == dataset) & (df["model"].str.lower() == "unet")]
        if ds.empty:
            continue
        ds.to_csv(OUTDIR / f"unet_by_loss_{dataset}.csv", index=False)
        print(f"✓ Saved: {OUTDIR / f'unet_by_loss_{dataset}.csv'}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not SUMMARY_CSV.exists():
        print(f"⚠️  Finner ikke CSV-fil: {SUMMARY_CSV}")
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_summary()
    if df.empty:
        print("⚠️  Ingen data funnet i CSV.")
        return

    print(f"📊 Laster data for {len(df)} eksperimenter...")
    plot_best_per_model(select_best_per_model(df))
    plot_unet_loss_comparison(df)
    export_best_tables(df)
    print("\n✅ Ferdig! Alle figurer og tabeller er lagret i:")
    print(f"   {OUTDIR}")

if __name__ == "__main__":
    main()
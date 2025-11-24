#!/usr/bin/env bash
set -euo pipefail

# Generate UMAP plots (buffer + generated) and per-class buffer plots
# Uses valuators saved in image/logs and writes figures to image/plots

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="python"

VAL_DIR="$DIR/logs"
OUT_DIR="$DIR/plots"

mkdir -p "$OUT_DIR"

# Avoid thread issues in some environments
export OMP_NUM_THREADS=1

# Combined all-classes UMAP (buffer), fixed params and high-contrast colormap
$PY "$DIR/plot_pca_values.py" \
  --mode buffer \
  --valuator-dir "$VAL_DIR" \
  --output-dir "$OUT_DIR" \
  --dataset fashion_mnist \
  --neighbors 30 \
  --min-dists 0.6 \
  --cmap viridis

# (Removed) per-class buffer UMAP plots

# Many generated images UMAP (combined plots), fixed params
$PY "$DIR/plot_pca_values_generated.py" \
  --mode gen \
  --valuator-dir "$VAL_DIR" \
  --output-dir "$OUT_DIR" \
  --dataset fashion_mnist \
  --gen-per-class 5000 \
  --gen-fit-cap 800 \
  --neighbors 30 \
  --min-dist 0.6 \
  --cmap viridis

echo "UMAP plots generated in: $OUT_DIR"



#!/usr/bin/env python3
"""Plot PCA of MANY generated images colored by saved valuator predictions.

- Fits PCA on the full (train+test) dataset for consistent axes
- Trains a GaussianGenerator on a subsample of the train set per class
- Generates many images per class, predicts values using saved valuators
- Saves one plot per stage (seed-tagged valuators supported)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import os
import re
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from umap import UMAP

from config import Config
from environment import MNISTStageLoader
from generator import GaussianGenerator


def build_full_dataset(loader: MNISTStageLoader) -> tuple[np.ndarray, np.ndarray]:
    train_imgs, train_lbls = zip(*(loader.train_dataset[i] for i in range(len(loader.train_dataset))))
    X_train = torch.stack(train_imgs).view(-1, 28 * 28).cpu().numpy().astype(np.float32)
    y_train = np.array(train_lbls, dtype=np.int64)
    test_imgs, test_lbls = zip(*(loader.test_val_dataset[i] for i in range(len(loader.test_val_dataset))))
    X_test = torch.stack(test_imgs).view(-1, 28 * 28).cpu().numpy().astype(np.float32)
    y_test = np.array(test_lbls, dtype=np.int64)
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    return X_all, y_all


def compute_class_means(X_all: np.ndarray, y_all: np.ndarray) -> Dict[int, np.ndarray]:
    means: Dict[int, np.ndarray] = {}
    for cls in range(Config.NUM_CLASSES):
        means[cls] = X_all[y_all == cls].mean(axis=0)
    return means


def fit_generator_from_train(loader: MNISTStageLoader, per_class_cap: int = 1000, n_clusters: int = 5) -> GaussianGenerator:
    # Subsample the train set up to per_class_cap per class for generator fitting
    class_to_imgs = {c: [] for c in range(Config.NUM_CLASSES)}
    for i in range(len(loader.train_dataset)):
        img, lbl = loader.train_dataset[i]
        lbl_int = int(lbl)
        if len(class_to_imgs[lbl_int]) < per_class_cap:
            class_to_imgs[lbl_int].append((img.numpy(), lbl_int))
        # Quick exit if we've reached caps for all classes
        if all(len(v) >= per_class_cap for v in class_to_imgs.values()):
            break
    buffer_samples = [pair for pairs in class_to_imgs.values() for pair in pairs]
    gen = GaussianGenerator(n_clusters=n_clusters)
    gen.fit(buffer_samples)
    return gen


def generate_many(gen: GaussianGenerator, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for c in sorted(gen.class_cluster_params.keys()):
        imgs, labs = gen.generate(c, num_samples=per_class)
        X_list.append(imgs.view(imgs.size(0), -1).cpu().numpy().astype(np.float32))
        y_list.append(labs.cpu().numpy().astype(np.int64))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def plot_values(coords: np.ndarray, values: np.ndarray, out_path: Path, title: str, *, cmap: str = "magma", point_size: int = 8, alpha: float = 0.7, vmin: float | None = None, vmax: float | None = None) -> None:
    plt.figure(figsize=(5, 4))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=point_size, alpha=alpha, edgecolors="none", vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label="Value")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    # No title
    # Annotate mean value in top-left
    ax = plt.gca()
    ax.text(0.06, 0.95, f"Mean={float(np.mean(values)):.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))
    ax.text(0.06, 0.89, f"STD={float(np.std(values)):.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_per_class_grid(coords: np.ndarray,
                        values: np.ndarray,
                        labels: np.ndarray,
                        out_path: Path,
                        *,
                        title: str,
                        cmap: str = "magma",
                        point_size: int = 8,
                        alpha: float = 0.7,
                        vmin: float | None = None,
                        vmax: float | None = None) -> None:
    if vmin is None or vmax is None:
        vmin, vmax = float(values.min()), float(values.max())
    fig, axes = plt.subplots(2, 5, figsize=(12, 5), squeeze=False)
    last_sc = None
    for cls in range(10):
        r, c = divmod(cls, 5)
        ax = axes[r][c]
        mask = labels == cls
        if np.any(mask):
            last_sc = ax.scatter(coords[mask, 0], coords[mask, 1], c=values[mask], cmap=cmap,
                                 s=point_size, alpha=alpha, edgecolors="none",
                                 vmin=vmin, vmax=vmax)
        # star at class mean in embedded space
        if np.any(mask):
            mean_xy = coords[mask].mean(axis=0)
            ax.scatter([mean_xy[0]], [mean_xy[1]], marker="*", s=180, c="red", edgecolors="black", linewidths=0.8)
        ax.set_title(f"Class {cls}")
        ax.set_xticks([]); ax.set_yticks([])
    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes.ravel().tolist(), label="Value", fraction=0.035, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("buffer", "gen"), default="gen", help="Which valuators to use: _buffer or _generated")
    p.add_argument("--valuator-dir", type=str, default=None, help="Directory with valuators; default image/logs")
    p.add_argument("--output-dir", type=str, default=None, help="Where to save plots; default image/plots")
    p.add_argument("--dataset", choices=("fashion_mnist", "mnist"), default="fashion_mnist")
    p.add_argument("--gen-fit-cap", type=int, default=800, help="Max train samples per class to fit generator")
    p.add_argument("--gen-clusters", type=int, default=5, help="Clusters per class in generator")
    p.add_argument("--gen-per-class", type=int, default=2000, help="Number of generated images per class to plot")
    p.add_argument("--neighbors", type=int, default=30, help="UMAP n_neighbors")
    p.add_argument("--min-dist", type=float, default=0.6, help="UMAP min_dist")
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--point-size", type=int, default=6)
    p.add_argument("--alpha", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    setattr(Config, "DATASET", args.dataset)
    setattr(Config, "LEVELS", (0.0, 0.0))
    setattr(Config, "NOISY_TEST", False)

    here = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else here / "plots"
    valuator_dir = Path(args.valuator_dir) if args.valuator_dir else here / "logs"

    loader = MNISTStageLoader()
    X_all, y_all_full = build_full_dataset(loader)
    umap_model = UMAP(n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist, metric="euclidean", random_state=0)

    gen = fit_generator_from_train(loader, per_class_cap=args.gen_fit_cap, n_clusters=args.gen_clusters)
    Xg, yg = generate_many(gen, per_class=args.gen_per_class)
    one_hot_g = np.eye(Config.NUM_CLASSES, dtype=np.float32)[yg]
    feats_g = np.concatenate([Xg, one_hot_g], axis=1)
    # Fit UMAP on generated set to avoid transform overhead; cache coords for re-runs
    tag = f"n{args.neighbors}_d{str(args.min_dist).replace('.', 'p')}_gc{args.gen_per_class}"
    cache_coords_path = out_dir / f"umap_generated_cache_coords_{tag}.npz"
    if cache_coords_path.exists():
        data = np.load(cache_coords_path)
        coords_g = data["coords"]
    else:
        coords_g = umap_model.fit_transform(Xg)
        try:
            np.savez(cache_coords_path, coords=coords_g)
        except Exception:
            pass

    suffix = "generated" if args.mode == "gen" else "buffer"
    # autodetect valuators grouped by model (seed) and stage
    model_to_stage_paths: Dict[str, Dict[int, Path]] = {}
    for pattern in (
        f"value_estimator_trial*_stage*_{suffix}.joblib",
        f"value_estimator_seed*_stage*_{suffix}.joblib",
        f"value_estimator_stage*_{suffix}.joblib",
    ):
        for p in valuator_dir.glob(pattern):
            m_stage = re.search(r"_stage(\d+)_" + re.escape(suffix) + r"\.joblib$", p.name)
            if not m_stage:
                continue
            st = int(m_stage.group(1))
            m_trial = re.search(r"_trial(\d+)_", p.name)
            m_seed = re.search(r"_seed(\d+)_", p.name)
            if m_trial:
                model_id = f"trial{int(m_trial.group(1))}"
            elif m_seed:
                model_id = f"model{int(m_seed.group(1))}"
            else:
                base = p.stem.split("_stage")[0]
                model_id = f"{base}"
            model_to_stage_paths.setdefault(model_id, {})[st] = p

    # For each model (seed), produce its own set of plots with prefixed filenames
    for model_id, stage_paths in sorted(model_to_stage_paths.items()):
        stages = sorted(stage_paths.keys())
        values_per_stage: Dict[int, np.ndarray] = {}
        for st in stages:
            est = joblib.load(stage_paths[st])
            vals = est.predict(feats_g).astype(np.float32)
            values_per_stage[st] = vals
        if values_per_stage:
            all_vals = np.concatenate(list(values_per_stage.values()))
            vmin_all, vmax_all = float(all_vals.min()), float(all_vals.max())
        else:
            vmin_all, vmax_all = None, None
        for st in stages:
            vals = values_per_stage[st]
            # overall scatter only (no per-class plots)
            out_all = out_dir / f"{model_id}_pca_values_generated_stage{st}.png"
            plot_values(coords_g, vals, out_all, "", cmap=args.cmap, point_size=args.point_size, alpha=args.alpha, vmin=vmin_all, vmax=vmax_all)

    print(f"Saved generated PCA value plots to {out_dir}")


if __name__ == "__main__":
    main()



import numpy as np
import torch
from classifier import Classifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import * 
from typing import Optional


def _safe_clone(model):
    """Use the estimator’s own cloning method (no pickling)."""
    return model.__sklearn_clone__()    


def _ensure_initialized(model, X_pool, y_pool):
    """
    Ensure we start from a fresh, pre-initialized classifier for Shapley baselines.
    If model.initialized_ is already True, assume it's fine.
    Otherwise, create a new Classifier clone with weights copied if available.
    """
    if getattr(model, "initialized_", False):
        return model

    fresh = Classifier(
        verbose=getattr(model, "verbose", 0),
        train_split=False
    )
    fresh.initialize()
    if hasattr(model, "module_"):
        fresh.module_.load_state_dict(model.module_.state_dict())
    fresh.initialized_ = True
    return fresh


def _val_acc(model, X, y, device="mps"):
    """Compute accuracy of a sklearn-like classifier."""
    preds = model.predict(X)
    return (preds == y).mean().item()


def _finetune(model, X, y, epochs=10):
    """
    Fine-tune `model` on data (X, y) for `epochs` passes.
    If uninitialized, perform a partial_fit with class list first.
    """
    y = y.astype(np.int64)
    if not hasattr(model, "classes_"):
        model.partial_fit(X, y, classes=np.arange(10), epochs=epochs)
        return
    model.partial_fit(X, y, epochs=epochs)


def monte_carlo_shapley(
    model,
    X_pool, y_pool,
    X_val,  y_val,
    *,
    iterations: int = 50,
    steps: int = 10,
    subset_size: Optional[int] = None,
    subset_fraction: float = 0.1,
    device: str = "mps",
    random_state: int = 0
) -> np.ndarray:
    """
    Approximate Shapley values φ_i ≈ E_S[v(S ∪ {i}) − v(S)] by Monte-Carlo:
      - For each sample i:
        • repeat `iterations` times:
            – draw a random subset S ⊆ pool\{i} of size `subset_size` or fraction of pool
            – fine-tune clone on S for `steps` epochs → accuracy v_S
            – fine-tune another clone on S ∪ {i} for `steps` epochs → v_S+i
            – record marginal contribution v_S+i − v_S
        • φ_i = average marginal over iterations

    After computing φ for all i, plot the top-10 and bottom-10 valued samples.

    Returns:
        phi: np.ndarray of length N = len(X_pool)
    """
    rng = np.random.default_rng(random_state)
    N   = len(X_pool)
    phi = np.zeros(N, dtype=np.float32)
    all_idx = np.arange(N)

    for i in tqdm(range(N), desc="Shapley samples"):
        marginals = []
        others = np.delete(all_idx, i)
        for _ in range(iterations):
            # 1) sample random background S (fixed-size or fixed-fraction subset)
            if subset_size is None:
                k = max(1, int(len(others) * subset_fraction))
                S = rng.choice(others, size=k, replace=False)
            else:
                if subset_size >= len(others):
                    S = others
                else:
                    S = rng.choice(others, size=subset_size, replace=False)

            # 2) train on S → v_bg
            bg_model = _safe_clone(model)
            bg_model = _ensure_initialized(bg_model, X_pool, y_pool)
            if len(S) > 0:
                X_bg = X_pool[S]; y_bg = y_pool[S]
                perm = rng.permutation(len(S))
                _finetune(bg_model, X_bg[perm], y_bg[perm], epochs=steps)
            v_bg = _val_acc(bg_model, X_val, y_val, device)

            # 3) train on S ∪ {i} → v_bgi
            S_i = np.concatenate([S, [i]])
            model_i = _safe_clone(model)
            model_i = _ensure_initialized(model_i, X_pool, y_pool)
            X_bgi = X_pool[S_i]; y_bgi = y_pool[S_i]
            perm  = rng.permutation(len(S_i))
            _finetune(model_i, X_bgi[perm], y_bgi[perm], epochs=steps)
            v_bgi = _val_acc(model_i, X_val, y_val, device)

            marginals.append(v_bgi - v_bg)

        phi[i] = float(np.mean(marginals))

    sorted_idx = np.argsort(phi)
    bottom10   = sorted_idx[:20]
    top10      = sorted_idx[-20:][::-1]

    # Plot bottom 10
    fig, axes = plt.subplots(1, 20, figsize=(20, 2))
    for ax, idx in zip(axes, bottom10):
        img = X_pool[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{phi[idx]:.3f}")
        ax.axis('off')
    fig.suptitle("Bottom 10 Shapley Value Samples")
    plt.tight_layout()
    plt.show()

    # Plot top 10
    fig, axes = plt.subplots(1, 20, figsize=(20, 2))
    for ax, idx in zip(axes, top10):
        img = X_pool[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{phi[idx]:.3f}")
        ax.axis('off')
    fig.suptitle("Top 10 Shapley Value Samples")
    plt.tight_layout()
    plt.show()

    return phi

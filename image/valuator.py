import os
from pathlib import Path
import joblib
import logging
import numpy as np
from shapley_custom import monte_carlo_shapley
from config import Config


def train_value_estimator_with_shapley(
    estimator,
    buffer_samples,
    classifier,
    val_dataset,
    *,
    model_path = "shapley_value_estimator.joblib",
    use_existing = False
):
    """
    Train (or fine-tune) a sklearn regressor to predict Shapley values,
    with disk persistence: load existing estimator if present,
    then save the updated estimator back to disk.

    Args:
        estimator: a sklearn regressor instance (e.g., SGDRegressor).
        buffer_samples: list of (img_np, label) tuples used for Shapley.
        classifier: the classifier used to compute Shapley values.
        val_dataset: validation set for Shapley estimation.
        model_path: filesystem path to persist the estimator.

    Returns:
        The trained (and saved) estimator.
    """
    # Resolve model path relative to this module directory if not absolute
    path_obj = Path(model_path)
    if not path_obj.is_absolute():
        path_obj = Path(__file__).resolve().parent / path_obj

    # --- load existing estimator if available ---
    if path_obj.exists() and use_existing is True:
        try:
            estimator = joblib.load(str(path_obj))
            logging.info(f"Loaded Shapley estimator from '{path_obj}'")
            return estimator
        except Exception as e:
            logging.warning(f"Failed to load existing estimator: {e}; using provided instance.")

    if not buffer_samples:
        logging.warning("Buffer empty – skipping Shapley value estimator.")
        return estimator

    # hyperparameters from config
    fraction = Config.TRUE_SHAPLEY_SUBSET
    mc_iters = Config.MC_ITERS
    device = Config.DEVICE
    finetune_steps = Config.PARTIAL_FIT_EPS

    # --- build the Shapley training pool ---
    sub_size = max(1, int(len(buffer_samples) * fraction))
    subset   = buffer_samples[:sub_size]
    X_pool   = np.stack([img.flatten() for img, _ in subset]).astype(np.float32)
    y_pool   = np.array([lbl for _, lbl in subset], dtype=np.int64)

    # --- build the full validation set ---
    X_val = np.stack([img.numpy().flatten() for img, _ in val_dataset]).astype(np.float32)
    y_val = np.array([y for _, y in val_dataset], dtype=np.int64)

    # --- shuffle for Monte-Carlo stability ---
    perm = np.random.default_rng(0).permutation(len(X_pool))
    X_pool, y_pool = X_pool[perm], y_pool[perm]
    perm = np.random.default_rng(0).permutation(len(X_val))
    X_val,   y_val = X_val[perm],   y_val[perm]

    # --- compute true Shapley values ---
    shapley_vals = monte_carlo_shapley(
        classifier,
        X_pool, y_pool,
        X_val,  y_val,
        iterations=mc_iters,
        steps=finetune_steps,
        device=device
    )

    # --- prepare training data ---
    n_classes = Config.NUM_CLASSES
    one_hot   = np.eye(n_classes, dtype=np.float32)[y_pool]
    X_train   = np.concatenate([X_pool, one_hot], axis=1)

    # --- fit and persist the estimator ---
    estimator.fit(X_train, shapley_vals)
    # Always persist the freshly trained estimator; loading remains gated by use_existing
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, str(path_obj))
        logging.info(f"Saved Shapley estimator to '{path_obj}'")
    except Exception as e:
        logging.warning(f"Failed to save estimator: {e}")

    return estimator

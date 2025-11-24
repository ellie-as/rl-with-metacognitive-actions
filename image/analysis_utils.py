import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifier import Classifier
from config import Config
from generator import GaussianGenerator
from valuator import train_value_estimator_with_shapley


def _summary_stats(values: Sequence[float]) -> Dict[str, object]:
    values = [float(v) for v in values]
    count = len(values)
    if count == 0:
        return {"mean": None, "std": None, "sem": None, "count": 0}

    mean_val = float(np.mean(values))
    if count == 1:
        return {"mean": mean_val, "std": 0.0, "sem": 0.0, "count": 1}

    std_val = float(np.std(values, ddof=1))
    sem_val = float(std_val / math.sqrt(count))
    return {"mean": mean_val, "std": std_val, "sem": sem_val, "count": count}


def _rng_seed_seq(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))


def _sample_training_data(dataset: Iterable, sample_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    total = len(dataset)
    if sample_size >= total:
        indices = np.arange(total)
    else:
        indices = rng.choice(total, size=sample_size, replace=False)
    X = np.empty((len(indices), 28 * 28), dtype=np.float32)
    y = np.empty(len(indices), dtype=np.int64)
    for row, idx in enumerate(indices):
        img_tensor, label = dataset[int(idx)]
        X[row] = np.asarray(img_tensor, dtype=np.float32).reshape(-1)
        y[row] = int(label)
    return X, y


def _build_ensemble_models(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> List[Pipeline]:
    ensemble: List[Pipeline] = [
        Pipeline([
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    multi_class="multinomial",
                    solver="lbfgs",
                ),
            ),
        ]),
        Pipeline([
            ("scale", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=1000,
                    tol=1e-3,
                    random_state=_rng_seed_seq(rng),
                ),
            ),
        ]),
        Pipeline([
            ("scale", StandardScaler()),
            (
                "clf",
                KNeighborsClassifier(n_neighbors=5),
            ),
        ]),
        Pipeline([
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=15,
                    random_state=_rng_seed_seq(rng),
                ),
            ),
        ]),
    ]

    for model in ensemble:
        model.fit(X, y)
    return ensemble


def _compute_difficulties(buffer_samples: List[tuple[np.ndarray, int]], ensemble: List[Pipeline]) -> np.ndarray:
    scores = np.zeros(len(buffer_samples), dtype=np.float32)
    for idx, (img_np, label) in enumerate(buffer_samples):
        features = img_np.reshape(1, -1)
        correct = 0
        for model in ensemble:
            pred = int(model.predict(features)[0])
            if pred == int(label):
                correct += 1
        scores[idx] = correct / max(1, len(ensemble))
    return scores


def _mmr_select_indices(
    X_flat: np.ndarray,
    vals: np.ndarray,
    k: int,
    *,
    labels: Optional[np.ndarray] = None,
    balance_classes: bool = True,
) -> List[int]:
    λ = Config.LAMBDA_PARAM
    n = len(vals)
    if n == 0 or k == 0:
        return []
    k = min(k, n)

    v_min, v_max = float(vals.min()), float(vals.max())
    if math.isclose(v_min, v_max):
        rel_norm = np.zeros_like(vals, dtype=np.float32)
    else:
        rel_norm = (vals - v_min) / (v_max - v_min)

    distances = np.linalg.norm(X_flat[:, None, :] - X_flat[None, :, :], axis=2)
    d_min, d_max = float(distances.min()), float(distances.max())
    if math.isclose(d_min, d_max):
        dist_norm = np.zeros_like(distances, dtype=np.float32)
    else:
        dist_norm = (distances - d_min) / (d_max - d_min)

    candidates = set(range(n))
    selected: List[int] = []
    class_counts: Dict[int, int] = {}

    if balance_classes and labels is not None and labels.size > 0:
        unique_labels = np.unique(labels)
        quota = max(1, int(math.ceil(k / len(unique_labels))))
    else:
        quota = None

    while len(selected) < k and candidates:
        best_idx, best_score = None, -np.inf
        for i in list(candidates):
            if quota is not None and labels is not None:
                lbl = int(labels[i])
                if class_counts.get(lbl, 0) >= quota:
                    continue

            relevance = rel_norm[i]
            if not selected:
                score = relevance
            else:
                div = dist_norm[i, selected].min()
                score = λ * relevance + (1.0 - λ) * div

            if score > best_score:
                best_idx, best_score = i, score

        if best_idx is None:
            break
        selected.append(best_idx)
        candidates.remove(best_idx)
        if quota is not None and labels is not None:
            lbl = int(labels[best_idx])
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

    return selected


def _partial_fit_classifier(model: Classifier, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
    y_int = y.astype(np.int64)
    remaining_epochs = max(0, int(epochs))
    if not hasattr(model, "initialized_"):
        model.partial_fit(X, y_int, classes=np.arange(Config.NUM_CLASSES))
        remaining_epochs -= 1
    for _ in range(max(0, remaining_epochs)):
        model.partial_fit(X, y_int)


def _extract_arrays(
    buffer_samples: Sequence[tuple[np.ndarray, int]],
    indices: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    idx_list = [int(i) for i in indices]
    if not idx_list:
        return np.empty((0, 28 * 28), dtype=np.float32), np.empty(0, dtype=np.int64)
    X = np.stack([buffer_samples[i][0] for i in idx_list]).astype(np.float32)
    y = np.array([buffer_samples[i][1] for i in idx_list], dtype=np.int64)
    return X, y


def _label_counts(samples: Sequence[tuple[np.ndarray, int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, label in samples:
        lbl = int(label)
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _sample_generative_batch(
    generator: GaussianGenerator,
    label_counts: Dict[int, int],
    batch_size: int,
    rng: np.random.Generator,
) -> List[tuple[np.ndarray, int]]:
    if batch_size <= 0 or not label_counts:
        return []

    labels = np.array(list(label_counts.keys()), dtype=np.int64)
    weights = np.array([label_counts[int(lbl)] for lbl in labels], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return []
    probs = weights / total
    counts = rng.multinomial(batch_size, probs)

    generated: List[tuple[np.ndarray, int]] = []
    for lbl, count in zip(labels, counts):
        if count <= 0:
            continue
        try:
            imgs, labels_tensor = generator.generate(int(lbl), num_samples=int(count))
        except Exception:
            continue
        img_np = imgs.cpu().numpy()
        lbl_np = labels_tensor.cpu().numpy()
        for idx in range(img_np.shape[0]):
            generated.append((img_np[idx], int(lbl_np[idx])))

    return generated


def _select_dvn_indices(
    classifier: Classifier,
    estimator: SGDRegressor,
    buffer_samples: Sequence[tuple[np.ndarray, int]],
    val_dataset,
    rng: np.random.Generator,
) -> Tuple[List[int], SGDRegressor]:
    if not buffer_samples:
        return [], estimator

    order = rng.permutation(len(buffer_samples))
    shuffled = [buffer_samples[i] for i in order]
    estimator = train_value_estimator_with_shapley(
        estimator,
        shuffled,
        classifier,
        val_dataset,
    )

    X_flat = np.stack([img.flatten() for img, _ in buffer_samples]).astype(np.float32)
    labels = np.array([label for _, label in buffer_samples], dtype=np.int64)
    one_hot = np.eye(Config.NUM_CLASSES, dtype=np.float32)[labels]
    features = np.concatenate([X_flat, one_hot], axis=1)
    values = estimator.predict(features)
    k = min(Config.NUM_TO_SELECT, len(buffer_samples))
    selected = _mmr_select_indices(X_flat, values, k, labels=labels)
    return selected, estimator


def _select_random_indices(length: int, k: int, rng: np.random.Generator) -> List[int]:
    if length == 0 or k == 0:
        return []
    k = min(k, length)
    return list(rng.choice(length, size=k, replace=False))


def _select_extreme_indices(difficulty: np.ndarray, k: int, *, hardest: bool) -> List[int]:
    if difficulty.size == 0 or k == 0:
        return []
    k = min(k, difficulty.size)
    order = np.argsort(difficulty)
    if hardest:
        return list(order[:k])
    return list(order[-k:])


def _compute_sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."



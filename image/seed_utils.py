import hashlib
import random
from typing import Optional

import numpy as np
import torch


def seed_all(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        manual_seed = getattr(torch.mps, "manual_seed", None)
        if manual_seed is not None:
            manual_seed(seed)

    try:  # pragma: no cover - optional dependency
        from faker import Faker
        Faker.seed(seed)
    except Exception:
        pass


def derive_seed(base_seed: int, *components: object, modulo: int = 2 ** 32) -> int:
    """
    Derive a deterministic but distinct seed from a base seed and components.

    Keeps related experiments correlated to the same base seed while ensuring
    different modes/trials land on independent streams.
    """

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(int(base_seed).to_bytes(8, "big", signed=False))
    for item in components:
        hasher.update(repr(item).encode("utf-8"))
    digest = int.from_bytes(hasher.digest(), "big", signed=False)
    if modulo <= 0:
        return digest
    return digest % modulo

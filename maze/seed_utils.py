import hashlib
import random
from typing import Optional

import numpy as np
import torch


def seed_all(seed: Optional[int]) -> None:
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

    try:  # optional dependency
        from faker import Faker
        Faker.seed(seed)
    except Exception:
        pass


def derive_seed(base_seed: int, *components: object, modulo: int = 2 ** 32) -> int:
    """Derive a deterministic child seed from a base seed and arbitrary components."""

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(int(base_seed).to_bytes(8, "big", signed=False))
    for item in components:
        hasher.update(repr(item).encode("utf-8"))
    digest = int.from_bytes(hasher.digest(), "big", signed=False)
    if modulo <= 0:
        return digest
    return digest % modulo

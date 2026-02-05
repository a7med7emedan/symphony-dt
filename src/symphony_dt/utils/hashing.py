from __future__ import annotations

import hashlib


def stable_int_hash(value: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be > 0")
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % modulo

"""Episode split files (same JSON schema as vm_diffusion/config/splits/*.json)."""

from __future__ import annotations

import json
from pathlib import Path


def load_split(path: str | Path) -> dict[str, list[str]]:
    d = json.loads(Path(path).read_text())
    out = {}
    for key in ("train", "val", "test"):
        eps = list(d.get(key, []))
        for e in eps:
            if not e.endswith(".pt"):
                raise ValueError(f"{path}: split '{key}' entry '{e}' must end in .pt")
        out[key] = eps
    if set(out["train"]) & set(out["test"]):
        raise ValueError(f"{path}: train and test overlap")
    if set(out["train"]) & set(out["val"]):
        raise ValueError(f"{path}: train and val overlap")
    return out

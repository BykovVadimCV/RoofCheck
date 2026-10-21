"""Train / val / test split for real photos: by object, never by photo.

All photos of one object (every vantage point, every date) go to the same split, so the test set
measures how the module works on roofs it has never seen. The assignment is a stable hash of the
object id: adding new objects never moves old ones between splits. The test set is frozen once
fixed (tests/frozen_test.json); later objects only go to train/val.

    python3 -m roofcheck.split data/objects            -> data/splits.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SHARES = {"test": 0.2, "val": 0.1}          # rest -> train


def bucket(object_id: str) -> float:
    h = hashlib.sha256(object_id.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign(object_id: str, frozen_test: set[str] | None = None) -> str:
    if frozen_test is not None:
        if object_id in frozen_test:
            return "test"
        b = bucket(object_id)
        return "val" if b < SHARES["val"] / (1 - SHARES["test"]) else "train"
    b = bucket(object_id)
    if b < SHARES["test"]:
        return "test"
    if b < SHARES["test"] + SHARES["val"]:
        return "val"
    return "train"


def build(objects_dir: Path, frozen: Path | None = None) -> dict:
    frozen_test = set(json.loads(frozen.read_text())) if frozen and frozen.exists() else None
    out = {"train": [], "val": [], "test": []}
    for od in sorted(p for p in objects_dir.iterdir() if p.is_dir()):
        out[assign(od.name, frozen_test)].append(od.name)
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/objects")
    res = build(root, root.parent / "frozen_test.json")
    (root.parent / "splits.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print({k: len(v) for k, v in res.items()})

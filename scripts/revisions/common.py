"""Small, explicit data/provenance helpers for the reviewer experiments."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_id(name):
    match = re.fullmatch(r"PKC(\d+)\..+", str(name))
    if not match:
        raise ValueError(f"Cannot infer parent from {name!r}; expected PKC<number>.<variant>")
    return f"PKC{int(match.group(1))}"


def canonical_id(name):
    match = re.fullmatch(r"PKC(\d+)\.(\d+)", str(name))
    return f"PKC{int(match.group(1))}.{int(match.group(2))}" if match else str(name)


def load_snapshot_ids(path):
    df = pd.read_pickle(path)
    col = next((c for c in ("Enzyme", "name") if c in df), None)
    if col is None:
        raise ValueError(f"No identifier column in snapshot {path}")
    return set(df[col].map(canonical_id))


def load_snapshot_sequences(path):
    # Only load trusted repository/user-supplied pickles, never arbitrary downloads.
    df = pd.read_pickle(path)
    col = next((c for c in ("Aligned_Sequence", "sequence", "Sequence") if c in df), None)
    if col is None:
        raise ValueError(f"No sequence column in snapshot {path}")
    return set(df[col].astype(str).str.replace("-", "", regex=False))


def json_write(path, data):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, Path):
            return str(x)
        return x
    Path(path).write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")

import numpy as np
from typing import Dict, List
import json
import os

def safe_load_numpy_array(path: str) -> np.ndarray:
    """
    Load an npy file robustly: allow_pickle=True only for trusted files,
    and if we get an object-dtype array of per-node vectors, try to
    convert it to a 2D numeric array (padding if necessary).
    """
    # Trusted file assumption: only use allow_pickle=True for files you created/trust
    arr = np.load(path, allow_pickle=True)

    # If it's already a numeric ndarray, return as float32
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return arr.astype(np.float32)

    # If object array: try to stack (handles list/array-of-vectors)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            # Convert every element to 1D numeric array and pad to same length
            vectors = [np.asarray(x, dtype=np.float32) for x in arr]
            lengths = [v.shape[0] for v in vectors]
            if len(set(l.shape for l in vectors)) == 1:
                return np.vstack(vectors)
            # pad to max length
            max_len = max(lengths)
            padded = np.zeros((len(vectors), max_len), dtype=np.float32)
            for i, v in enumerate(vectors):
                padded[i, :v.shape[0]] = v
            return padded
        except Exception as e:
            raise RuntimeError(f"Could not convert object array from {path} to numeric matrix: {e}")

    # If loaded something else (list/dict), try to coerce to array
    try:
        return np.asarray(arr, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Unsupported semantic features format in {path}: {e}")

def parse_hetionet_triplets(path: str) -> List[Dict]:
    triplets = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split('	')
            if len(parts) < 3:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
            h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
            triplets.append({'source': h, 'target': t, 'relation': r})
    return triplets

    
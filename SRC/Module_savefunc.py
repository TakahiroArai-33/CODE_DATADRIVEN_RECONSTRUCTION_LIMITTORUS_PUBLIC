#!/usr/bin/env python3
# coding: utf-8


import os
import sys
import numpy as np
from typing import Dict, Any

def save_npz_atomic(filename: str, base: Dict[str, Any], 
                    updates: Dict[str, Any], *, message: str | None = None) -> None:
    """Atomically save ``base`` and ``updates`` to a closed NPZ path."""
    tmp_path = f"{filename}.tmp"
    try:
        with open(tmp_path, "wb") as tmp_file:
            np.savez(tmp_file, **base, **updates)
        os.replace(tmp_path, filename)
        if message:
            print()
            print(message, flush=True)
    except Exception as exc:
        print(f"Warning: failed to save npz file to {filename}: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



if __name__ == '__main__':
    """Minimal usage example."""

    x = np.array([1,2,3])
    
    data1 = np.load("hogehoge.npz")
    target = dict(data1.items())
    data1.close()

    save_npz_atomic(
        filename="dataname.npz",
        base=target,
        updates={"add": x},
        message="Saved `add` to dataname.npz"
        )

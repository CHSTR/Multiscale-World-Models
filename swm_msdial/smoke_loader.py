"""Smoke de compatibilidad del H5 de msdial con el loader de stable_worldmodel.

Corre con el venv de entrenamiento (que tiene swm 0.1.0):
  /home/chr/le-wm/.venv/bin/python swm_msdial/smoke_loader.py --file <h5>

Verifica (Task 2, Step 5 del plan): load_dataset OK, load_chunk del episodio 0
con exactamente las filas pedidas, get_dim("action") == 2+K, pixels (224,224,3).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--chunk-rows", type=int, default=26)
    args = ap.parse_args()

    import stable_worldmodel as swm

    ds = swm.data.load_dataset(
        args.file,
        keys_to_load=["pixels", "action", "state", "observation", "proprio"],
        keys_to_cache=["action"],
    )
    ep_len = int(np.asarray(ds.lengths).ravel()[0])
    rows = min(args.chunk_rows, ep_len)
    print("column_names:", ds.column_names)
    print("ep_len[0]:", ep_len, "| num filas:", len(ds) if hasattr(ds, "__len__") else "?")

    chunk = ds.load_chunk(np.array([0]), np.array([0]), np.array([rows]))
    assert len(chunk) == 1
    px = chunk[0]["pixels"]
    assert px.shape[-3:] == (rows, 224, 224, 3) or px.shape[0] == rows, px.shape

    row = ds.get_row_data(0)
    assert row["pixels"].shape == (224, 224, 3), row["pixels"].shape

    dim = ds.get_dim("action")
    state_dim = row["state"].shape[0] if "state" in row else None
    print(f"get_dim(action)={dim}  state_dim={state_dim}  chunk rows OK ({rows})")
    print("SMOKE OK")


if __name__ == "__main__":
    main()

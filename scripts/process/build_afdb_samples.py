#!/usr/bin/env python3
"""Build AFDB training sample table from kept_all ∩ pack_index.

Writes ``samples.pkl`` under the training_bridge meta dir. AFDB monomers each
have a unique cluster_id, so ClusterSampler weights are uniform — we precompute
equal normalized weights and skip loading 6.46M Record objects at train init.

Copies ``pack_index.sqlite`` to node-local temporary storage before reading.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kept", type=Path, required=True)
    p.add_argument("--pack-index", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "samples.pkl"
    t0 = time.time()

    print(f"reading kept from {args.kept} ...", flush=True)
    kept = pd.read_csv(
        args.kept, sep="\t", usecols=["accession"], dtype={"accession": "string"}
    )
    print(f"  kept={len(kept)} ({time.time() - t0:.1f}s)", flush=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="packidx_", dir="/tmp"))
    local = tmpdir / "pack_index.sqlite"
    try:
        print(f"copying pack_index → {local} ...", flush=True)
        shutil.copy2(args.pack_index, local)
        print(f"  copy done ({time.time() - t0:.1f}s)", flush=True)
        con = sqlite3.connect(str(local))
        indexed = pd.read_sql_query("SELECT accession FROM pack_index", con)
        con.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"  indexed={len(indexed)} ({time.time() - t0:.1f}s)", flush=True)
    merged = kept.merge(indexed, on="accession", how="inner")
    miss = len(kept) - len(merged)
    print(f"  in_both={len(merged)} missing_from_index={miss}", flush=True)
    if merged.empty:
        raise SystemExit("no overlapping accessions")

    n = len(merged)
    samples = pd.DataFrame(
        {
            "record_id": merged["accession"].astype("string"),
            "chain_id": np.zeros(n, dtype=np.int32),
            "interface_id": pd.array([pd.NA] * n, dtype="Int32"),
            "weight": np.full(n, 1.0 / n, dtype=np.float64),
        }
    )
    print(f"writing {out} ...", flush=True)
    samples.to_pickle(out)
    summary = {
        "n_samples": n,
        "n_kept": int(len(kept)),
        "n_indexed": int(len(indexed)),
        "n_missing_from_index": int(miss),
        "path": str(out),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (args.outdir / "SAMPLES_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

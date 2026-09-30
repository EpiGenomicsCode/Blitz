"""Convert a directory of ColabFold ``.a3m`` MSAs into Boltz-2 ``MSA`` ``.npz``.

Rewrite of ``scripts/process/msa.py`` that:
  * Drops the Redis dependency. Taxonomy (for UniRef species pairing) is loaded
    from an optional on-disk pickle ``{uniref_id: taxonomy_id}`` (``--taxonomy``);
    with no taxonomy the MSA is unpaired (``taxonomy_id = -1``), which matches
    ``parse_a3m(..., taxonomy=None)``.
  * Names each output ``{stem}.npz`` where ``stem`` is the a3m filename stem. Our
    query FASTA uses the sequence sha256 as the header, so ColabFold emits
    ``{hash}.a3m`` -> ``{hash}.npz``, matching ``chain.msa_id`` from ``rcsb_v2.py``.
"""

import argparse
import multiprocessing as mp
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from boltz.data.parse.a3m import parse_a3m

# Populated in the parent before forking so workers inherit via copy-on-write.
_TAXONOMY: Optional[dict] = None
_OUTDIR: Optional[Path] = None
_MAX_SEQS: Optional[int] = None


def _process_one(path: Path) -> tuple[str, str]:
    """Parse one a3m -> MSA.npz (skip if already done)."""
    assert _OUTDIR is not None
    out_path = _OUTDIR / f"{path.stem}.npz"
    if out_path.exists():
        return path.name, "skip"
    try:
        msa = parse_a3m(path, taxonomy=_TAXONOMY, max_seqs=_MAX_SEQS)
        np.savez_compressed(out_path, **asdict(msa))
    except Exception as e:  # noqa: BLE001
        return path.name, f"fail:{e}"
    return path.name, "ok"


def main(args: argparse.Namespace) -> None:
    """Convert all a3m files to npz."""
    global _TAXONOMY, _OUTDIR, _MAX_SEQS

    args.outdir.mkdir(parents=True, exist_ok=True)
    _OUTDIR = args.outdir
    _MAX_SEQS = args.max_seqs

    if args.taxonomy is not None:
        with Path(args.taxonomy).open("rb") as f:
            _TAXONOMY = pickle.load(f)  # noqa: S301
        print(f"Loaded taxonomy: {len(_TAXONOMY)} entries.")  # noqa: T201

    data = [p for p in args.msadir.rglob("*.a3m*") if not p.name.endswith(".tmp")]
    print(f"Found {len(data)} MSAs.")  # noqa: T201

    num_processes = max(1, min(args.num_processes, len(data) or 1))
    if num_processes > 1 and len(data) > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_process_one, data, chunksize=4),
                    total=len(data),
                )
            )
        failed = [r for r in results if r[1].startswith("fail:")]
        for name, msg in failed[:10]:
            print(f"Failed {name}: {msg}")  # noqa: T201
        if len(failed) > 10:
            print(f"... and {len(failed) - 10} more failures")  # noqa: T201
    else:
        for path in tqdm(data):
            name, status = _process_one(path)
            if status.startswith("fail:"):
                print(f"Failed {name}: {status}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColabFold a3m -> Boltz-2 MSA npz.")
    parser.add_argument("--msadir", type=Path, required=True, help="Dir of .a3m(.gz) files.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output dir of {hash}.npz.")
    parser.add_argument("--num-processes", type=int, default=mp.cpu_count())
    parser.add_argument("--max-seqs", type=int, default=16384, help="Cap sequences per MSA.")
    parser.add_argument("--taxonomy", type=Path, default=None, help="Optional pickle {uniref_id: tax_id}.")
    args = parser.parse_args()
    main(args)

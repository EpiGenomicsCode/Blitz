"""Build the unique-protein-sequence query FASTA for MSA generation.

Scans ``processed/sequences/{id}.json`` sidecars, collects every distinct
protein sequence, and writes one FASTA record per unique sequence whose header
is the sequence sha256 (== ``msa_id`` from ``rcsb_v2.py``). ColabFold then emits
``{hash}.a3m`` which converts to ``{hash}.npz`` for the training loader.
"""

import argparse
import json
from pathlib import Path

from boltz.data import const

PROTEIN = const.chain_type_ids["PROTEIN"]


def main(args: argparse.Namespace) -> None:
    """Collect unique protein sequences and write the query FASTA."""
    seqdir: Path = args.seqdir
    seen: dict[str, str] = {}
    n_files = 0
    for jf in seqdir.rglob("*.json"):
        n_files += 1
        try:
            with jf.open() as f:
                chains = json.load(f)
        except Exception:  # noqa: BLE001, PERF203
            continue
        for _, info in chains.items():
            if int(info.get("mol_type", -1)) != PROTEIN:
                continue
            h = info.get("hash")
            seq = info.get("sequence")
            if h and seq and h not in seen:
                seen[h] = seq

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for h, seq in seen.items():
            f.write(f">{h}\n{seq}\n")
    print(f"Scanned {n_files} sidecars; wrote {len(seen)} unique protein sequences to {args.out}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unique protein query FASTA.")
    parser.add_argument("--seqdir", type=Path, required=True, help="processed/sequences dir.")
    parser.add_argument("--out", type=Path, required=True, help="Output FASTA path.")
    args = parser.parse_args()
    main(args)

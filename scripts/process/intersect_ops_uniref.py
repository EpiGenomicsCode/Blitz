#!/usr/bin/env python3
"""Intersect OPS shard scans with UniRef30 accessions (+ optional AFDB id set).

Inputs:
  --scan-dir   TSV shards from scan_ops_shards.py
               cols: ops_cluster_id accession seq_len sequence zip_name a3m_member
  --uniref-accessions  one accession per line (UniRef30 mapping, UniProt only)
  --afdb-accessions    optional: AFDB accession_ids.csv or one-acc-per-line

Outputs (under --outdir):
  candidates.tsv     UniRef30 ∩ OPS (+ AFDB if provided)
  summary.json
  val_exact_matches.tsv   informational: candidates whose sequence equals a val chain
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_accessions(path: Path) -> set[str]:
    accs: set[str] = set()
    with path.open() as f:
        first = f.readline()
        # accession_ids.csv: ACC,start,end,AF-ACC-F1,version
        if "," in first and not first.startswith("#"):
            accs.add(first.split(",", 1)[0].strip())
            for line in f:
                accs.add(line.split(",", 1)[0].strip())
        else:
            if first.strip():
                accs.add(first.strip())
            for line in f:
                s = line.strip()
                if s:
                    accs.add(s)
    return accs


def load_val_hashes(path: Path) -> dict[str, str]:
    """hash -> example val id (from val_protein_hashes.txt: hash\\tid or just hash)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            if len(parts) == 1:
                out[parts[0]] = parts[0]
            else:
                out[parts[0]] = parts[1]
    return out


def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scan-dir", type=Path, required=True)
    p.add_argument("--uniref-accessions", type=Path, required=True)
    p.add_argument("--afdb-accessions", type=Path, default=None)
    p.add_argument("--val-hashes", type=Path, default=None)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading UniRef30 accessions from {args.uniref_accessions} ...")
    uniref = load_accessions(args.uniref_accessions)
    print(f"  {len(uniref):,} accessions")

    afdb = None
    if args.afdb_accessions is not None:
        print(f"Loading AFDB accessions from {args.afdb_accessions} ...")
        afdb = load_accessions(args.afdb_accessions)
        print(f"  {len(afdb):,} accessions")

    val_hashes = load_val_hashes(args.val_hashes) if args.val_hashes else {}
    print(f"Val sequence hashes: {len(val_hashes)}")

    scan_files = sorted(args.scan_dir.glob("shard_*.tsv")) + sorted(
        p for p in args.scan_dir.glob("*.tsv") if not p.name.startswith("shard_")
    )
    # de-dup
    seen_paths = set()
    files = []
    for fp in scan_files:
        if fp not in seen_paths:
            files.append(fp)
            seen_paths.add(fp)
    print(f"Scan files: {len(files)}")

    n_ops = n_ur = n_afdb = n_val = 0
    # Dedup by accession (keep first / shortest cluster id)
    best: dict[str, str] = {}

    cand_path = args.outdir / "candidates.tsv"
    val_path = args.outdir / "val_exact_matches.tsv"
    with cand_path.open("w") as cout, val_path.open("w") as vout:
        cout.write(
            "accession\tops_cluster_id\tseq_len\tsequence\tzip_name\ta3m_member\n"
        )
        vout.write(
            "accession\tval_ref\tseq_len\tseq_sha256\tops_cluster_id\tzip_name\n"
        )
        for fp in files:
            with fp.open() as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 6:
                        continue
                    cluster_id, acc, seq_len_s, seq, zip_name, a3m_member = parts[:6]
                    n_ops += 1
                    if acc not in uniref:
                        continue
                    n_ur += 1
                    if afdb is not None and acc not in afdb:
                        continue
                    if afdb is not None:
                        n_afdb += 1
                    if acc in best:
                        continue
                    best[acc] = cluster_id
                    cout.write(
                        f"{acc}\t{cluster_id}\t{seq_len_s}\t{seq}\t{zip_name}\t{a3m_member}\n"
                    )
                    h = seq_hash(seq)
                    if h in val_hashes:
                        n_val += 1
                        vout.write(
                            f"{acc}\t{val_hashes[h]}\t{seq_len_s}\t{h}\t{cluster_id}\t{zip_name}\n"
                        )

    summary = {
        "ops_rows": n_ops,
        "uniref_overlap_rows": n_ur,
        "unique_candidates": len(best),
        "afdb_filtered": afdb is not None,
        "afdb_overlap_rows": n_afdb if afdb is not None else None,
        "val_exact_matches": n_val,
        "candidates_tsv": str(cand_path),
        "val_matches_tsv": str(val_path),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

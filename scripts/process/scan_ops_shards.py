#!/usr/bin/env python3
"""Scan OpenProteinSet uniclust30_unfiltered zips for MSA query accessions.

For each ``uniclust30.a3m`` in a shard zip, emit one TSV row:
    ops_cluster_id  accession  seq_len  sequence  zip_name  a3m_member

Paper A.1.3 §7 starts from UniRef30 ∩ OpenFold UniClust MSAs. OPS folder IDs
are *not* UniRef30 ffindex keys; the join key is the representative UniProt
accession parsed from the a3m header (``#tr|ACC|...`` / ``>tr|ACC|...``).

Supports Slurm arrays via ``--index N`` against ``shards_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

# UniProt accession-ish token inside |ACC| headers (also catches UniParc UPI*)
ACC_RE = re.compile(r"(?:^#?|>)(?:tr|sp)\|([A-Z0-9]+)\|")
ACC_FALLBACK_RE = re.compile(r"\|([A-Z0-9]{6,10})\|")


def parse_query(a3m_bytes: bytes) -> tuple[str | None, str]:
    """Return (accession, ungapped uppercase query sequence) from an a3m."""
    text = a3m_bytes.decode("utf-8", errors="replace")
    accession = None
    seq_parts: list[str] = []
    in_seq = False
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("#"):
            if accession is None:
                m = ACC_RE.match(line) or ACC_FALLBACK_RE.search(line)
                if m:
                    accession = m.group(1)
            continue
        if line.startswith(">"):
            if in_seq:
                break  # only first sequence (query / consensus)
            if accession is None:
                m = ACC_RE.match(line) or ACC_FALLBACK_RE.search(line)
                if m:
                    accession = m.group(1)
            in_seq = True
            continue
        if in_seq:
            # Drop insertions (lowercase) and gaps for the query sequence used
            # for AFDB / val exact-match checks. Keep only uppercase AA.
            seq_parts.append("".join(c for c in line.strip() if c.isupper()))
    return accession, "".join(seq_parts)


def scan_zip(zip_path: Path, out_fh) -> tuple[int, int]:
    n_ok = n_fail = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith("uniclust30.a3m"):
                continue
            cluster_id = name.rstrip("/").split("/")[-2]
            try:
                with zf.open(name) as fh:
                    # Query is near the top; 256 KB is plenty for header+query
                    blob = fh.read(262144)
                acc, seq = parse_query(blob)
                if not acc or not seq:
                    n_fail += 1
                    continue
                out_fh.write(
                    f"{cluster_id}\t{acc}\t{len(seq)}\t{seq}\t{zip_path.name}\t{name}\n"
                )
                n_ok += 1
            except Exception:  # noqa: BLE001
                n_fail += 1
    return n_ok, n_fail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ops-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--index", type=int, default=None, help="Shard index (Slurm array)")
    p.add_argument("--shard", type=str, default=None, help="Zip filename or key")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or (args.ops_dir / "shards_manifest.json")
    shards = json.loads(manifest.read_text())

    if args.index is not None:
        if args.index < 0 or args.index >= len(shards):
            print(f"index {args.index} out of range 0..{len(shards)-1}", file=sys.stderr)
            sys.exit(1)
        zip_name = Path(shards[args.index]["key"]).name
        zip_path = args.ops_dir / zip_name
        out_path = args.outdir / f"shard_{args.index:04d}.tsv"
    elif args.shard is not None:
        zip_name = Path(args.shard).name
        zip_path = args.ops_dir / zip_name
        out_path = args.outdir / f"{zip_path.stem}.tsv"
    else:
        print("pass --index or --shard", file=sys.stderr)
        sys.exit(2)

    if not zip_path.exists():
        print(f"missing zip: {zip_path}", file=sys.stderr)
        sys.exit(1)

    with out_path.open("w") as fh:
        n_ok, n_fail = scan_zip(zip_path, fh)
    print(f"SCAN_OK {zip_path.name} ok={n_ok} fail={n_fail} -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Process AFDB mmCIFs + OpenProteinSet MSAs into Boltz-2 StructureV2 training data.

For each kept accession (from ``filter_afdb_confidence.py``):
  1. Download AFDB v6 mmCIF (resume-safe).
  2. Parse with stock ``parse_mmcif`` (unmodified).
  3. Extract the OPS ``uniclust30.a3m`` from its zip, convert via ``parse_a3m``.
  4. Write structures/{acc}.npz, records/{acc}.json, msa_npz/{acc}.npz.

The generated records follow the RCSB processing conventions:
  method=None on StructureInfo (override_method="AFDB" at train time),
  template_ids=None, cluster_id=accession, msa_id=accession.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from boltz.data import const  # noqa: E402
from boltz.data.mol import load_canonicals  # noqa: E402
from boltz.data.parse.a3m import parse_a3m  # noqa: E402
from boltz.data.parse.mmcif import parse_mmcif  # noqa: E402
from boltz.data.types import ChainInfo, Record, StructureInfo, StructureV2  # noqa: E402

PROTEIN = const.chain_type_ids["PROTEIN"]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=180) as r, part.open("wb") as f:
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    part.rename(dest)


def finalize_manifest(outdir: Path) -> None:
    """Build manifest.json from records/*.json (same shape as rcsb_v2.finalize)."""
    records_dir = outdir / "records"
    records = []
    failed = 0
    for path in records_dir.glob("*.json"):
        try:
            with path.open() as f:
                records.append(json.load(f))
        except Exception:  # noqa: BLE001
            failed += 1
    print(f"Manifest: {len(records)} records, {failed} unreadable.")
    with (outdir / "manifest.json").open("w") as f:
        json.dump(records, f)


def extract_a3m(ops_dir: Path, zip_name: str, a3m_member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ops_dir / zip_name) as zf, zf.open(a3m_member) as src, dest.open("wb") as out:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def build_record(pdb_id: str, parsed, msa_id: str) -> Record:
    structure = parsed.data
    chains = []
    for i, chain in enumerate(structure.chains):
        mol_type = int(chain["mol_type"])
        chains.append(
            ChainInfo(
                chain_id=i,
                chain_name=str(chain["name"]),
                mol_type=mol_type,
                cluster_id=pdb_id if mol_type == PROTEIN else -1,
                msa_id=msa_id if mol_type == PROTEIN else -1,
                num_residues=int(chain["res_num"]),
                entity_id=int(chain["entity_id"]),
                template_ids=None,
            )
        )
    info = parsed.info
    info = StructureInfo(
        resolution=info.resolution,
        method=None,
        deposited=info.deposited,
        released=info.released,
        revised=info.revised,
        num_chains=info.num_chains,
        num_interfaces=info.num_interfaces,
        pH=getattr(info, "pH", None),
        temperature=getattr(info, "temperature", None),
    )
    return Record(id=pdb_id, structure=info, chains=chains, interfaces=[])


def process_one(
    row: dict,
    *,
    ops_dir: Path,
    raw_dir: Path,
    struct_dir: Path,
    record_dir: Path,
    msa_dir: Path,
    a3m_dir: Path,
    mols,
    moldir: str,
    taxonomy,
    max_seqs: int,
    keep_cif: bool,
    keep_a3m: bool,
) -> str:
    acc = row["accession"]
    struct_path = struct_dir / f"{acc}.npz"
    record_path = record_dir / f"{acc}.json"
    msa_path = msa_dir / f"{acc}.npz"
    if struct_path.exists() and record_path.exists() and msa_path.exists():
        return "skip"

    cif_path = raw_dir / f"{acc}.cif"
    a3m_path = a3m_dir / f"{acc}.a3m"
    try:
        if not cif_path.exists():
            download(row["cifUrl"], cif_path)

        if not a3m_path.exists():
            extract_a3m(ops_dir, row["zip_name"], row["a3m_member"], a3m_path)

        if not msa_path.exists():
            msa = parse_a3m(a3m_path, taxonomy=taxonomy, max_seqs=max_seqs)
            msa.dump(msa_path)

        if not struct_path.exists() or not record_path.exists():
            parsed = parse_mmcif(
                str(cif_path), mols=mols, moldir=moldir, compute_interfaces=False
            )
            structure: StructureV2 = parsed.data
            structure.dump(struct_path)
            record = build_record(acc, parsed, msa_id=acc)
            with record_path.open("w") as f:
                json.dump(record.to_dict(), f)
    finally:
        if not keep_a3m and a3m_path.exists():
            a3m_path.unlink(missing_ok=True)
        if not keep_cif and cif_path.exists():
            cif_path.unlink(missing_ok=True)
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kept", type=Path, default=None, help="kept_*.tsv from confidence filter")
    p.add_argument("--ops-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True, help="afdb/processed root")
    p.add_argument("--msa-outdir", type=Path, required=True)
    p.add_argument("--moldir", type=Path, required=True)
    p.add_argument("--taxonomy", type=Path, default=None)
    p.add_argument("--max-seqs", type=int, default=16384)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--keep-a3m", action="store_true")
    p.add_argument("--keep-cif", action="store_true", help="Keep raw CIFs (default: delete after parse)")
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()

    raw_dir = args.outdir / "raw_cif"
    struct_dir = args.outdir / "structures"
    record_dir = args.outdir / "records"
    a3m_dir = args.outdir.parent / "msa_a3m"
    for d in (raw_dir, struct_dir, record_dir, args.msa_outdir, a3m_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        finalize_manifest(args.outdir)
        return

    if args.kept is None:
        raise SystemExit("--kept is required unless --finalize-only")

    taxonomy = None
    if args.taxonomy is not None:
        import pickle

        with args.taxonomy.open("rb") as f:
            taxonomy = pickle.load(f)  # noqa: S301
        print(f"Loaded taxonomy: {len(taxonomy)} entries")

    print(f"Loading canonical mols from {args.moldir} ...")
    mols = load_canonicals(str(args.moldir))

    rows = []
    with args.kept.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for i, line in enumerate(f):
            if i < args.offset:
                continue
            if args.limit is not None and len(rows) >= args.limit:
                break
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts)))

    print(f"Processing {len(rows)} accessions ...")
    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows, 1):
        try:
            status = process_one(
                row,
                ops_dir=args.ops_dir,
                raw_dir=raw_dir,
                struct_dir=struct_dir,
                record_dir=record_dir,
                msa_dir=args.msa_outdir,
                a3m_dir=a3m_dir,
                mols=mols,
                moldir=str(args.moldir),
                taxonomy=taxonomy,
                max_seqs=args.max_seqs,
                keep_cif=args.keep_cif,
                keep_a3m=args.keep_a3m,
            )
            if status == "skip":
                n_skip += 1
            else:
                n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"FAIL {row.get('accession')}: {e}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)

    print(f"PROCESS_DONE ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()

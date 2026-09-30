"""Batch-process RCSB mmCIF files into Boltz-2 StructureV2 training data.

This is a rewrite of ``scripts/process/rcsb.py`` that wires the *v2* mmCIF parser
(``boltz.data.parse.mmcif.parse_mmcif`` -> ``StructureV2``) into a batch loop, so
the output is directly consumable by ``boltz.data.module.training_mscd``.

Key differences from the original ``rcsb.py``:
  * Emits ``StructureV2`` (atoms/bonds/residues/chains/interfaces/mask/coords/
    ensemble) instead of the Boltz-1 ``Structure``.
  * Uses a CCD ``moldir`` (dir of ``{CCD}.pkl``) instead of a Redis resource.
  * Computes ``cluster_id`` and ``msa_id`` consistently with
    ``scripts/process/cluster.py`` (protein/nucleic: sha256(sequence); ligand: CCD
    code), rather than leaving ``msa_id=""``.
  * Writes ``sequences/{id}.json`` (chain -> {mol_type, sequence, hash}) so the
    global MSA FASTA and the msa_id<->sequence join can be built afterwards.

Outputs under ``--outdir``::

    structures/{id}.npz     StructureV2
    records/{id}.json       Record
    sequences/{id}.json     {chain_name: {mol_type, sequence, hash}}
    manifest.json           list[Record]
"""

import argparse
import hashlib
import json
import multiprocessing
import traceback
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import rdkit
from p_tqdm import p_umap
from tqdm import tqdm

from boltz.data import const
from boltz.data.filter.static.filter import StaticFilter
from boltz.data.filter.static.ligand import ExcludedLigands
from boltz.data.filter.static.polymer import (
    ClashingChainsFilter,
    ConsecutiveCA,
    MinimumLengthFilter,
    UnknownFilter,
)
from boltz.data.parse.mmcif import find_interfaces, parse_mmcif
from boltz.data.types import ChainInfo, InterfaceInfo, Record

PROTEIN = const.chain_type_ids["PROTEIN"]
DNA = const.chain_type_ids["DNA"]
RNA = const.chain_type_ids["RNA"]
NONPOLYMER = const.chain_type_ids["NONPOLYMER"]
POLYMER_TYPES = {PROTEIN, DNA, RNA}


@dataclass(frozen=True, slots=True)
class PDB:
    """A raw mmCIF PDB file."""

    id: str
    path: str


def hash_sequence(seq: str) -> str:
    """Hash a sequence with sha256 (matches cluster.py)."""
    return hashlib.sha256(seq.encode()).hexdigest()


def _uncompressed_size(file: Path) -> int:
    """Return the true uncompressed size of a file.

    For ``.gz`` files, ``Path.stat().st_size`` is the *compressed* on-disk size,
    which is not comparable to the paper's ">7MB" cutoff (defined on the raw
    mmCIF text). Gzip stores the uncompressed size (mod 2**32) in the last 4
    bytes of the stream, so we can read it in O(1) without decompressing. Our
    mmCIF files are always far below 4GB, so the mod-2**32 wraparound never
    applies in practice.
    """
    if file.suffix != ".gz":
        return file.stat().st_size
    with file.open("rb") as f:
        f.seek(-4, 2)
        return int.from_bytes(f.read(4), byteorder="little")


def fetch(datadir: Path, max_file_size: Optional[int] = None) -> list[PDB]:
    """Fetch the mmCIF files, optionally skipping oversized ones (paper: >7MB)."""
    data = []
    excluded = 0
    for file in datadir.rglob("*.cif*"):
        name = file.name
        if name.endswith(".tmp"):
            continue
        # id is the part before the first dot: "1abc.cif" / "1abc.cif.gz" -> "1abc"
        pdb_id = name.split(".", 1)[0].lower()
        if max_file_size is not None and (_uncompressed_size(file) > max_file_size):
            excluded += 1
            continue
        data.append(PDB(id=pdb_id, path=str(file)))
    print(f"Found {len(data)} files; excluded {excluded} due to size.")  # noqa: T201
    return data


def _ccd_code_for_chain(structure, chain) -> Optional[str]:
    """Return the CCD code of the first residue of a (nonpolymer) chain."""
    res_start = int(chain["res_idx"])
    if chain["res_num"] < 1:
        return None
    return str(structure.residues[res_start]["name"])


def build_record(
    pdb_id: str,
    parsed,
    clusters: Optional[dict],
    msa_mode: str,
) -> tuple[Record, dict]:
    """Build the Record and the chain-sequence sidecar for one structure."""
    structure = parsed.data
    sequences = parsed.sequences  # {chain_name: sequence} for polymers

    chain_info: list[ChainInfo] = []
    seq_sidecar: dict[str, dict] = {}

    for i, chain in enumerate(structure.chains):
        name = str(chain["name"])
        mol_type = int(chain["mol_type"])
        entity_id = int(chain["entity_id"])
        num_res = int(chain["res_num"])

        seq = sequences.get(name)
        seq_hash = hash_sequence(seq) if seq else None

        # cluster_id / msa_id, consistent with scripts/process/cluster.py
        if mol_type in POLYMER_TYPES and seq_hash is not None:
            key = seq_hash
        elif mol_type == NONPOLYMER:
            key = _ccd_code_for_chain(structure, chain)
        else:
            key = None

        if key is None:
            cluster_id: object = -1
        elif clusters is not None:
            cluster_id = clusters.get(key.lower(), clusters.get(key, key))
        else:
            cluster_id = key

        # MSA is generated for protein chains only.
        if mol_type == PROTEIN and seq_hash is not None and msa_mode == "hash":
            msa_id: object = seq_hash
        else:
            msa_id = -1

        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=name,
                mol_type=mol_type,
                cluster_id=cluster_id,
                msa_id=msa_id,
                num_residues=num_res,
                entity_id=entity_id,
                template_ids=None,
            )
        )

        if seq:
            seq_sidecar[name] = {
                "mol_type": mol_type,
                "entity_id": entity_id,
                "sequence": seq,
                "hash": seq_hash,
            }

    interface_info = [
        InterfaceInfo(chain_1=int(itf["chain_1"]), chain_2=int(itf["chain_2"]))
        for itf in structure.interfaces
    ]

    record = Record(
        id=pdb_id,
        structure=parsed.info,
        chains=chain_info,
        interfaces=interface_info,
    )
    return record, seq_sidecar


def process_structure(  # noqa: PLR0913
    data: PDB,
    outdir: Path,
    moldir: str,
    clusters: Optional[dict],
    use_assembly: bool,
    msa_mode: str,
    max_total_residues: int,
    clash_max_chains: int,
) -> None:
    """Parse, filter and write a single structure.

    Two-stage to bound the cost of pathological giant assemblies:
      1. Parse with ``compute_interfaces=False`` (skips the O(atoms) KDTree).
      2. Drop the whole entry if it exceeds ``max_total_residues`` (paper's
         ">5000 residues" rule) *before* the expensive interface/clash steps.
      3. Only run ``ClashingChainsFilter`` (O(chains^2)) when the chain count is
         at or below ``clash_max_chains``.
    """
    struct_path = outdir / "structures" / f"{data.id}.npz"
    record_path = outdir / "records" / f"{data.id}.json"
    seq_path = outdir / "sequences" / f"{data.id}.json"

    if struct_path.exists() and record_path.exists():
        return

    try:
        parsed = parse_mmcif(
            data.path,
            mols=None,
            moldir=moldir,
            use_assembly=use_assembly,
            compute_interfaces=False,
        )
        structure = parsed.data

        # Size gate (post-assembly): drop oversized entries per the paper.
        if len(structure.residues) > max_total_residues:
            print(f"Skipping {data.id}: {len(structure.residues)} residues > {max_total_residues}")  # noqa: T201, E501
            return

        # Now safe to compute interfaces (bounded by the size gate).
        interfaces = find_interfaces(structure.atoms, structure.chains)
        structure = replace(structure, interfaces=interfaces)
        parsed = replace(parsed, data=structure, info=replace(parsed.info, num_interfaces=len(interfaces)))

        # Static filters. ClashingChains is O(chains^2); skip on huge chain counts.
        filters: list[StaticFilter] = [
            ExcludedLigands(),
            MinimumLengthFilter(min_len=4, max_len=5000),
            UnknownFilter(),
            ConsecutiveCA(max_dist=10.0),
        ]
        if len(structure.chains) <= clash_max_chains:
            filters.append(ClashingChainsFilter(freq=0.3, dist=1.7))

        mask = structure.mask.copy()
        for f in filters:
            mask = mask & f.filter(structure)

        record, seq_sidecar = build_record(data.id, parsed, clusters, msa_mode)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"Failed to parse {data.id}")  # noqa: T201
        return

    # Mark validity on chains / interfaces from the filter mask
    chains = [replace(c, valid=bool(mask[c.chain_id])) for c in record.chains]
    interfaces = [
        replace(itf, valid=bool(mask[itf.chain_1] and mask[itf.chain_2]))
        for itf in record.interfaces
    ]
    structure = replace(structure, mask=mask)
    record = replace(record, chains=chains, interfaces=interfaces)

    np.savez_compressed(struct_path, **asdict(structure))
    with record_path.open("w") as f:
        json.dump(record.to_dict(), f)
    with seq_path.open("w") as f:
        json.dump(seq_sidecar, f)


def finalize(outdir: Path) -> None:
    """Group all records into a single manifest.json."""
    records_dir = outdir / "records"
    records = []
    failed = 0
    for record in records_dir.iterdir():
        try:
            with record.open("r") as f:
                records.append(json.load(f))
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"Failed to read {record}")  # noqa: T201
    print(f"Manifest: {len(records)} records, {failed} unreadable.")  # noqa: T201
    with (outdir / "manifest.json").open("w") as f:
        json.dump(records, f)


def process(args: argparse.Namespace) -> None:
    """Run the batch processing task."""
    outdir: Path = args.outdir
    for sub in ("structures", "records", "sequences"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        finalize(outdir)
        return

    clusters = None
    if args.clusters is not None:
        with Path(args.clusters).open("r") as f:
            raw = json.load(f)
        clusters = {str(k).lower(): str(v).lower() for k, v in raw.items()}
        print(f"Loaded {len(clusters)} cluster entries.")  # noqa: T201

    # Preserve RDKit properties across the pickle boundary in workers.
    rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)

    data = fetch(args.datadir, args.max_file_size)

    # Optional job-array sharding: deterministically slice the file list so
    # independent Slurm array tasks each own a disjoint subset.
    if args.nshards > 1:
        data = sorted(data, key=lambda p: p.id)
        data = [p for i, p in enumerate(data) if i % args.nshards == args.shard]
        print(f"Shard {args.shard}/{args.nshards}: {len(data)} files.")  # noqa: T201

    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(data)))

    fn = partial(
        process_structure,
        outdir=outdir,
        moldir=str(args.moldir),
        clusters=clusters,
        use_assembly=args.use_assembly,
        msa_mode=args.msa_mode,
        max_total_residues=args.max_total_residues,
        clash_max_chains=args.clash_max_chains,
    )

    if num_processes > 1:
        p_umap(fn, data, num_cpus=num_processes)
    else:
        for item in tqdm(data):
            fn(item)

    # Only build the manifest in single-shard runs; sharded runs finalize once
    # afterwards via --finalize-only to avoid concurrent manifest rewrites.
    if args.nshards == 1:
        finalize(outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RCSB mmCIF -> StructureV2.")
    parser.add_argument("--datadir", type=Path, required=True, help="Dir of mmCIF files.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output dir.")
    parser.add_argument("--moldir", type=Path, required=True, help="CCD mols dir ({CCD}.pkl).")
    parser.add_argument("--clusters", type=Path, default=None, help="clustering.json (optional).")
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--use-assembly", action="store_true", help="Expand assembly 1.")
    parser.add_argument("--max-file-size", type=int, default=None, help="Skip files larger than this (bytes; for uncompressed .cif).")
    parser.add_argument("--max-total-residues", type=int, default=5000, help="Drop entries with more residues than this (post-assembly; paper rule).")
    parser.add_argument("--clash-max-chains", type=int, default=300, help="Skip ClashingChainsFilter above this chain count (avoids O(n^2) blowup).")
    parser.add_argument("--nshards", type=int, default=1, help="Total number of job-array shards.")
    parser.add_argument("--shard", type=int, default=0, help="This task's shard index in [0, nshards).")
    parser.add_argument("--finalize-only", action="store_true", help="Only (re)build manifest.json from records/ and exit.")
    parser.add_argument(
        "--msa-mode",
        choices=["none", "hash"],
        default="none",
        help="'none': msa_id=-1 (loadable without MSAs). 'hash': msa_id=sha256(seq) (requires MSAs present).",
    )
    args = parser.parse_args()
    process(args)

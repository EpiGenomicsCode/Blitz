"""Mine structural templates for protein chains.

Pipeline:
  1. For each query protein chain, build an HMM profile from its MSA (a3m from
     ColabFold search, converted via hmmbuild).
  2. hmmsearch that profile against a database of all known PDB protein chain
     sequences.
  3. Apply an E-value threshold and a 60-day filter: only hits whose (earliest)
     deposition date is >= 60 days before the query's own deposition date are
     kept.
  4. For surviving hits, extract the aligned residue window's backbone frame
     and CB/CA coordinates, then write a ``Template`` npz.
  5. Emit updated ``ChainInfo.template_ids`` and ``Record.templates`` in a side
     manifest patch.

ColabFold A3M files place the query first, so HMM match columns map directly to
query residue positions. The template database is deduplicated by exact
sequence and uses the earliest deposition date for leakage filtering.
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from boltz.data.tokenize.boltz2 import compute_frame
from boltz.data.types import Record, Template, TemplateInfo

TEMPLATE_COORDINATES_DTYPE = [
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i1")),
    ("frame_rot", np.dtype("9f4")),
    ("frame_t", np.dtype("3f4")),
    ("coords_cb", np.dtype("3f4")),
    ("coords_ca", np.dtype("3f4")),
    ("mask_frame", np.dtype("?")),
    ("mask_cb", np.dtype("?")),
    ("mask_ca", np.dtype("?")),
]

# AlphaFold 3 template-search settings. Quality filtering happens after search,
# so hmmsearch itself uses permissive thresholds.
HMMSEARCH_FLAGS = [
    "--noali",
    "--F1", "0.1",
    "--F2", "0.1",
    "--F3", "0.1",
    "-E", "100",
    "--incE", "100",
    "--domE", "100",
    "--incdomE", "100",
]
DATE_GATE_DAYS = 60
MAX_TEMPLATES_PER_CHAIN = 20  # "At most 20 templates can be returned by our search"
MAX_PROFILE_SEQUENCES = 300  # "cropped to the first 300 sequences" (UniRef90 MSA)
MAX_SUBSEQUENCE_RATIO = 0.95  # exclude near-identical/duplicate-sequence hits
MIN_HIT_LENGTH = 10  # exclude templates shorter than this many residues
MIN_QUERY_COVERAGE = 0.10  # exclude templates covering less than 10% of the query


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def atomic_dump(template: Template, path: Path) -> None:
    """Write a Template npz atomically (write to a unique temp file + os.replace).

    Content-addressed template names mean multiple parallel shard processes
    can independently decide to produce the same (target, range) template at
    the same time. ``Template.dump()`` (Boltz's ``NumpySerializable.dump``)
    calls ``np.savez_compressed`` directly on the target path, which is NOT
    atomic -- a concurrent writer could observe a partially-written file.
    Writing to a per-process/per-call-unique temp path first and using
    ``os.replace`` (atomic on POSIX/Lustre) avoids that entirely, without
    requiring any change to Boltz's own (de)serialization code.

    NOTE: the temp path MUST already end in ``.npz`` -- ``np.savez_compressed``
    (used internally by ``Template.dump``) silently appends a ``.npz`` suffix
    to any path that doesn't already have one, which would otherwise break
    the intended temp filename and the subsequent ``os.replace``.
    """
    tmp_path = path.parent / f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.npz"
    template.dump(tmp_path)
    os.replace(tmp_path, path)


def find_a3m(msa_root: Path, seq_hash: str, shard_index: dict[str, Path]) -> Optional[Path]:
    """Locate the raw a3m for a sequence hash using a prebuilt shard index."""
    return shard_index.get(seq_hash)


def build_shard_index(msa_root: Path) -> dict[str, Path]:
    """Map sequence hash -> a3m path across all msa_a3m/shard_*/ directories."""
    index = {}
    for shard_dir in sorted(msa_root.glob("shard_*")):
        for a3m in shard_dir.glob("*.a3m"):
            index[a3m.stem] = a3m
    return index


def prepare_profile_input(a3m_path: Path, out_path: Path) -> None:
    """Crop a raw ColabFold a3m to mirror AF3's HMM-profile input.

    AF3 builds the template-search profile from the *deduplicated UniRef90
    MSA*, cropped to the first 300 sequences (SI Sec 2.4). Our raw ColabFold
    a3m mixes UniRef30 hits (headers ``>UniRef100_...``) with environmental/
    metagenomic hits (all other headers) in one file. We approximate AF3's
    input by always keeping the query (first record, whose header is not a
    UniRef header) and then the first ``MAX_PROFILE_SEQUENCES`` UniRef-only
    hits, in the file's existing (e-value-sorted) order, dropping
    environmental sequences entirely from the profile.
    """
    lines = a3m_path.read_text().splitlines()
    records = []  # list of (header, seq)
    i = 0
    while i < len(lines):
        header = lines[i]
        seq = lines[i + 1] if i + 1 < len(lines) else ""
        records.append((header, seq))
        i += 2

    if not records:
        out_path.write_text("")
        return

    query_header, query_seq = records[0]
    kept = [(query_header, query_seq)]
    for header, seq in records[1:]:
        if header.startswith(">UniRef100_"):
            kept.append((header, seq))
        if len(kept) - 1 >= MAX_PROFILE_SEQUENCES:
            break

    with out_path.open("w") as f:
        for header, seq in kept:
            f.write(f"{header}\n{seq}\n")


def run_hmmbuild(a3m_path: Path, hmm_out: Path, hmmer_bin: Path) -> bool:
    """Build an HMM profile from an a3m alignment (a2m-compatible format)."""
    cmd = [
        str(hmmer_bin / "hmmbuild"),
        "--amino",
        "--informat", "a2m",
        str(hmm_out),
        str(a3m_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def run_hmmsearch(
    hmm_path: Path, db_fasta: Path, domtbl_out: Path, hmmer_bin: Path
) -> bool:
    cmd = [
        str(hmmer_bin / "hmmsearch"),
        "--domtblout", str(domtbl_out),
        *HMMSEARCH_FLAGS,
        "--cpu", "1",
        "-o", "/dev/null",
        str(hmm_path),
        str(db_fasta),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def parse_domtblout(path: Path) -> list[dict]:
    """Parse hmmer --domtblout into a list of hit dicts.

    Columns (0-indexed) per hmmer domtblout spec:
      0 target name, 3 query name, 6 full-seq E-value,
      12 this-domain E-value (i-Evalue), 15/16 hmm-from/hmm-to (query match cols),
      17/18 ali-from/ali-to (target sequence coords, 1-indexed).
    """
    hits = []
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            hits.append(
                {
                    "target_name": parts[0],
                    "query_name": parts[3],
                    "i_evalue": float(parts[12]),
                    "hmm_from": int(parts[15]),
                    "hmm_to": int(parts[16]),
                    "ali_from": int(parts[17]),
                    "ali_to": int(parts[18]),
                }
            )
    return hits


def load_template_db_meta(db_fasta: Path) -> dict[str, tuple[str, str, str, str]]:
    """Parse '>{hash} {pdb_id}|{chain_name}|{date}' headers -> metadata + sequence.

    NOTE: hmmer/FASTA treat everything up to the first whitespace as the
    sequence ID (this is what shows up as ``target_name`` in hmmsearch's
    domtblout); anything after the first space is a free-text description.
    The DB fasta MUST follow that convention (hash as ID, rest as
    description) for ``parse_domtblout``'s ``target_name`` to match these
    dict keys.
    """
    meta = {}
    with db_fasta.open() as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line[1:].strip()
            elif header is not None:
                seq_hash, rest = header.split(" ", 1)
                pdb_id, chain_name, date = rest.split("|")
                meta[seq_hash] = (pdb_id, chain_name, date, line)
                header = None
    return meta


def extract_template_coordinates(
    target_struct_path: Path, chain_name: str, ali_from: int, ali_to: int
) -> Optional[np.ndarray]:
    """Extract TemplateCoordinates for residues [ali_from-1, ali_to) (0-indexed).

    ``ali_from``/``ali_to`` are 1-indexed, inclusive residue positions within
    the target chain's *own* reference sequence, as reported by hmmsearch.
    """
    data = np.load(target_struct_path)
    chains = data["chains"]
    residues = data["residues"]
    atoms = data["atoms"]

    chain_rows = chains[chains["name"] == chain_name]
    if len(chain_rows) == 0:
        return None
    chain = chain_rows[0]

    res_start = int(chain["res_idx"])
    res_num = int(chain["res_num"])

    lo = ali_from - 1
    hi = ali_to
    if lo < 0 or hi > res_num:
        # Clip defensively; a mismatch here indicates a reference/structure
        # sequence length discrepancy (e.g. engineered constructs).
        lo = max(lo, 0)
        hi = min(hi, res_num)
    if hi <= lo:
        return None

    rows = []
    for p in range(lo, hi):
        res = residues[res_start + p]
        if int(res["res_idx"]) != p:
            # Defensive: residues should be stored in res_idx order per chain.
            continue
        atom_st = int(res["atom_idx"])
        atom_num = int(res["atom_num"])
        res_atoms = atoms[atom_st:atom_st + atom_num]

        frame_mask = False
        frame_rot = np.eye(3, dtype=np.float32).flatten()
        frame_t = np.zeros(3, dtype=np.float32)
        if atom_num >= 3:  # noqa: PLR2004
            atom_n, atom_ca, atom_c = res_atoms[0], res_atoms[1], res_atoms[2]
            frame_mask = bool(
                atom_n["is_present"] and atom_ca["is_present"] and atom_c["is_present"]
            )
            if frame_mask:
                rot, t = compute_frame(
                    atom_n["coords"], atom_ca["coords"], atom_c["coords"]
                )
                frame_rot = rot.astype(np.float32).flatten()
                frame_t = t.astype(np.float32)

        center_atom = atoms[int(res["atom_center"])]
        disto_atom = atoms[int(res["atom_disto"])]

        rows.append(
            (
                p,
                int(res["res_type"]),
                frame_rot,
                frame_t,
                disto_atom["coords"].astype(np.float32),
                center_atom["coords"].astype(np.float32),
                frame_mask,
                bool(disto_atom["is_present"]),
                bool(center_atom["is_present"]),
            )
        )

    if not rows:
        return None
    return np.array(rows, dtype=TEMPLATE_COORDINATES_DTYPE)


def search_hits_for_sequence(
    seq_hash: str,
    msa_root: Path,
    db_fasta: Path,
    hmmer_bin: Path,
    shard_index: dict[str, Path],
) -> Optional[list[dict]]:
    """Run hmmbuild+hmmsearch ONCE per unique sequence hash.

    Many chain instances across different PDB entries share the exact same
    sequence (dataset-wide redundancy factor ~3.76x: 513,613 chain instances
    vs. 136,524 unique sequences). The raw hmmsearch hit list depends only on
    the query's sequence/MSA, not on which structure instance is asking, so
    it is cached/computed once per hash and reused across all instances
    (mirroring the same deduplication already used for MSA generation).
    """
    a3m_path = find_a3m(msa_root, seq_hash, shard_index)
    if a3m_path is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        profile_input = tmp / "profile_input.a3m"
        prepare_profile_input(a3m_path, profile_input)

        hmm_path = tmp / "profile.hmm"
        if not run_hmmbuild(profile_input, hmm_path, hmmer_bin):
            return None

        domtbl_path = tmp / "hits.domtbl"
        if not run_hmmsearch(hmm_path, db_fasta, domtbl_path, hmmer_bin):
            return None

        return parse_domtblout(domtbl_path)


def mine_one_chain(
    pdb_id: str,
    chain_name: str,
    hits: list[dict],
    query_seq: str,
    query_deposited: str,
    db_meta: dict[str, tuple[str, str, str, str]],
    structures_dir: Path,
    outdir: Path,
) -> Optional[list[TemplateInfo]]:
    """Filter cached raw hits + extract templates for a single query chain instance."""
    query_date = parse_date(query_deposited)
    cutoff = query_date - timedelta(days=DATE_GATE_DAYS)
    query_len = len(query_seq)

    # Filters, matching AlphaFold3 SI Sec 2.4 exactly:
    #   1. date gate (60 days before query deposition)
    #   2. never the query's own PDB entry
    #   3. min_hit_length: template's aligned region must be >= 10 residues
    #   4. min_query_coverage: aligned region must cover >= 10% of the query
    #   5. max_subsequence_ratio: exclude near-duplicate hits whose aligned
    #      template text is both >95% of the query's length AND a verbatim
    #      substring of the query (i.e. would trivially "leak" the answer)
    valid_hits = []
    for hit in hits:
        target_hash = hit["target_name"]
        if target_hash not in db_meta:
            continue
        target_pdb_id, target_chain_name, target_date_str, target_seq = db_meta[target_hash]

        target_date = parse_date(target_date_str)
        if target_date > cutoff:
            continue  # too recent relative to query; would leak information
        if target_pdb_id == pdb_id:
            continue  # defensive: never use the query's own structure

        hit_len = hit["ali_to"] - hit["ali_from"] + 1
        if hit_len < MIN_HIT_LENGTH:
            continue

        query_coverage = (hit["hmm_to"] - hit["hmm_from"] + 1) / query_len
        if query_coverage < MIN_QUERY_COVERAGE:
            continue

        matching_seq = target_seq[hit["ali_from"] - 1 : hit["ali_to"]]
        length_ratio = len(matching_seq) / query_len
        if length_ratio > MAX_SUBSEQUENCE_RATIO and matching_seq in query_seq:
            continue  # near-identical duplicate of the query; would leak the answer

        valid_hits.append((hit, target_pdb_id, target_chain_name))

    valid_hits.sort(key=lambda x: x[0]["i_evalue"])
    valid_hits = valid_hits[:MAX_TEMPLATES_PER_CHAIN]

    template_infos = []
    for hit, target_pdb_id, target_chain_name in valid_hits:
        target_struct_path = structures_dir / f"{target_pdb_id}.npz"
        if not target_struct_path.exists():
            continue

        # The target and alignment range determine the file contents, allowing
        # identical templates to be shared across query records.
        subdir = target_pdb_id[:2]
        template_name = f"{subdir}/{target_pdb_id}_{target_chain_name}_{hit['ali_from']}_{hit['ali_to']}"
        template_path = outdir / f"{template_name}.npz"
        if not template_path.exists():
            coords = extract_template_coordinates(
                target_struct_path, target_chain_name, hit["ali_from"], hit["ali_to"]
            )
            if coords is None:
                continue
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template = Template(coordinates=coords)
            atomic_dump(template, template_path)

        template_infos.append(
            TemplateInfo(
                name=template_name,
                query_chain=chain_name,
                query_st=hit["hmm_from"] - 1,
                query_en=hit["hmm_to"],
                template_chain=target_chain_name,
                template_st=hit["ali_from"] - 1,
                template_en=hit["ali_to"],
            )
        )

    return template_infos if template_infos else None


def _load_json_with_retry(path: Path, retries: int = 3, delay_s: float = 1.0) -> Optional[dict]:
    """Read a JSON file, retrying transient filesystem errors."""
    last_err = None
    for attempt in range(retries):
        try:
            with path.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay_s)
    print(f"WARNING: failed to read {path} after {retries} attempts ({last_err}); skipping.")  # noqa: T201
    return None


def build_query_list(manifest_path: Path, seq_dir: Path) -> list[tuple[str, str, str, str, str]]:
    """Build the full (pdb_id, chain_name, seq_hash, sequence, deposited) query list."""
    with manifest_path.open() as f:
        manifest = json.load(f)
    records = manifest["records"] if isinstance(manifest, dict) else manifest

    queries = []
    for r in records:
        pdb_id = r["id"]
        deposited = r.get("structure", {}).get("deposited")
        if not deposited:
            continue
        seq_path = seq_dir / f"{pdb_id}.json"
        if not seq_path.exists():
            continue
        seqs = _load_json_with_retry(seq_path)
        if seqs is None:
            continue
        for chain_name, info in seqs.items():
            if info.get("mol_type") != 0:  # PROTEIN
                continue
            h = info.get("hash")
            seq = info.get("sequence")
            if h is None or not seq:
                continue
            queries.append((pdb_id, chain_name, h, seq, deposited))
    return queries


def main(args: argparse.Namespace) -> None:
    hmmer_bin = Path(args.hmmer_bin)
    msa_root = Path(args.msa_root)
    db_fasta = Path(args.template_db)
    structures_dir = Path(args.structures_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Building a3m shard index ...")  # noqa: T201
    shard_index = build_shard_index(msa_root)
    print(f"  {len(shard_index)} a3m files indexed.")  # noqa: T201

    print("Loading template DB metadata ...")  # noqa: T201
    db_meta = load_template_db_meta(db_fasta)
    print(f"  {len(db_meta)} template DB entries.")  # noqa: T201

    queries = build_query_list(Path(args.manifest), Path(args.sequences_dir))
    print(f"Total query chains in dataset: {len(queries)}")  # noqa: T201

    if args.nshards:
        # Shard by UNIQUE HASH GROUPS, not by flat instance list: many chain
        # instances share a sequence, and splitting them across shards would
        # force the same (expensive) hmmbuild+hmmsearch to be redundantly
        # repeated in multiple shards, defeating the per-hash cache below.
        by_hash: dict[str, list[tuple]] = {}
        for q in queries:
            by_hash.setdefault(q[2], []).append(q)
        all_hashes = sorted(by_hash.keys())
        shard_hashes = all_hashes[args.shard::args.nshards]
        queries = [q for h in shard_hashes for q in by_hash[h]]
        print(  # noqa: T201
            f"Shard {args.shard}/{args.nshards}: {len(shard_hashes)} unique "
            f"sequences, {len(queries)} query chain instances."
        )

    if args.limit:
        queries = queries[: args.limit]

    print(f"Mining templates for {len(queries)} query chains ...")  # noqa: T201
    unique_hashes = {q[2] for q in queries}
    print(f"  ({len(unique_hashes)} unique sequences; results cached per-hash)")  # noqa: T201

    hits_cache: dict[str, Optional[list[dict]]] = {}
    results = {}
    n_with_templates = 0
    n_total_templates = 0
    for pdb_id, chain_name, seq_hash, seq, deposited in queries:
        if seq_hash not in hits_cache:
            hits_cache[seq_hash] = search_hits_for_sequence(
                seq_hash, msa_root, db_fasta, hmmer_bin, shard_index
            )
        hits = hits_cache[seq_hash]
        if hits is None:
            continue

        infos = mine_one_chain(
            pdb_id, chain_name, hits, seq, deposited, db_meta, structures_dir, outdir,
        )
        if infos:
            results[f"{pdb_id}:{chain_name}"] = [
                {
                    "name": t.name,
                    "query_chain": t.query_chain,
                    "query_st": t.query_st,
                    "query_en": t.query_en,
                    "template_chain": t.template_chain,
                    "template_st": t.template_st,
                    "template_en": t.template_en,
                }
                for t in infos
            ]
            n_with_templates += 1
            n_total_templates += len(infos)

    print(f"Chains with >=1 template: {n_with_templates}/{len(queries)}")  # noqa: T201
    print(f"Total template hits mined: {n_total_templates}")  # noqa: T201

    shard_suffix = f"_shard{args.shard}" if args.nshards else ""
    results_path = outdir / f"mining_results{shard_suffix}.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {results_path}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--sequences-dir", type=str, required=True)
    parser.add_argument("--structures-dir", type=str, required=True)
    parser.add_argument("--msa-root", type=str, required=True)
    parser.add_argument("--template-db", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--hmmer-bin", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Pilot: limit number of query chains.")
    parser.add_argument("--nshards", type=int, default=None, help="Total number of shards (job array).")
    parser.add_argument("--shard", type=int, default=None, help="This shard's index (0-based).")
    args = parser.parse_args()
    main(args)

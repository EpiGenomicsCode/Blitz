#!/usr/bin/env python3
"""Build a durable AFDB accession → pack/miss shard index (sqlite).

Reads zero-padded pack and miss manifests from the packed/ directory and
writes:

  * pack_index.sqlite  — (accession PRIMARY KEY, kind, shard_id, tar_path)
  * INDEX_SUMMARY.json — counts vs kept_all.tsv

Prefer sqlite over jsonl.gz for random accession lookup.

Example::

    python scripts/process/build_afdb_pack_index.py \\
        --packed-dir /data/afdb/packed \\
        --kept-tsv /data/afdb/meta/gmv_filter/kept_all.tsv \\
        --out-dir /data/afdb/meta
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PACK_RE = re.compile(r"^afdb_pack_(\d{4})\.manifest\.json$")
MISS_RE = re.compile(r"^afdb_miss_(\d{4})\.manifest\.json$")
BATCH = 50_000


def _load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _iter_manifests(packed_dir: Path) -> list[tuple[str, Path, int]]:
    """Return sorted (kind, path, shard_id) for zero-padded manifests only."""
    out: list[tuple[str, Path, int]] = []
    for p in packed_dir.iterdir():
        if not p.is_file():
            continue
        m = PACK_RE.match(p.name)
        if m:
            out.append(("pack", p, int(m.group(1))))
            continue
        m = MISS_RE.match(p.name)
        if m:
            out.append(("miss", p, int(m.group(1))))
    out.sort(key=lambda t: (0 if t[0] == "pack" else 1, t[2]))
    return out


def _count_kept(kept_tsv: Path) -> int:
    n = 0
    with kept_tsv.open() as f:
        header = f.readline()
        if not header:
            return 0
        for _ in f:
            n += 1
    return n


def build_index(
    packed_dir: Path,
    kept_tsv: Path,
    out_dir: Path,
    db_name: str = "pack_index.sqlite",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / db_name
    summary_path = out_dir / "INDEX_SUMMARY.json"

    if db_path.exists():
        db_path.unlink()

    manifests = _iter_manifests(packed_dir)
    n_pack_manifests = sum(1 for k, _, _ in manifests if k == "pack")
    n_miss_manifests = sum(1 for k, _, _ in manifests if k == "miss")
    if n_pack_manifests != 1000 or n_miss_manifests != 1000:
        print(
            f"WARNING: expected 1000 pack + 1000 miss zero-padded manifests; "
            f"got pack={n_pack_manifests} miss={n_miss_manifests}",
            file=sys.stderr,
        )

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE pack_index (
            accession TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            shard_id INTEGER NOT NULL,
            tar_path TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )

    n_pack = 0
    n_miss = 0
    n_dup = 0
    batch: list[tuple[str, str, int, str]] = []
    insert_sql = (
        "INSERT OR IGNORE INTO pack_index (accession, kind, shard_id, tar_path) "
        "VALUES (?, ?, ?, ?)"
    )

    def flush() -> None:
        nonlocal batch, n_dup
        if not batch:
            return
        before = conn.total_changes
        conn.executemany(insert_sql, batch)
        inserted = conn.total_changes - before
        n_dup += len(batch) - inserted
        batch = []

    for kind, path, shard_from_name in manifests:
        man = _load_manifest(path)
        shard_id = int(man.get("shard_id", shard_from_name))
        tar = man.get("tar") or str(
            path.parent / path.name.replace(".manifest.json", ".tar.gz")
        )
        accessions = man.get("accessions") or []
        n_packed = int(man.get("n_packed", len(accessions)))
        if n_packed != len(accessions):
            print(
                f"WARNING: {path.name}: n_packed={n_packed} != len(accessions)={len(accessions)}",
                file=sys.stderr,
            )
        for acc in accessions:
            batch.append((acc, kind, shard_id, tar))
            if kind == "pack":
                n_pack += 1
            else:
                n_miss += 1
            if len(batch) >= BATCH:
                flush()
        if (shard_from_name + 1) % 200 == 0:
            print(f"  … {kind} shard {shard_from_name:04d}", flush=True)

    flush()
    conn.commit()

    n_indexed = conn.execute("SELECT COUNT(*) FROM pack_index").fetchone()[0]
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    # Drop WAL sidecar into a clean single-file DB for portability.
    # Re-open and checkpoint.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()

    n_kept = _count_kept(kept_tsv)
    n_missing = n_kept - n_indexed

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "packed_dir": str(packed_dir),
        "kept_tsv": str(kept_tsv),
        "db_path": str(db_path),
        "n_pack_manifests": n_pack_manifests,
        "n_miss_manifests": n_miss_manifests,
        "n_pack_accessions": n_pack,
        "n_miss_accessions": n_miss,
        "n_manifest_rows": n_pack + n_miss,
        "n_dup_skipped": n_dup,
        "n_kept": n_kept,
        "n_indexed": n_indexed,
        "n_missing": n_missing,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {db_path}")
    print(f"Wrote {summary_path}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--packed-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--kept-tsv",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    args = p.parse_args()
    if not args.packed_dir.is_dir():
        print(f"ERROR: packed dir not found: {args.packed_dir}", file=sys.stderr)
        return 1
    if not args.kept_tsv.is_file():
        print(f"ERROR: kept tsv not found: {args.kept_tsv}", file=sys.stderr)
        return 1
    build_index(args.packed_dir, args.kept_tsv, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

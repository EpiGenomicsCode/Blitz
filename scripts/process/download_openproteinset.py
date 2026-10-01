#!/usr/bin/env python3
"""Download OpenProteinSet uniclust30_unfiltered shards from s3://openfold (unsigned).

Supports:
  --list          print shard keys + sizes
  --shard KEY     download one shard (resume-safe)
  --index N       download the N-th shard from a sorted listing (for Slurm arrays)
  --limit N       only consider first N shards (pilot)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from botocore import UNSIGNED
from botocore.config import Config
import boto3


BUCKET = "openfold"
PREFIX = "uniclust30_unfiltered/"


def client():
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, region_name="us-east-1"),
    )


def list_shards(s3) -> list[dict]:
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".zip"):
                out.append({"key": o["Key"], "size": o["Size"]})
    out.sort(key=lambda x: x["key"])
    return out


def download_one(s3, key: str, outdir: Path) -> Path:
    name = Path(key).name
    dest = outdir / name
    part = outdir / f".{name}.partial"
    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]

    if dest.exists() and dest.stat().st_size == size:
        print(f"SKIP exists {dest} ({size} bytes)")
        return dest

    # Resume via Range if partial exists
    start = 0
    mode = "wb"
    if part.exists():
        start = part.stat().st_size
        if start < size:
            mode = "ab"
            print(f"RESUME {key} from byte {start}/{size}")
        elif start == size:
            part.rename(dest)
            print(f"FINALIZE partial {dest}")
            return dest
        else:
            part.unlink()
            start = 0

    extra = {}
    if start > 0:
        extra["Range"] = f"bytes={start}-{size - 1}"

    print(f"GET {key} -> {dest} ({size} bytes, start={start})")
    resp = s3.get_object(Bucket=BUCKET, Key=key, **extra)
    body = resp["Body"]
    written = start
    with part.open(mode) as f:
        while True:
            chunk = body.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if written % (256 * 1024 * 1024) < 8 * 1024 * 1024:
                print(f"  ... {written}/{size} ({100 * written / size:.1f}%)", flush=True)

    if written != size:
        raise RuntimeError(f"size mismatch for {key}: got {written} expected {size}")
    part.rename(dest)
    print(f"DONE {dest}")
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None, help="Cache shard listing JSON")
    p.add_argument("--list", action="store_true")
    p.add_argument("--shard", type=str, default=None)
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    s3 = client()

    manifest = args.manifest or (args.outdir / "shards_manifest.json")
    if manifest.exists():
        shards = json.loads(manifest.read_text())
        print(f"Loaded manifest {manifest}: {len(shards)} shards")
    else:
        print("Listing s3://openfold/uniclust30_unfiltered/ ...")
        shards = list_shards(s3)
        manifest.write_text(json.dumps(shards, indent=2))
        print(f"Wrote {manifest}: {len(shards)} shards, {sum(s['size'] for s in shards)/1e12:.2f} TB")

    if args.limit is not None:
        shards = shards[: args.limit]

    if args.list:
        for i, s in enumerate(shards):
            print(f"{i:04d}  {s['size']/1e9:7.3f}G  {s['key']}")
        return

    if args.shard is not None:
        key = args.shard if args.shard.startswith(PREFIX) else PREFIX + args.shard
        download_one(s3, key, args.outdir)
        return

    if args.index is not None:
        if args.index < 0 or args.index >= len(shards):
            print(f"index {args.index} out of range 0..{len(shards)-1}")
            sys.exit(1)
        download_one(s3, shards[args.index]["key"], args.outdir)
        return

    print("Nothing to do; pass --list / --shard / --index")
    sys.exit(2)


if __name__ == "__main__":
    main()

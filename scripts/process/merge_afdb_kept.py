#!/usr/bin/env python3
"""Merge per-shard kept_shard_*.tsv from confidence filter into kept_all.tsv + summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gmv-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()
    outdir = args.outdir or args.gmv_dir
    outdir.mkdir(parents=True, exist_ok=True)

    shards = sorted(args.gmv_dir.glob("kept_shard_*.tsv"))
    if not shards:
        raise SystemExit(f"no kept_shard_*.tsv in {args.gmv_dir}")

    out = outdir / "kept_all.tsv"
    n = 0
    gmv_sum = 0.0
    fails_jsonl = 0
    lows = 0
    # optional: count jsonl outcomes
    conf_dir = args.gmv_dir / "conf_shards"
    if conf_dir.exists():
        for jp in conf_dir.glob("shard_*.jsonl"):
            with jp.open() as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if not d.get("ok"):
                        fails_jsonl += 1
                    elif float(d.get("globalMetricValue", 0)) < 50:
                        lows += 1

    with out.open("w") as w:
        header = None
        for sp in shards:
            with sp.open() as f:
                h = f.readline()
                if header is None:
                    header = h
                    w.write(h)
                for line in f:
                    if not line.strip():
                        continue
                    w.write(line)
                    n += 1
                    try:
                        gmv_sum += float(line.split("\t", 2)[1])
                    except Exception:  # noqa: BLE001
                        pass

    summary = {
        "n_shards": len(shards),
        "n_kept": n,
        "mean_gmv": (gmv_sum / n) if n else None,
        "jsonl_fail": fails_jsonl,
        "jsonl_low": lows,
        "kept_all": str(out),
    }
    (outdir / "kept_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

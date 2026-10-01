#!/usr/bin/env python3
"""Filter AFDB candidates by mean pLDDT from confidence_v6.json (no prediction API).

For each accession in candidates.tsv:
  1. GET https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-confidence_v6.json
  2. globalMetricValue := mean(confidenceScore)  (0–100 scale)
  3. Keep if >= --min-gmv (default 50 = paper global lDDT >= 0.5)

Resume-safe JSONL + kept TSV per shard. CIF download happens later only for keepers.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def confidence_url(accession: str, version: int = 6) -> str:
    return (
        f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-confidence_v{version}.json"
    )


def cif_url(accession: str, version: int = 6) -> str:
    return f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v{version}.cif"


def fetch_gmv(accession: str, version: int = 6, timeout: float = 60.0) -> dict:
    url = confidence_url(accession, version)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        scores = data["confidenceScore"]
        if not scores:
            return {"accession": accession, "ok": False, "error": "empty confidenceScore"}
        gmv = float(sum(scores) / len(scores))
        return {
            "accession": accession,
            "ok": True,
            "globalMetricValue": gmv,
            "n_res": len(scores),
            "cifUrl": cif_url(accession, version),
            "confidenceUrl": url,
        }
    except urllib.error.HTTPError as e:
        return {"accession": accession, "ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"accession": accession, "ok": False, "error": str(e)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--min-gmv", type=float, default=50.0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--version", type=int, default=6)
    args = p.parse_args()

    shard_dir = args.outdir / "conf_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = shard_dir / f"shard_{args.shard_id:04d}.jsonl"
    kept_path = args.outdir / f"kept_shard_{args.shard_id:04d}.tsv"

    rows = []
    with args.candidates.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for i, line in enumerate(f):
            if i < args.offset:
                continue
            if args.limit is not None and len(rows) >= args.limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            rows.append(dict(zip(header, parts)))

    done: set[str] = set()
    if out_jsonl.exists():
        with out_jsonl.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["accession"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [r for r in rows if r["accession"] not in done]
    print(
        f"shard={args.shard_id} total={len(rows)} done={len(done)} todo={len(todo)}",
        flush=True,
    )

    n_kept = n_low = n_fail = 0
    write_header = not kept_path.exists() or kept_path.stat().st_size == 0
    with out_jsonl.open("a") as jf, kept_path.open("a") as kf:
        if write_header:
            kf.write(
                "accession\tglobalMetricValue\tcifUrl\tops_cluster_id\tseq_len\t"
                "zip_name\ta3m_member\tn_res\n"
            )

        def work(row: dict):
            acc = row["accession"]
            last = None
            for attempt in range(args.retries):
                last = fetch_gmv(acc, version=args.version)
                if last.get("ok") or (
                    isinstance(last.get("error"), str)
                    and last["error"].startswith("HTTP 404")
                ):
                    return row, last
                time.sleep(0.25 * (attempt + 1))
            return row, last

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, r) for r in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                row, meta = fut.result()
                jf.write(json.dumps(meta) + "\n")
                if not meta.get("ok"):
                    n_fail += 1
                elif meta["globalMetricValue"] < args.min_gmv:
                    n_low += 1
                else:
                    n_kept += 1
                    kf.write(
                        f"{row['accession']}\t{meta['globalMetricValue']:.4f}\t"
                        f"{meta['cifUrl']}\t{row['ops_cluster_id']}\t{row['seq_len']}\t"
                        f"{row['zip_name']}\t{row['a3m_member']}\t{meta['n_res']}\n"
                    )
                if i % 500 == 0:
                    print(
                        f"  {i}/{len(todo)} kept={n_kept} low={n_low} fail={n_fail}",
                        flush=True,
                    )

    print(f"CONF_FILTER_DONE kept={n_kept} low={n_low} fail={n_fail} -> {kept_path}")


if __name__ == "__main__":
    main()

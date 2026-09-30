"""Apply cluster and template metadata to RCSB record files.

The training loader reads individual files under ``records/``. This script
updates those files and rebuilds ``manifest.json`` from the same records.
"""

import argparse
import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

from boltz.data.types import Record, TemplateInfo

# Populated in the parent before forking so workers inherit via copy-on-write,
# avoiding re-loading/re-pickling the (large) clustering + template-patch dicts
# once per worker.
_CLUSTERING: dict[str, str] = {}
_TEMPLATE_PATCH: dict[str, list[dict]] = {}


def load_template_patch(results_dir: Path) -> dict[str, list[dict]]:
    """Merge all mining_results_shard*.json (or non-sharded pilot output)."""
    patch: dict[str, list[dict]] = {}
    shard_files = sorted(results_dir.glob("mining_results_shard*.json"))
    if not shard_files:
        shard_files = sorted(results_dir.glob("mining_results.json"))
    print(f"Merging {len(shard_files)} template result shard files ...")  # noqa: T201
    for sf in shard_files:
        with sf.open() as f:
            data = json.load(f)
        for key, hits in data.items():
            patch.setdefault(key, []).extend(hits)
    print(f"Total query chains with template hits: {len(patch)}")  # noqa: T201
    return patch


def patch_one_record(
    rec_path: Path,
    clustering: dict[str, str],
    template_patch: dict[str, list[dict]],
) -> tuple[bool, bool]:
    """Patch a single record file in place. Returns (cluster_changed, templates_added)."""
    record = Record.load(rec_path)
    pdb_id = record.id

    cluster_changed = False
    new_chains = []
    for chain in record.chains:
        new_cluster_id = clustering.get(str(chain.cluster_id), chain.cluster_id)
        if new_cluster_id != chain.cluster_id:
            cluster_changed = True

        key = f"{pdb_id}:{chain.chain_name}"
        hits = template_patch.get(key)
        if hits:
            template_ids = [h["name"] for h in hits]
            new_chains.append(
                replace(chain, cluster_id=new_cluster_id, template_ids=template_ids)
            )
        else:
            new_chains.append(replace(chain, cluster_id=new_cluster_id))

    templates_added = False
    new_templates = list(record.templates) if record.templates else []
    for chain in record.chains:
        key = f"{pdb_id}:{chain.chain_name}"
        hits = template_patch.get(key)
        if hits:
            templates_added = True
            new_templates.extend(
                TemplateInfo(
                    name=h["name"],
                    query_chain=h["query_chain"],
                    query_st=h["query_st"],
                    query_en=h["query_en"],
                    template_chain=h["template_chain"],
                    template_st=h["template_st"],
                    template_en=h["template_en"],
                )
                for h in hits
            )

    if cluster_changed or templates_added:
        new_record = replace(record, chains=new_chains, templates=new_templates or None)
        with rec_path.open("w") as f:
            json.dump(new_record.to_dict(), f)

    return cluster_changed, templates_added


def worker(rec_path: Path) -> tuple[bool, bool]:
    try:
        return patch_one_record(rec_path, _CLUSTERING, _TEMPLATE_PATCH)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED {rec_path}: {e}")  # noqa: T201
        return False, False


def main(args: argparse.Namespace) -> None:
    global _CLUSTERING, _TEMPLATE_PATCH  # noqa: PLW0603
    records_dir = Path(args.records_dir)

    print("Loading clustering.json ...")  # noqa: T201
    with Path(args.clustering).open() as f:
        _CLUSTERING = json.load(f)
    print(f"  {len(_CLUSTERING)} entries.")  # noqa: T201

    _TEMPLATE_PATCH = load_template_patch(Path(args.template_results_dir))

    all_records = sorted(records_dir.glob("*.json"))
    if args.nshards:
        all_records = all_records[args.shard :: args.nshards]
    print(f"Patching {len(all_records)} record files with {args.num_processes} workers ...")  # noqa: T201

    n_cluster = 0
    n_templates = 0
    n_done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.num_processes) as pool:
        for c, t in pool.imap_unordered(worker, all_records, chunksize=200):
            n_cluster += int(c)
            n_templates += int(t)
            n_done += 1
            if n_done % 20000 == 0:
                print(f"  ... {n_done}/{len(all_records)} done", flush=True)  # noqa: T201

    print(f"Records with cluster_id changes: {n_cluster}")  # noqa: T201
    print(f"Records with template_ids added: {n_templates}")  # noqa: T201
    print("PATCH_SHARD_DONE")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=str, required=True)
    parser.add_argument("--clustering", type=str, required=True)
    parser.add_argument("--template-results-dir", type=str, required=True)
    parser.add_argument("--nshards", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()
    main(args)

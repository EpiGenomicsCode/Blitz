"""Merge sharded template-mining results into an updated manifest.

Reads the original ``manifest.json`` plus all ``mining_results_shard*.json``
fragments produced by ``mine_templates.py``, and writes a NEW
``manifest_with_templates.json`` with ``ChainInfo.template_ids`` and
``Record.templates`` populated. The original manifest is left untouched, so
this can be inspected/re-run without risk to the base dataset.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from boltz.data.types import Manifest, Record, TemplateInfo


def main(args: argparse.Namespace) -> None:
    manifest = Manifest.load(Path(args.manifest))
    records_by_id = {r.id: r for r in manifest.records}

    patch: dict[str, list[dict]] = {}
    shard_files = sorted(Path(args.results_dir).glob("mining_results_shard*.json"))
    if not shard_files:
        # Non-sharded pilot output
        shard_files = sorted(Path(args.results_dir).glob("mining_results.json"))
    print(f"Merging {len(shard_files)} result shard files ...")  # noqa: T201
    for sf in shard_files:
        with sf.open() as f:
            data = json.load(f)
        for key, hits in data.items():
            patch.setdefault(key, []).extend(hits)

    print(f"Total query chains with template hits: {len(patch)}")  # noqa: T201

    n_updated_records = 0
    n_updated_chains = 0
    n_total_templates = 0
    for key, hits in patch.items():
        pdb_id, chain_name = key.split(":", 1)
        record = records_by_id.get(pdb_id)
        if record is None:
            continue

        template_infos = [
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
        ]
        template_ids = [t.name for t in template_infos]

        new_chains = []
        for chain in record.chains:
            if chain.chain_name == chain_name:
                new_chains.append(replace(chain, template_ids=template_ids))
                n_updated_chains += 1
            else:
                new_chains.append(chain)

        existing_templates = list(record.templates) if record.templates else []
        new_record = replace(
            record,
            chains=new_chains,
            templates=existing_templates + template_infos,
        )
        records_by_id[pdb_id] = new_record
        n_updated_records += 1
        n_total_templates += len(template_infos)

    updated_records = list(records_by_id.values())
    new_manifest = Manifest(records=updated_records)
    out_path = Path(args.outdir) / "manifest_with_templates.json"
    new_manifest.dump(out_path)

    print(f"Records updated: {n_updated_records}")  # noqa: T201
    print(f"Chains updated: {n_updated_chains}")  # noqa: T201
    print(f"Total TemplateInfo entries added: {n_total_templates}")  # noqa: T201
    print(f"Wrote {out_path}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    main(args)

# Training data

Training samples are drawn from RCSB and AlphaFold DB with probabilities 0.55
and 0.45. RCSB examples can use templates; AFDB examples do not. The repository
includes the processing code and loader configuration, but not the datasets.

## RCSB

The RCSB set was built from PDB entries released by 2023-06-01. Structures were
converted to Boltz `StructureV2` records, and protein sequences were clustered
at 40% identity. MSAs, taxonomy metadata, and templates were then added to the
records. The final manifest contains 198,487 training records and 398
post-cutoff validation targets.

Training keeps structures with 1–300 chains and resolution no worse than 9 A.
Templates use the AlphaFold 3 60-day date gate. Up to four templates are loaded,
with a 60% chance of dropping templates during training.

The structure conversion used the deposited structures without an additional
biological-assembly expansion step. The effect of that choice has not been
measured separately.

## AlphaFold DB

AFDB examples are the intersection of OpenProteinSet UniClust MSAs, UniRef30
accessions, and available AlphaFold DB models. Models with
`globalMetricValue < 50` are excluded.

| stage | examples |
|---|---:|
| OpenProteinSet rows scanned | 15,161,405 |
| also present in UniRef30 | 10,140,666 |
| unique AFDB candidates | 8,769,792 |
| confidence or download failures | 78,634 |
| below the confidence cutoff | 2,230,975 |
| retained for training | 6,460,183 |

The corpus is stored in tar shards. Each accession has a record, a structure,
and an MSA. A SQLite index locates the files, and the loader extracts them to a
node-local cache as needed.

AFDB records use the accession as the cluster and MSA identifier. The loader
sets the experimental method to AFDB, overrides B-factors, and disables
templates. Two retained AFDB sequences exactly match validation sequences. This
overlap was identified after the training corpus had been frozen and should be
taken into account when interpreting validation results.

The RCSB build scripts are under `scripts/process/` (`rcsb_v2.py`, `cluster.py`,
`msa_v2.py`, `mine_templates.py`, `finalize_templates.py`, and
`apply_manifest_patches.py`). The AFDB path uses `scan_ops_shards.py`,
`intersect_ops_uniref.py`, `filter_afdb_confidence.py`, `process_afdb.py`,
`build_afdb_pack_index.py`, and `build_afdb_samples.py`.

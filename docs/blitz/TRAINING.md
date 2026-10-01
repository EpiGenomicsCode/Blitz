# Training Blitz

Blitz fine-tunes the Boltz-2 structure module while keeping the trunk,
confidence model, and affinity model frozen. Training starts from the released
Boltz-2 weights, which also provide the teacher targets.

The released students were selected at these checkpoints:

| model | optimizer steps | student sampling steps | teacher grid at the checkpoint |
|---|---:|---:|---:|
| K8 | 3,000 | 8 | 350 |
| K16 | 2,200 | 16 | 300 |

The teacher grid grows log-linearly from 200 to the endpoint shown in the table.
The supplied configurations stop at the selected checkpoint and reproduce that
schedule.

Both models use discrete multistep consistency distillation with a deterministic
student sampler (`gamma=0`). The teacher supplies one Heun target hop with no
step-scale correction. The loss is pseudo-Huber with square-root Karras
weighting. Dropout and sigma sampling are synchronized across multiplicity.

Both models use AdamW with a peak learning rate of `3e-5`, a 500-step linear
warmup, and no weight decay. The effective batch size is 128: one example per
GPU across 32 GPUs, with four accumulated microbatches. K8 uses
`sigma_min_relative=0.004`. K16 uses `0.01`, per-sample alignment-weight
normalization, and additional weight on nucleic-acid and ligand atoms.

## Data paths

`scripts/train/run_mscd.sh` refuses to start until these variables are set:

| variable | contents |
|---|---|
| `BOLTZ_MSCD_OUTPUT` | output directory |
| `BOLTZ2_TEACHER_CKPT` | released Boltz-2 structure/confidence checkpoint |
| `BOLTZ_RCSB_TARGET_DIR` | RCSB manifest, records, and structures |
| `BOLTZ_RCSB_MSA_DIR` | RCSB MSA files |
| `BOLTZ_RCSB_TEMPLATE_DIR` | mined RCSB templates |
| `BOLTZ_AFDB_TARGET_DIR` | logical AFDB target root |
| `BOLTZ_AFDB_MSA_DIR` | logical AFDB MSA root |
| `BOLTZ_AFDB_PACK_INDEX` | SQLite index for the AFDB tar shards |
| `BOLTZ_AFDB_SAMPLES_PATH` | AFDB sampling table |
| `BOLTZ_CCD_MOL_DIR` | pickled CCD molecules |

The complete configurations are
`scripts/train/configs/mscd_boltz2_k8_rho40.yaml` and
`scripts/train/configs/mscd_boltz2_k16_rho40_normalign.yaml`. The launcher
accepts OmegaConf overrides for cluster-specific settings such as the number of
nodes and GPUs. Keep the effective batch size at 128 when reproducing the
training schedule.

The inference checkpoints contain the trunk, distilled structure model, and
confidence model. They do not require a separate Boltz-2 checkpoint. The
reported results use the student weights at the selected training step rather
than EMA weights.

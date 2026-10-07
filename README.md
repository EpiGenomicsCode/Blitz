# Blitz

**Accelerating structure prediction through diffusion-model distillation.**

Blitz distills biomolecular structure-prediction models into deterministic
8-step (K8) and 16-step (K16) students. The released **Boltz-2** students retain
accuracy comparable to the teacher on the reported complex-structure benchmark
while using far fewer denoiser evaluations. **K16 is the recommended default.**

This repository includes inference checkpoints, training and evaluation code,
and a standalone [SynthID Bio structure detector](#synthid-bio-structure-detector).
The distillation method was also tested with AlphaFold 3, however only Boltz-2 student weights are distributed here.

[Quickstart](docs/blitz/QUICKSTART.md) ·
[Results](docs/blitz/RESULTS.md) ·
[Training](docs/blitz/TRAINING.md) ·
[Training data](docs/blitz/DATA.md) ·
[Model provenance](docs/blitz/PROVENANCE.md)

## Installation

Use a fresh environment with **Python 3.10–3.12**. Clone the `main` branch,
which includes both Blitz and the SynthID Bio detector:

```bash
git clone --branch main https://github.com/EpiGenomicsCode/Blitz.git
cd Blitz
python -m pip install -e '.[cuda]'
python -m pip install huggingface_hub
```

For CPU-only or non-CUDA hardware, replace `'.[cuda]'` with `'.'`; CPU inference
is substantially slower. The optional `synthid` extra is needed only for the
[detector](#synthid-bio-structure-detector).

## Run a prediction

### 1. Download the checkpoints

The [Hugging Face release](https://huggingface.co/vinaymatt/Blitz-Boltz2)
contains complete K8 and K16 prediction models.

```bash
hf download vinaymatt/Blitz-Boltz2 \
  blitz-boltz2-k8.ckpt blitz-boltz2-k16.ckpt SHA256SUMS \
  --local-dir checkpoints

(cd checkpoints && sha256sum -c SHA256SUMS)
```


### 2. Prepare the input

Create `input.yaml` using the [Boltz input format](docs/prediction.md).
See [`examples/`](examples/) for sample inputs. For protein inputs, supply the
required MSAs or add `--use_msa_server` to generate them through the MSA server.

### 3. Run K16

```bash
boltz predict input.yaml \
  --model boltz2 \
  --checkpoint checkpoints/blitz-boltz2-k16.ckpt \
  --blitz_policy k16 \
  --recycling_steps 5 \
  --diffusion_samples 5 \
  --seed 1 \
  --out_dir predictions-k16
```

For K8, use `blitz-boltz2-k8.ckpt` with `--blitz_policy k8`.

The `--blitz_policy` flag loads the matching sampling schedule and endpoint
geometry correction, and sets `--sampling_steps` to 8 or 16. 
These are the documented benchmark inference settings. 

## Benchmark results

Results on the [Boltz-2 benchmark](https://doi.org/10.1101/2025.06.14.659707)
of approximately 2,300 targets:

| System | Model | NFE | Complex lDDT ↑ | RF-valid ↑ | DockQ ↑ | Ligand RMSD ↓ | Any violation ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| AlphaFold 3 | teacher | 200 | 0.8615 | 41.5% | 0.4198 | 6.63 Å | 9.8% |
| AlphaFold 3 | Blitz K8 | 8 | **0.8635** | 40.5% | **0.4207** | **6.44 Å** | 11.3% |
| AlphaFold 3 | Blitz K16 | 16 | **0.8634** | **42.3%** | 0.4167 | **6.51 Å** | **6.8%** |
| Boltz-2 | teacher | 600 | 0.8499 | 98.5% | 0.3929 | 9.00 Å | 30.3% |
| Boltz-2 | Blitz K8 | 8 | 0.8492 | 93.7% | 0.3897 | **8.75 Å** | 41.9% |
| Boltz-2 | Blitz K16 | 16 | **0.8533** | 94.1% | **0.3932** | **8.78 Å** | **26.7%** |


NFE denotes denoiser evaluations.  

## Training

Blitz fine-tunes the Boltz-2 structure module using **discrete multistep
consistency distillation (MSCD)**. The trunk, confidence model, and affinity
model remain frozen. Training starts from the released Boltz-2 weights, which
also provide the teacher targets.

Set the data-path variables described in the [training guide](docs/blitz/TRAINING.md),
then run the launcher for the desired student:

```bash
# Train K8.
scripts/train/run_mscd.sh k8

# Or train K16.
scripts/train/run_mscd.sh k16
```

The launcher checks required data paths and accepts OmegaConf overrides for
cluster-specific settings. See:

- [Training](docs/blitz/TRAINING.md) for schedules, optimizer settings, and checkpoint selection.
- [Configurations](scripts/train/configs/) for the supplied training configs.
- [Training data](docs/blitz/DATA.md) for preparation of the RCSB and AlphaFold DB corpora.
- [Provenance](docs/blitz/PROVENANCE.md) for model lineage and the limitations of the reconstructed K8 training configuration.

## Evaluation

Evaluation and aggregation scripts are in [`scripts/eval/`](scripts/eval/),
including `run_evals.py`, `aggregate_evals.py`, and `physcialsim_metrics.py`.
See [Results](docs/blitz/RESULTS.md) for the benchmark comparison and its
relationship to the Boltz-2 evaluation protocol.

## SynthID Bio structure detector

[`scripts/synthid_bio/`](scripts/synthid_bio/) contains the standalone scorer
used in our SynthID Bio experiments. It reconstructs the detector described in
supplementary Listing 1 of [Stutz et al. (*Nature*, 2026)](https://doi.org/10.1038/s41586-026-10965-y)
and runs without an AlphaFold 3 source-code dependency. **It is not an official
DeepMind detector.** Some conventions were inferred from the paper and
checkpoint layout.

The scorer requires a full, uncompressed AF3 checkpoint containing the detector
tensors under `diffuser/~/watermark_detector/point_net/` and a single-model
prediction in mmCIF format. It reconstructs AF3's 24-slot atom layout, computes
distance and torsion features, and applies the five-layer point-net detector.

```bash
python -m pip install -e '.[synthid]'

python scripts/synthid_bio/score.py \
  --checkpoint /path/to/af3.bin \
  --structure /path/to/prediction.cif
```

The output includes the raw detector logit and a Boolean indicating `logit > 0`.
**Zero is the diagnostic threshold used in our experiments, not DeepMind's
calibrated detection threshold.**  See the
[detector documentation](scripts/synthid_bio/README.md) for tokenization details
and supported atom layouts.

## Repository layout

| Path | Contents |
|---|---|
| `src/boltz/` | Model, data pipeline, and command-line interface |
| `src/boltz/model/potentials/blitz.py` | K8/K16 deterministic sampling policies |
| `src/boltz/consistency/` | MSCD training code |
| `scripts/train/` | Training launcher, configurations, and ID splits |
| `scripts/process/` | RCSB and AlphaFold DB data processing |
| `scripts/eval/` | Benchmark evaluation and aggregation |
| `scripts/synthid_bio/` | Standalone SynthID Bio structure detector |
| `docs/blitz/` | Quickstart, results, training, data, and provenance |
| `examples/` | Example prediction inputs |

## License and acknowledgments

Blitz builds on [Boltz](https://github.com/jwohlwend/boltz). The repository code
is distributed under the [MIT License](LICENSE), with upstream attribution
preserved. Boltz-2 model weights are released under MIT.

AlphaFold 3 parameters are subject to separate Google DeepMind terms and are
not included. Obtain any AF3 checkpoint required for the detector through the
official access process and use it under the applicable terms.

## Citation
Please also cite the relevant upstream Boltz work below. If you use the SynthID
Bio detector, cite [Stutz et al.](https://doi.org/10.1038/s41586-026-10965-y).

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

If you use automatic MSA generation, also cite ColabFold:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```

# Quickstart

Clone the Blitz branch and install it in a fresh environment:

```bash
git clone --branch blitz git@github.com:EpiGenomicsCode/Blitz.git
cd Blitz
pip install -e '.[cuda]'
```

Download the checkpoints from Hugging Face:

```bash
hf auth login
hf download vinaymatt/Blitz-Boltz2 \
  blitz-boltz2-k8.ckpt blitz-boltz2-k16.ckpt SHA256SUMS \
  --local-dir checkpoints
cd checkpoints && sha256sum -c SHA256SUMS && cd ..
```

## Run a prediction

K16 is the recommended default:

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
geometry correction. Lowering `--sampling_steps` on ordinary Boltz-2 is not
equivalent.

The benchmark settings are included in the command above. Reducing the number
of diffusion samples is useful for a quick run, but changes the comparison in
[RESULTS.md](RESULTS.md).

## Start a training run

Set the paths described in [TRAINING.md](TRAINING.md), then use the supplied
launcher:

```bash
scripts/train/run_mscd.sh k8
scripts/train/run_mscd.sh k16
```

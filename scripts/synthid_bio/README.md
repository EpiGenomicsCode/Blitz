# SynthID Bio structure scorer

This is the scorer used in our SynthID Bio experiments, packaged without an
AlphaFold 3 source dependency. It is our reconstruction of the detector in
[supplementary Listing 1][supplement] of [Stutz et al.][paper], not an
officially released DeepMind detector. Some conventions had to be inferred
from the paper and checkpoint layout.

The scorer reads the detector tensors under
`diffuser/~/watermark_detector/point_net/` in a full, uncompressed AF3 model
checkpoint. These are separate from the diffusion-head parameters. It then
reconstructs AF3's 24-slot atom layout from a single-model prediction mmCIF,
computes the distance and torsion features, and runs the five-layer detector.

Install the optional JAX dependency from the repository root:

```bash
pip install -e '.[synthid]'
```

Run:

```bash
python scripts/synthid_bio/score.py \
  --checkpoint /path/to/af3.bin \
  --structure /path/to/prediction.cif
```

The command prints the raw logit and a boolean for `logit > 0`. Zero is the
diagnostic threshold used in our experiments; it is not DeepMind's calibrated
detection threshold.

The standalone mmCIF reader follows the tokenization used in our evaluation:
standard protein, RNA, and DNA residues occupy fixed CCD-style atom slots;
nonstandard polymer residues and non-polymers use one heavy-atom token per
atom. Ambiguous alternate locations and unsupported atom layouts are rejected
rather than silently reordered.

[paper]: https://doi.org/10.1038/s41586-026-10965-y
[supplement]: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10965-y/MediaObjects/41586_2026_10965_MOESM1_ESM.pdf

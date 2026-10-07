#!/usr/bin/env python3
"""Score one local structure with detector tensors from a local checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import numpy as np
from detector import coordinates_from_mmcif, detector_logits, load_detector_weights


def main() -> None:
    """Parse arguments and print the detector result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    args = parser.parse_args()

    weights = load_detector_weights(args.checkpoint)
    positions, atom_mask = coordinates_from_mmcif(args.structure)
    forward = jax.jit(lambda p, m: detector_logits(weights, p, m))
    logit = float(np.asarray(forward(positions, atom_mask))[0])
    if not math.isfinite(logit):
        raise RuntimeError("Detector returned a non-finite logit")
    print(json.dumps({"logit": logit, "diagnostic_positive": logit > 0.0}))


if __name__ == "__main__":
    main()

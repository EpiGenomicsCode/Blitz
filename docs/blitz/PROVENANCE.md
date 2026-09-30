# Model lineage

The `blitz` branch is based on official Boltz commit
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. The training work originally began
from the old upstream `v2` branch at
`8e590cf8e11b188c51f927d3c1ea964267cc150e`. The distillation changes were
ported onto the current Boltz codebase instead of merging the two histories.

## Checkpoints

The K8 and K16 inference checkpoints are available from
[`vinaymatt/Blitz-Boltz2`](https://huggingface.co/vinaymatt/Blitz-Boltz2).
Each is a complete prediction model containing the Boltz-2 trunk, distilled
structure model, and confidence model. File hashes are provided alongside the
weights in `SHA256SUMS`.

The K16 training configuration was recovered from its resolved run
configuration. The K8 configuration was reconstructed from the surviving run
record because its complete source snapshot was not retained. It reproduces the
selected schedule and checkpoint settings, but is not claimed to be a
byte-for-byte copy of the original training tree.

The sampling schedules and endpoint corrections for both models are defined in
`src/boltz/model/potentials/blitz.py`.

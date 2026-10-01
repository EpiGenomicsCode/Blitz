from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def compute_trunk_features(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    recycles: Optional[int] = 3,
    enable_grad: bool = False,
) -> Dict[str, Any]:
    """
    Compute trunk + diffusion conditioning for distillation via Boltz2.forward().

    ``return_trunk_features=True`` also returns ``diffusion_conditioning``, which
    Boltz-2 factors out of ``AtomDiffusion`` (see boltz2.py forward).

    Args:
        model: Boltz2 model to extract trunk from
        batch: Input batch
        recycles: Number of recycling steps (None = 0)
        enable_grad: If True, keep gradient graph (student trunk). If False, detach.

    Returns:
        Dict with s_inputs, s, z, relative_position_encoding, diffusion_conditioning, batch.
    """
    if recycles is None:
        recycles = 0

    _REQUIRED = (
        "s_inputs",
        "s",
        "z",
        "relative_position_encoding",
        "diffusion_conditioning",
        "batch",
    )
    _DC_REQUIRED = (
        "q",
        "c",
        "to_keys",
        "atom_enc_bias",
        "atom_dec_bias",
        "token_trans_bias",
    )

    def _check_keys(trunk_features: Dict[str, Any]) -> None:
        missing = [k for k in _REQUIRED if k not in trunk_features]
        if missing:
            raise KeyError(
                f"return_trunk_features missing {missing}; "
                "Boltz2 early-exit contract broken (boltz2.py)"
            )
        dc = trunk_features["diffusion_conditioning"]
        if not isinstance(dc, dict):
            raise TypeError(
                f"diffusion_conditioning must be dict, got {type(dc)}"
            )
        missing_dc = [k for k in _DC_REQUIRED if k not in dc]
        if missing_dc:
            raise KeyError(
                f"diffusion_conditioning missing {missing_dc}; "
                "see boltz2.py return_trunk_features"
            )

    if enable_grad:
        trunk_features = model.forward(
            feats=batch,
            recycling_steps=recycles,
            return_trunk_features=True,
        )
        _check_keys(trunk_features)
        return {
            "s_inputs": trunk_features["s_inputs"],
            "s": trunk_features["s"],
            "z": trunk_features["z"],
            "relative_position_encoding": trunk_features["relative_position_encoding"],
            "diffusion_conditioning": trunk_features["diffusion_conditioning"],
            "batch": trunk_features["batch"],
        }

    # Use no_grad (not inference_mode): diffusion_conditioning tensors are fed
    # into the trainable structure_module forward. Inference tensors cannot be
    # saved for backward even when only module parameters need grads
    # (PyTorch: "Inference tensors cannot be saved for backward"). no_grad +
    # detach keeps the trunk out of the graph; skip clone to avoid
    # duplicating O(N_tok²) pair tensors.
    with torch.no_grad():
        trunk_features = model.forward(
            feats=batch,
            recycling_steps=recycles,
            return_trunk_features=True,
        )

    _check_keys(trunk_features)
    dc = trunk_features["diffusion_conditioning"]
    dc_out: dict[str, Any] = {}
    for k, v in dc.items():
        if torch.is_tensor(v):
            dc_out[k] = v.detach()
        else:
            # e.g. to_keys callable (batch-independent indexing_matrix)
            dc_out[k] = v

    return {
        "s_inputs": trunk_features["s_inputs"].detach(),
        "s": trunk_features["s"].detach(),
        "z": trunk_features["z"].detach(),
        "relative_position_encoding": trunk_features[
            "relative_position_encoding"
        ].detach(),
        "diffusion_conditioning": dc_out,
        "batch": trunk_features["batch"],
    }

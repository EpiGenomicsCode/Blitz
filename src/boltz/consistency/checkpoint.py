from __future__ import annotations

import hashlib
import inspect
from typing import Callable

import torch
try:
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # optional dependency during inference
    DictConfig = object  # type: ignore
    OmegaConf = None  # type: ignore


_warned_once = {"ema_missing": False}


def sanitize_boltz2_hparams(hparams: dict) -> dict:
    """Filter checkpoint hparams to kwargs accepted by this tree's ``Boltz2``.

    HF / ablative ``boltz2_conf.ckpt`` may carry nested keys that newer or
    older trees reject (e.g. ``diffusion_process_args.mse_rotational_alignment``
    is absent from boltz_v2 ``AtomDiffusion``). Top-level keys are filtered to
    ``Boltz2.__init__``; ``diffusion_process_args`` is filtered to
    ``AtomDiffusion.__init__`` (minus ctor-injected ``score_model_args`` /
    ``compile_score``).
    """
    from boltz.model.models.boltz2 import Boltz2
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    if isinstance(hparams, DictConfig) and OmegaConf is not None:
        plain = OmegaConf.to_container(hparams, resolve=True)  # type: ignore
    else:
        plain = dict(hparams)

    sig = inspect.signature(Boltz2.__init__)
    allowed = {k for k in sig.parameters.keys() if k != "self"}
    filtered = {k: v for k, v in plain.items() if k in allowed}
    filtered.pop("validators", None)
    if "validate_structure" in allowed:
        filtered.setdefault("validate_structure", False)

    dropped_top = sorted(set(plain.keys()) - set(filtered.keys()) - {"validators"})
    if dropped_top:
        print(
            f"[load_teacher] Dropping unsupported hparams: {dropped_top[:10]}"
            + (" ..." if len(dropped_top) > 10 else "")
        )

    dpa = filtered.get("diffusion_process_args")
    if isinstance(dpa, DictConfig) and OmegaConf is not None:
        dpa = OmegaConf.to_container(dpa, resolve=True)  # type: ignore
    if isinstance(dpa, dict):
        atom_sig = inspect.signature(AtomDiffusion.__init__)
        # Boltz2 injects score_model_args / compile_score; they must not come
        # from the checkpoint diffusion_process_args blob.
        atom_allowed = {
            k
            for k in atom_sig.parameters.keys()
            if k not in ("self", "score_model_args", "compile_score")
        }
        dpa_filt = {k: v for k, v in dpa.items() if k in atom_allowed}
        dropped_dpa = sorted(set(dpa.keys()) - set(dpa_filt.keys()))
        if dropped_dpa:
            print(
                "[load_teacher] Dropping unsupported diffusion_process_args: "
                f"{dropped_dpa}"
            )
        filtered["diffusion_process_args"] = dpa_filt

    # Boltz-2 HF / ablative boltz2_conf.ckpt omits pairformer_args.v2 even though
    # the saved weights are AttentionPairBiasV2 (no attention.norm_s; layer
    # pre_norm_s only). Default v2=False builds V1 AttentionPairBias with
    # randomly-init norm_s → 128 missing keys and PairformerLayer.forward
    # TypeError on k_in=. Match inference PairformerArgsV2 / structurev2.yaml.
    pa = filtered.get("pairformer_args")
    if isinstance(pa, DictConfig) and OmegaConf is not None:
        pa = OmegaConf.to_container(pa, resolve=True)  # type: ignore
        filtered["pairformer_args"] = pa
    if isinstance(pa, dict) and pa.get("v2") is not True:
        pa = dict(pa)
        pa["v2"] = True
        filtered["pairformer_args"] = pa
        print(
            "[load_teacher] Injecting pairformer_args.v2=True "
            "(ckpt hparams omit it; weights are AttentionPairBiasV2)"
        )

    return filtered


def _select_state_dict_for_loading(ckpt: Dict[str, torch.Tensor], use_ema: bool) -> Dict[str, torch.Tensor]:
    """Select EMA or plain state dict from checkpoint."""
    if use_ema:
        if "state_dict_ema" in ckpt and isinstance(ckpt["state_dict_ema"], dict):
            return ckpt["state_dict_ema"]
        if "ema" in ckpt and isinstance(ckpt["ema"], dict):
            ema_block = ckpt["ema"]
            if "shadow_params" in ema_block and isinstance(ckpt.get("state_dict"), dict):
                if not _warned_once["ema_missing"]:
                    print("[load_teacher] EMA requested but only shadow_params found; using plain state_dict.")
                    _warned_once["ema_missing"] = True
            else:
                if not _warned_once["ema_missing"]:
                    print("[load_teacher] EMA requested but not found; using plain state_dict.")
                    _warned_once["ema_missing"] = True
        else:
            if not _warned_once["ema_missing"]:
                print("[load_teacher] EMA requested but not found; using plain state_dict.")
                _warned_once["ema_missing"] = True

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def _short_sha256_of_state(state_dict: Dict[str, torch.Tensor]) -> str:
    """Create a short SHA256 digest of state dict for provenance tracking."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        try:
            h.update(k.encode())
            h.update(t.detach().cpu().numpy().tobytes())
        except Exception:
            # skip non-tensor entries
            continue
    return h.hexdigest()[:12]


def load_teacher(
    checkpoint_path: str,
    use_ema: bool = True,
    map_location: str | torch.device = "cpu",
    *,
    model_ctor: Callable[..., torch.nn.Module] | None = None,
    model_kwargs: dict | None = None,
) -> torch.nn.Module:
    """
    Load a Boltz-2 teacher model checkpoint.

    Reconstructs the model from checkpoint hyperparameters, loads weights, and
    freezes everything. Defaults to ``Boltz2`` (not Boltz1).

    Args:
        checkpoint_path: Path to checkpoint file (``boltz2_conf.ckpt`` via
            ``download_boltz2`` / ``$BOLTZ_CACHE`` / ``BOLTZ2_TEACHER_CKPT``)
        use_ema: Whether to prefer EMA weights
        map_location: Device to load checkpoint to
        model_ctor: Optional model constructor (defaults to Boltz2)
        model_kwargs: Optional kwargs for model constructor

    Returns:
        Loaded and frozen model with teacher_provenance attached
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if model_ctor is None:
        # Reconstruct from checkpoint hyperparameters (Boltz-2 default)
        hparams = ckpt.get("hyper_parameters", None)
        if hparams is None:
            raise RuntimeError(
                "Checkpoint missing hyper_parameters. Provide model_ctor/model_kwargs explicitly."
            )

        from boltz.model.models.boltz2 import Boltz2

        model_ctor = Boltz2
        model_kwargs = sanitize_boltz2_hparams(hparams)

    # Supply default steering arguments for checkpoints that omit them.
    original_hparams = ckpt.get("hyper_parameters", {})
    has_steering_in_ckpt = "steering_args" in original_hparams

    if model_ctor.__name__ in ("Boltz1", "Boltz2") and (
        model_kwargs is None or "steering_args" not in model_kwargs
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if "steering_args" not in model_kwargs:
            model_kwargs["steering_args"] = {
                "fk_steering": False,
                "num_particles": 3,
                "fk_lambda": 4.0,
                "fk_resampling_interval": 3,
                "guidance_update": False,
                "physical_guidance_update": False,
                "contact_guidance_update": False,
                "num_gd_steps": 16,
            }
            if has_steering_in_ckpt:
                print("[load_teacher] WARNING: steering_args in checkpoint but filtered. Using defaults.")
            else:
                print("[load_teacher] Added default steering_args (not in checkpoint)")

    # Construct model
    model = model_ctor(**(model_kwargs or {}))

    # Select and load state dict
    state_dict = _select_state_dict_for_loading(ckpt, use_ema=use_ema)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print load summary
    print(f"[load_teacher] Loaded checkpoint:")
    print(f"  Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Loaded keys: {len(state_dict) - len(missing_keys)}")
    print(f"  Missing keys: {len(missing_keys)}")
    if missing_keys and len(missing_keys) <= 5:
        for key in missing_keys:
            print(f"    - {key}")
    elif len(missing_keys) > 5:
        print(f"    First 5: {missing_keys[:5]}")

    # Freeze everything and set to eval
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Attach digest for provenance
    digest = _short_sha256_of_state(state_dict)
    model.teacher_provenance = {
        "path": str(checkpoint_path),
        "digest": digest,
        "used_ema": bool(use_ema),
    }

    return model


def init_student_from_teacher(
    teacher: torch.nn.Module,
    model_ctor: Callable[..., torch.nn.Module],
    model_kwargs: dict,
    *,
    strict: bool = False,
) -> torch.nn.Module:
    """
    Create a fresh student and load teacher weights.

    Simple wrapper around model construction + load_state_dict.
    Reuses PyTorch's built-in strict=False mechanism.

    Args:
        teacher: Teacher model to copy weights from
        model_ctor: Constructor for student model
        model_kwargs: Kwargs for student constructor
        strict: Whether to require exact match

    Returns:
        Student model with teacher weights loaded
    """
    student = model_ctor(**model_kwargs)
    missing, unexpected = student.load_state_dict(teacher.state_dict(), strict=False)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"State dict mismatch. Missing: {missing[:10]}... Unexpected: {unexpected[:10]}..."
        )

    return student

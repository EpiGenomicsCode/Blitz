from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from pathlib import Path

from boltz.consistency.checkpoint import load_teacher, sanitize_boltz2_hparams
from boltz.consistency.ema import StructureModuleEMA
from boltz.consistency.features import compute_trunk_features
from boltz.consistency.loss_mscd import MSCDLoss
from boltz.consistency.phema import PowerFunctionEMA, snapshot_filename
from boltz.consistency.structure_validation_metrics import (
    StandaloneValidationMetrics,
    accumulate_structure_validation_batch,
    log_structure_validation_epoch,
)
from boltz.data import const
from boltz.model.modules.utils import center_random_augmentation
from boltz.model.models.boltz2 import Boltz2


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, SimpleNamespace):
        return dict(vars(cfg))
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return dict(cfg)


def _get_section(cfg: Any, name: str, default: Any | None = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def normalize_atom_weights_per_sample(
    align_w: torch.Tensor,
    resolved_mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve relative atom weights while matching the unit-weight mass.

    Vector pseudo-Huber consumes ``sqrt(sum_i w_i ||e_i||^2)`` without a
    ``sum(w)`` denominator.  Raw 1/6/11 protein/NA/ligand weights therefore
    change a crop's total loss and gradient scale as its modality composition
    changes.  This opt-in normalization keeps those relative weights (and the
    weighted Kabsch solution, which is invariant to a uniform per-sample
    rescaling) while enforcing ``sum(w * resolved) == sum(resolved)``.

    Empty or fully unresolved samples are left unchanged to avoid division by
    zero. The behavior is controlled by
    ``loss.normalize_align_weights_per_sample``.
    """

    mask = resolved_mask.to(device=align_w.device, dtype=align_w.dtype)
    base_mass = mask.sum(dim=-1, keepdim=True)
    weighted_mass = (align_w * mask).sum(dim=-1, keepdim=True)
    scale = torch.where(
        weighted_mass > 0,
        base_mass / weighted_mass.clamp(min=torch.finfo(align_w.dtype).tiny),
        torch.ones_like(weighted_mass),
    )
    return align_w * scale


def _default_structure_validation_args() -> dict[str, Any]:
    return {
        "enabled": True,
        "recycling_steps": 3,
        "sampling_steps": 8,
        "diffusion_samples": 5,
        "symmetry_correction": True,
        "run_confidence_sequentially": False,
    }


def _merge_structure_validation_args(cfg: Any) -> SimpleNamespace:
    merged = {**_default_structure_validation_args()}
    raw = _get_section(cfg, "structure_validation_args")
    if raw is not None:
        merged.update(_to_plain_dict(raw))
    return SimpleNamespace(**merged)


def _looks_fairscale_checkpointed(m: torch.nn.Module) -> bool:
    f = getattr(m, "forward", None)
    return isinstance(f, functools.partial) and getattr(f.func, "__name__", "") == "_checkpointed_forward"


class BoltzMSCDistiller(pl.LightningModule):
    """Multistep consistency distillation for Boltz-2 structure generation."""

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = True

        checkpointing = _get_section(cfg, "checkpointing")
        teacher_loader = _get_section(checkpointing, "teacher_loader")
        teacher_path = _get_section(teacher_loader, "path")
        teacher_use_ema = bool(_get_section(teacher_loader, "use_ema", False))
        self.teacher = load_teacher(teacher_path, use_ema=teacher_use_ema, map_location="cpu")
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        student_cfg = _to_plain_dict(getattr(self.teacher, "hparams", {}))
        student_cfg["confidence_prediction"] = False
        student_cfg["structure_prediction_training"] = True
        # Distiller owns validation; avoid requiring yaml validators on the student.
        student_cfg["validate_structure"] = False
        student_cfg.pop("validators", None)
        for bucket in (
            "score_model_args",
            "msa_args",
            "pairformer_args",
            "template_args",
            "confidence_model_args",
        ):
            sub = student_cfg.get(bucket)
            if sub is None:
                continue
            if isinstance(sub, DictConfig):
                sub = OmegaConf.to_container(sub, resolve=True)
            elif isinstance(sub, Mapping):
                sub = dict(sub)
            else:
                continue
            if sub.get("activation_checkpointing", False):
                sub["activation_checkpointing"] = False
            student_cfg[bucket] = sub
        # Filter checkpoint arguments that are unsupported by this Boltz tree.
        filtered = sanitize_boltz2_hparams(student_cfg)
        self.student = Boltz2(**filtered)
        wrapped = [n for n, m in self.student.named_modules() if _looks_fairscale_checkpointed(m)]
        if wrapped:
            raise RuntimeError(
                "[mscd-distill] Student still has fairscale-checkpointed modules after disabling "
                f"activation checkpointing; examples: {wrapped[:3]}"
            )
        self.student.load_state_dict(self.teacher.state_dict(), strict=False)
        self._freeze_student_for_structure_module_only()
        self._apply_train_mode()
        self.freeze_report = self._parameter_counts()

        schedule = _get_section(cfg, "schedule", {})
        loss_cfg = _get_section(cfg, "loss", {})
        self.train_multiplicity = int(
            _get_section(_get_section(cfg, "multiplicity", {}), "train_multiplicity", 1)
        )
        self.synchronize_sigmas_across_multiplicity = bool(
            _get_section(_get_section(cfg, "multiplicity", {}), "synchronize_sigmas_across_multiplicity", True)
        )
        self.mscd_loss = MSCDLoss(
            student_steps=int(_get_section(schedule, "S", _get_section(schedule, "student_steps", 8))),
            teacher_steps_start=int(_get_section(schedule, "T_start", 200)),
            teacher_steps_end=int(_get_section(schedule, "T_end", 350)),
            teacher_steps_anneal=int(_get_section(schedule, "T_anneal_steps", 3000)),
            teacher_steps_anneal_shape=str(_get_section(schedule, "T_anneal_shape", "log_linear")),
            teacher_steps_log_rate_per_step=(
                None
                if _get_section(schedule, "T_log_rate_per_step", None) is None
                else float(_get_section(schedule, "T_log_rate_per_step", None))
            ),
            sigma_min=float(_get_section(schedule, "sigma_min_relative", _get_section(schedule, "sigma_min", 0.004))),
            sigma_max=float(_get_section(schedule, "sigma_max_relative", _get_section(schedule, "sigma_max", 160.0))),
            sigma_data=float(_get_section(schedule, "sigma_data", 16.0)),
            rho=float(_get_section(schedule, "rho", 40.0)),
            sampling_mode=str(_get_section(loss_cfg, "sampling_mode", "edm")),
            terminal_anchor=bool(_get_section(loss_cfg, "terminal_anchor", True)),
            terminal_teacher_hop=bool(_get_section(loss_cfg, "terminal_teacher_hop", True)),
            segment_top_mass=float(_get_section(loss_cfg, "segment_top_mass", 0.0)),
            P_mean=float(_get_section(loss_cfg, "P_mean", -1.2)),
            P_std=float(_get_section(loss_cfg, "P_std", 1.5)),
            loss_type=str(_get_section(loss_cfg, "loss_type", "pseudo_huber")),
            weight_mode=str(_get_section(loss_cfg, "weight_mode", "sqrt_karras")),
            huber_eps=float(_get_section(loss_cfg, "huber_eps", 1e-4)),
            huber_eps_mode=str(_get_section(loss_cfg, "huber_eps_mode", "fixed")),
            huber_c_base=float(_get_section(loss_cfg, "huber_c_base", 0.00054)),
            pseudo_huber_reduce=str(_get_section(loss_cfg, "pseudo_huber_reduce", "vector")),
            smooth_lddt_loss_weight=float(_get_section(loss_cfg, "smooth_lddt_loss_weight", 0.0)),
            teacher_step_scale=float(_get_section(loss_cfg, "teacher_step_scale", 1.0)),
            sync_dropout=bool(_get_section(loss_cfg, "sync_dropout", True)),
            debug_invariants=bool(_get_section(loss_cfg, "debug_invariants", False)),
        )
        self.smooth_lddt_loss_weight = float(_get_section(loss_cfg, "smooth_lddt_loss_weight", 0.0))
        self.upweight_nucleic_ligand_atoms = bool(
            _get_section(loss_cfg, "upweight_nucleic_ligand_atoms", True)
        )
        self.normalize_align_weights_per_sample = bool(
            _get_section(loss_cfg, "normalize_align_weights_per_sample", False)
        )

        opt_cfg = _get_section(cfg, "optimization", {})
        self.max_lr = float(_get_section(opt_cfg, "max_lr", 3e-5))
        self.weight_decay = float(_get_section(opt_cfg, "weight_decay", 0.0))
        self.beta1 = float(_get_section(opt_cfg, "adam_beta_1", 0.9))
        self.beta2 = float(_get_section(opt_cfg, "adam_beta_2", 0.999))
        self.adam_eps = float(_get_section(opt_cfg, "adam_eps", 1e-8))
        # Linear warmup followed by a constant learning rate.
        self.lr_warmup_steps = int(_get_section(opt_cfg, "warmup_steps", 500))
        self.lr_warmup_shape = str(_get_section(opt_cfg, "warmup_shape", "linear"))
        if self.lr_warmup_shape not in {"linear"}:
            raise ValueError(f"warmup_shape must be 'linear', got {self.lr_warmup_shape!r}")

        ema_cfg = _get_section(cfg, "ema", {})
        self.ema_halflife_kimg = float(_get_section(ema_cfg, "halflife_kimg", 500.0))
        rampup = _get_section(ema_cfg, "rampup_ratio", 0.05)
        self.ema_rampup_ratio = None if rampup is None else float(rampup)
        self.ema: StructureModuleEMA | None = None
        self._ema_swapped = False
        self._ema_last_global_step = -1

        # Post-hoc EMA (pHEMA): disabled by default so existing callers are unchanged.
        phema_cfg = _get_section(ema_cfg, "phema", {}) or {}
        self.phema_enabled = bool(_get_section(phema_cfg, "enabled", False))
        stds_raw = _get_section(phema_cfg, "stds", [0.050, 0.100])
        self.phema_stds = [float(s) for s in list(stds_raw)]
        self.phema_snapshot_every_n_steps = int(_get_section(phema_cfg, "snapshot_every_n_steps", 500))
        self.phema_prefix = str(_get_section(phema_cfg, "prefix", "phema"))
        self.phema: PowerFunctionEMA | None = None
        self._phema_param_names: list[str] = []

        # Stop early if trainable weights barely move.
        drift_cfg = _get_section(cfg, "drift_gate", {}) or {}
        self.drift_gate_enabled = bool(_get_section(drift_cfg, "enabled", True))
        self.drift_gate_step = int(_get_section(drift_cfg, "check_step", 200))
        # Allow the smaller updates produced during learning-rate warmup.
        self.drift_gate_min_rel_l2 = float(_get_section(drift_cfg, "min_rel_l2", 1e-4))
        self._init_trainable_params: list[torch.Tensor] | None = None
        self._drift_gate_fired = False

        val_cfg = _get_section(cfg, "validation_args", {})
        self.validation_recycling_steps = int(_get_section(val_cfg, "recycling_steps", 3))
        self.validation_sampling_steps = int(
            _get_section(val_cfg, "sampling_steps", self.mscd_loss.student_steps)
        )
        self.validation_diffusion_samples = int(_get_section(val_cfg, "diffusion_samples", 5))
        self.validation_symmetry_correction = bool(_get_section(val_cfg, "symmetry_correction", True))
        self.validation_run_confidence_sequentially = bool(
            _get_section(val_cfg, "run_confidence_sequentially", False)
        )
        # Boltz-2 AtomDiffusion.sample uses the Euler integrator. Gamma and step
        # scale can still be overridden for validation.
        self.validation_integrator = str(_get_section(val_cfg, "integrator", "euler"))
        g0_override = _get_section(val_cfg, "sample_gamma_0_override", None)
        self._val_gamma_override = None if g0_override is None else float(g0_override)
        ss_override = _get_section(val_cfg, "sample_step_scale_override", None)
        self._val_step_scale_override = None if ss_override is None else float(ss_override)
        # Validation defaults to the sigma grid used for training.
        rho_override = _get_section(val_cfg, "sample_rho_override", None)
        self._val_rho_override = float(
            _get_section(schedule, "rho", 7.0) if rho_override is None else rho_override
        )
        self._val_gamma_backup: float | None = None
        self._val_step_scale_backup: float | None = None
        self._val_rho_backup: float | None = None
        self._validation_metric_prefix = str(_get_section(val_cfg, "metric_prefix", "val/mscd"))
        self._validation_profile_stack: list[dict[str, Any]] = []
        self._structure_validation_args = _merge_structure_validation_args(cfg)
        self.student_struct_metrics = StandaloneValidationMetrics(False)
        self.register_buffer("_dummy", torch.tensor(0.0), persistent=False)

    def _freeze_student_for_structure_module_only(self) -> None:
        # Training reuses the frozen teacher trunk and its conditioning outputs.
        # Only the student's structure module receives gradients.
        for name, p in self.student.named_parameters():
            p.requires_grad_(name.startswith("structure_module."))

    def _apply_train_mode(self) -> None:
        self.student.eval()
        self.student.structure_module.train()
        self.teacher.eval()

    def _parameter_counts(self) -> dict[str, Any]:
        trainable = []
        frozen = []
        by_top_trainable: dict[str, int] = defaultdict(int)
        by_top_frozen: dict[str, int] = defaultdict(int)
        for name, p in self.student.named_parameters():
            top = name.split(".", 1)[0]
            if p.requires_grad:
                trainable.append(name)
                by_top_trainable[top] += p.numel()
            else:
                frozen.append(name)
                by_top_frozen[top] += p.numel()
        return {
            "trainable": trainable,
            "frozen": frozen,
            "student_trainable_numel": sum(self.student.get_parameter(n).numel() for n in trainable),
            "student_frozen_numel": sum(self.student.get_parameter(n).numel() for n in frozen),
            "student_trainable_by_submodule": dict(by_top_trainable),
            "student_frozen_by_submodule": dict(by_top_frozen),
        }

    def setup(self, stage: str) -> None:
        """Configure DDP for structure-module-only training."""
        super().setup(stage)
        strat = getattr(self.trainer, "strategy", None)
        if strat is not None and hasattr(strat, "_ddp_kwargs"):
            cur = bool(strat._ddp_kwargs.get("find_unused_parameters", False))
            if not cur:
                strat._ddp_kwargs["find_unused_parameters"] = True
                if getattr(self.trainer, "is_global_zero", True):
                    print(
                        "[mscd-distill] Forced DDP `find_unused_parameters=True` "
                        "(structure_module-only grads; pair with rank-identical MSCD)."
                    )

    def on_fit_start(self) -> None:
        if self.ema is None:
            self.ema = StructureModuleEMA(
                self.student.structure_module.parameters(),
                halflife_kimg=self.ema_halflife_kimg,
                rampup_ratio=self.ema_rampup_ratio,
            )
        self.ema.to(self.device)
        if self.phema_enabled and self.phema is None:
            named = [
                (n, p)
                for n, p in self.student.structure_module.named_parameters()
                if p.requires_grad
            ]
            self._phema_param_names = [n for n, _ in named]
            self.phema = PowerFunctionEMA(
                (p for _, p in named),
                stds=self.phema_stds,
            )
            self.phema.to(self.device)
        if self.trainer.is_global_zero:
            a = self.freeze_report
            print("\n[mscd-distill] Parameter freeze summary (student Boltz2):")
            print(f"  trainable numel (structure_module only): {a['student_trainable_numel'] / 1e6:.3f}M")
            print(f"  frozen numel:                              {a['student_frozen_numel'] / 1e6:.3f}M")
            print(f"  trainable by top-level submodule: {a['student_trainable_by_submodule']}\n")
            # diffusion_conditioning is intentionally frozen (see
            # _freeze_student_for_structure_module_only decision comment).
            dc_frozen = a["student_frozen_by_submodule"].get("diffusion_conditioning", 0)
            print(
                f"  diffusion_conditioning frozen numel: {dc_frozen / 1e6:.3f}M "
                "(intentional; shared teacher trunk supplies conditioning)"
            )
            print(
                f"  LR: max_lr={self.max_lr:g} warmup_steps={self.lr_warmup_steps} "
                f"warmup_shape={self.lr_warmup_shape}"
            )
            print(
                f"  T-anneal: {self.mscd_loss.teacher_steps_start}→"
                f"{self.mscd_loss.teacher_steps_end} over "
                f"{self.mscd_loss.teacher_steps_anneal} steps "
                f"({self.mscd_loss.teacher_steps_anneal_shape})"
            )
            print(
                f"  loss={self.mscd_loss.loss_type} weight={self.mscd_loss.weight_mode} "
                f"huber_eps_mode={self.mscd_loss.huber_eps_mode}"
            )
            if self.phema_enabled:
                print(
                    f"  pHEMA: enabled stds={self.phema_stds} "
                    f"snapshot_every_n_steps={self.phema_snapshot_every_n_steps}"
                )
            if self.drift_gate_enabled:
                print(
                    f"  drift_gate: FAIL if trainable rel L2 < {self.drift_gate_min_rel_l2:g} "
                    f"at optimizer step {self.drift_gate_step}"
                )
        # Snapshot on ALL ranks (gate runs on every rank under DDP).
        if self.drift_gate_enabled and self._init_trainable_params is None:
            self._init_trainable_params = [
                p.detach().float().cpu().clone()
                for p in self.student.structure_module.parameters()
                if p.requires_grad
            ]

    def on_train_start(self) -> None:
        self._apply_train_mode()
        # Checkpoint restore runs after on_fit_start and may reload EMA/pHEMA
        # onto CPU (on_load_checkpoint). Move shadows to the module device
        # before the first optimizer-gated update_ema call.
        if self.ema is not None:
            self.ema.to(self.device)
        if self.phema is not None:
            self.phema.to(self.device)

    def on_train_epoch_start(self) -> None:
        self._apply_train_mode()

    def _trainable_rel_l2_vs_init(self) -> float:
        """Relative L2 drift of trainable structure_module weights vs fit-start snapshot."""
        if not self._init_trainable_params:
            return 0.0
        live = [p for p in self.student.structure_module.parameters() if p.requires_grad]
        if len(live) != len(self._init_trainable_params):
            raise RuntimeError(
                f"drift_gate: param count mismatch live={len(live)} "
                f"init={len(self._init_trainable_params)}"
            )
        num = 0.0
        den = 0.0
        with torch.no_grad():
            for p, p0 in zip(live, self._init_trainable_params):
                a = p.detach().float().cpu()
                b = p0
                num += float(torch.sum((a - b) ** 2).item())
                den += float(torch.sum(b ** 2).item())
        if den <= 0.0:
            return 0.0
        return (num / den) ** 0.5

    def _maybe_enforce_drift_gate(self) -> None:
        if not self.drift_gate_enabled or self._drift_gate_fired:
            return
        if self._init_trainable_params is None:
            return
        step = int(self.global_step)
        if step <= 0:
            return
        # Log at 50 / 100 / check_step; hard-fail at check_step if under-travel.
        log_steps = {50, 100, int(self.drift_gate_step)}
        if step not in log_steps:
            return
        rel = self._trainable_rel_l2_vs_init()
        if getattr(self.trainer, "is_global_zero", True):
            print(
                f"[mscd-drift] step={step} trainable_rel_l2_vs_init={rel:.6g} "
                f"(gate_min={self.drift_gate_min_rel_l2:g} @ step {self.drift_gate_step})"
            )
        self.log(
            "mscd/trainable_rel_l2_vs_init",
            torch.tensor(rel, device=self.device),
            on_step=True,
            sync_dist=False,
        )
        if step == int(self.drift_gate_step):
            self._drift_gate_fired = True
            if rel < float(self.drift_gate_min_rel_l2):
                raise RuntimeError(
                    f"[mscd-drift] FATAL: trainable rel L2 vs init = {rel:.6g} "
                    f"< min {self.drift_gate_min_rel_l2:g} after {step} optimizer steps. "
                    "Trainable parameters are not updating as expected."
                )

    def configure_optimizers(self):
        params = [p for p in self.student.structure_module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.max_lr,
            betas=(self.beta1, self.beta2),
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )
        if self.lr_warmup_steps <= 0:
            return optimizer

        warmup_steps = int(self.lr_warmup_steps)

        def lr_lambda(step: int) -> float:
            # Lightning passes the number of optimizer steps already taken.
            if warmup_steps <= 0:
                return 1.0
            # step is 0-based on first call before the first update in some PL
            # versions; clamp so LR starts near peak/warmup_steps and reaches 1.0.
            progress = min(max(int(step) + 1, 0), warmup_steps) / float(warmup_steps)
            return float(progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _prepare_targets(
        self,
        batch: dict[str, torch.Tensor],
        multiplicity: int,
        augmentation: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = batch["coords"]
        bsz, n_samples, n_atoms = coords.shape[:3]
        atom_coords = coords.reshape(bsz * n_samples, n_atoms, 3)
        atom_pad_mask = batch["atom_pad_mask"]
        resolved_mask = batch["atom_resolved_mask"].float()
        if multiplicity > 1:
            atom_coords = atom_coords.repeat_interleave(multiplicity, dim=0)
            atom_pad_mask = atom_pad_mask.repeat_interleave(multiplicity, dim=0)
            resolved_mask = resolved_mask.repeat_interleave(multiplicity, dim=0)
        y = center_random_augmentation(
            atom_coords,
            atom_pad_mask,
            augmentation=augmentation,
            centering=True,
        )
        align_w = y.new_ones(y.shape[:2])
        if self.upweight_nucleic_ligand_atoms:
            atom_type = (
                torch.bmm(
                    batch["atom_to_token"].float(),
                    batch["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            if multiplicity > 1:
                atom_type = atom_type.repeat_interleave(multiplicity, dim=0)
            dna_or_rna = (
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
            )
            lig = torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
            align_w = align_w * (1.0 + 5.0 * dna_or_rna + 10.0 * lig)
        if self.normalize_align_weights_per_sample:
            align_w = normalize_atom_weights_per_sample(align_w, resolved_mask)
        return y, atom_pad_mask.float(), resolved_mask, align_w

    def _shared_trunk(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        recycles = int(_get_section(_get_section(_get_section(self.cfg, "checkpointing"), "distill", {}), "recycles", 3))
        return compute_trunk_features(
            self.teacher,
            batch,
            recycles=recycles,
            enable_grad=False,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # Avoid mid-step collectives because data loading can progress at
        # different rates on each rank. DDP synchronizes gradients.
        self.mscd_loss.set_global_step(int(self.global_step), batch_idx=int(batch_idx))
        trunk = self._shared_trunk(batch)
        y, _atom_pad_mask, resolved_mask, align_w = self._prepare_targets(
            batch,
            multiplicity=self.train_multiplicity,
            augmentation=True,
        )
        loss = self.mscd_loss(
            student_structure=self.student.structure_module,
            teacher_structure=self.teacher.structure_module,
            student_trunk=trunk,
            teacher_trunk=trunk,
            y=y,
            align_w=align_w,
            resolved_mask=resolved_mask,
            multiplicity=self.train_multiplicity,
            synchronize_sigmas_across_multiplicity=self.synchronize_sigmas_across_multiplicity,
        )
        # Step metrics remain rank-local.
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        for key, value in self.mscd_loss.last_metrics.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(key == "mscd/T_edges"),
                sync_dist=False,
            )
        self.log(
            "mscd/sigma_data",
            torch.tensor(float(self.mscd_loss.sigma_data), device=self.device),
            on_step=True,
            sync_dist=False,
        )
        return loss

    def _effective_batch_images(self, batch: dict[str, torch.Tensor]) -> int:
        """Images per optimizer step = microbatch × world_size × accumulate_grad_batches."""
        micro = int(batch["coords"].shape[0])
        world = max(int(getattr(self.trainer, "world_size", 1)), 1)
        accum = max(int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1), 1)
        return micro * world * accum

    def _grads_still_accumulating(self) -> bool:
        """Return whether Lightning is still accumulating microbatches."""
        fit_loop = getattr(self.trainer, "fit_loop", None)
        should_acc = getattr(fit_loop, "_should_accumulate", None)
        if not callable(should_acc):
            raise RuntimeError(
                "[mscd-EMA] trainer.fit_loop._should_accumulate missing; "
                "cannot gate EMA on the optimizer-step boundary (PL API change?). "
                "Refusing to update rather than over-update on micro-batches."
            )
        return bool(should_acc())

    def update_ema(self, batch: dict[str, torch.Tensor]) -> None:
        if self.ema is None:
            return
        # Exactly once per optimizer step (see on_train_batch_end + _should_accumulate).
        step = int(self.global_step)
        if step == 0:
            raise RuntimeError(
                "[mscd-EMA] update_ema called at global_step=0. EMA must only run "
                "after an optimizer step — gate on trainer.fit_loop._should_accumulate "
                "(PL 2.4; Trainer.should_accumulate does not exist)."
            )
        if step == self._ema_last_global_step:
            return
        batch_size = self._effective_batch_images(batch)
        cur_nimg = max(step * batch_size, batch_size)
        beta = self.ema.update(
            self.student.structure_module.parameters(),
            batch_size=batch_size,
            cur_nimg=cur_nimg,
        )
        self._ema_last_global_step = step
        # EMA updates must track optimizer steps one-to-one.
        self.ema.assert_update_cadence(global_step=step, context="mscd-EMA")
        self.log("ema/beta", torch.tensor(beta, device=self.device), on_step=True, prog_bar=False, sync_dist=False)
        self.log(
            "ema/num_updates",
            torch.tensor(float(self.ema.num_updates), device=self.device),
            on_step=True,
            prog_bar=False,
            sync_dist=False,
        )

        if self.phema is not None:
            phema_betas = self.phema.update(
                self.student.structure_module.parameters(),
                cur_nimg=cur_nimg,
                batch_size=batch_size,
            )
            # Keep pHEMA in lockstep with structure EMA (same num_updates).
            if int(self.phema.num_updates) != int(self.ema.num_updates):
                raise RuntimeError(
                    f"[mscd-pHEMA] cadence desync vs EMA: phema.num_updates="
                    f"{self.phema.num_updates} != ema.num_updates={self.ema.num_updates}"
                )
            if int(self.phema.num_updates) != int(step):
                raise RuntimeError(
                    f"[mscd-pHEMA] EMA update cadence mismatch: num_updates="
                    f"{self.phema.num_updates} != global_step={step}. "
                    f"pHEMA must update exactly once per optimizer step "
                    f"(account for accumulate_grad_batches)."
                )
            self.log(
                "phema/beta_0",
                torch.tensor(float(phema_betas[0]), device=self.device),
                on_step=True,
                sync_dist=False,
            )
            if (
                self.phema_snapshot_every_n_steps > 0
                and step > 0
                and step % self.phema_snapshot_every_n_steps == 0
            ):
                self._save_phema_snapshots(step)

    def _save_phema_snapshots(self, step: int) -> None:
        if self.phema is None or not getattr(self.trainer, "is_global_zero", True):
            return
        out_dir = Path(getattr(self.trainer, "default_root_dir", None) or ".")
        phema_dir = out_dir / "phema"
        phema_dir.mkdir(parents=True, exist_ok=True)
        named = self.phema.named_state_dicts(self._phema_param_names)
        for std, state in zip(self.phema.stds, named):
            path = phema_dir / snapshot_filename(self.phema_prefix, step, std)
            torch.save(
                {
                    "structure_module": state,
                    "step": int(step),
                    "std": float(std),
                    "nimg": int(step),
                    "format": "boltz_mscd_phema_v1",
                },
                path,
            )
            print(f"[mscd-pHEMA] wrote {path}")

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:  # noqa: ANN001
        # EMA and pHEMA are updated by StructureModuleEMACallback before the
        # checkpoint callback runs.
        if self._grads_still_accumulating():
            return
        self._maybe_enforce_drift_gate()

    def swap_in_ema(self) -> None:
        if self.ema is None or self._ema_swapped:
            return
        self.ema.store(self.student.structure_module.parameters())
        self.ema.copy_to(self.student.structure_module.parameters())
        self._ema_swapped = True

    def restore_from_ema(self) -> None:
        if self.ema is None or not self._ema_swapped:
            return
        self.ema.restore(self.student.structure_module.parameters())
        self._ema_swapped = False

    def _apply_val_sampling_overrides(self) -> None:
        """Deterministic val sampler: gamma_0=0 (no churn), step_scale override (default 1.0).

        Boltz-2 ``AtomDiffusion.sample`` is Euler-only (no integrator kwarg); overrides
        apply to ``gamma_0`` / ``step_scale`` on the live module.
        """
        sm = self.student.structure_module
        if self._val_gamma_override is not None:
            self._val_gamma_backup = float(sm.gamma_0)
            sm.gamma_0 = float(self._val_gamma_override)
        if self._val_step_scale_override is not None:
            self._val_step_scale_backup = float(sm.step_scale)
            sm.step_scale = float(self._val_step_scale_override)
        if self._val_rho_override is not None and hasattr(sm, "rho"):
            if float(sm.rho) != float(self._val_rho_override):
                print(
                    f"[mscd-val] sampler rho {float(sm.rho)} -> {float(self._val_rho_override)} "
                    f"(matching the supervised sigma grid)"
                )
            self._val_rho_backup = float(sm.rho)
            sm.rho = float(self._val_rho_override)

    def _restore_val_sampling_overrides(self) -> None:
        sm = self.student.structure_module
        if self._val_gamma_backup is not None:
            sm.gamma_0 = float(self._val_gamma_backup)
            self._val_gamma_backup = None
        if self._val_step_scale_backup is not None:
            sm.step_scale = float(self._val_step_scale_backup)
            self._val_step_scale_backup = None
        if self._val_rho_backup is not None:
            sm.rho = float(self._val_rho_backup)
            self._val_rho_backup = None

    def push_validation_profile(
        self,
        *,
        metric_prefix: str,
        sampling_steps: int | None = None,
        diffusion_samples: int | None = None,
        integrator: str | None = None,
        sample_gamma_0_override: float | None = None,
        sample_step_scale_override: float | None = None,
        sample_rho_override: float | None = None,
    ) -> None:
        """Temporarily override validation sampling settings for a single validate() call."""
        self._validation_profile_stack.append(
            {
                "validation_sampling_steps": self.validation_sampling_steps,
                "validation_diffusion_samples": self.validation_diffusion_samples,
                "validation_integrator": self.validation_integrator,
                "_val_gamma_override": self._val_gamma_override,
                "_val_step_scale_override": self._val_step_scale_override,
                "_val_rho_override": self._val_rho_override,
                "_validation_metric_prefix": self._validation_metric_prefix,
            }
        )
        self._validation_metric_prefix = str(metric_prefix)
        if sampling_steps is not None:
            self.validation_sampling_steps = int(sampling_steps)
        if diffusion_samples is not None:
            self.validation_diffusion_samples = int(diffusion_samples)
        if integrator is not None:
            self.validation_integrator = str(integrator)
        self._val_gamma_override = None if sample_gamma_0_override is None else float(sample_gamma_0_override)
        self._val_step_scale_override = (
            None if sample_step_scale_override is None else float(sample_step_scale_override)
        )
        # Unlike gamma/step_scale, an unspecified rho KEEPS the supervised grid rather
        # than clearing to None — a profile that forgets rho must not silently sample
        # off-grid.
        if sample_rho_override is not None:
            self._val_rho_override = float(sample_rho_override)

    def pop_validation_profile(self) -> None:
        """Restore validation sampling settings after push_validation_profile()."""
        if not self._validation_profile_stack:
            return
        state = self._validation_profile_stack.pop()
        self.validation_sampling_steps = int(state["validation_sampling_steps"])
        self.validation_diffusion_samples = int(state["validation_diffusion_samples"])
        self.validation_integrator = str(state["validation_integrator"])
        self._val_gamma_override = state["_val_gamma_override"]
        self._val_step_scale_override = state["_val_step_scale_override"]
        self._val_rho_override = state["_val_rho_override"]
        self._validation_metric_prefix = str(state["_validation_metric_prefix"])

    def on_validation_epoch_start(self) -> None:
        self.swap_in_ema()
        self._apply_val_sampling_overrides()
        self.student.eval()
        self.student_struct_metrics.reset_all()
        self.student_struct_metrics.to(self.device)

    def on_validation_epoch_end(self) -> None:
        log_structure_validation_epoch(
            self.student_struct_metrics.as_refs(),
            self,
            prefix=self._validation_metric_prefix,
            confidence_prediction=False,
        )
        self._restore_val_sampling_overrides()
        self.restore_from_ema()
        self._apply_train_mode()

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # Boltz-2 sample is Euler-only; validation_integrator is recorded for logs.
        out = self.student(
            batch,
            recycling_steps=self.validation_recycling_steps,
            num_sampling_steps=self.validation_sampling_steps,
            diffusion_samples=self.validation_diffusion_samples,
            run_confidence_sequentially=self.validation_run_confidence_sequentially,
        )
        accumulate_structure_validation_batch(
            self.student_struct_metrics.as_refs(),
            self.student,
            batch,
            out,
            diffusion_samples=self.validation_diffusion_samples,
            symmetry_correction=self.validation_symmetry_correction,
            confidence_prediction=False,
        )
        self.log(
            f"{self._validation_metric_prefix}/sampling_steps",
            torch.tensor(float(self.validation_sampling_steps), device=self.device),
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            f"{self._validation_metric_prefix}/integrator",
            torch.tensor(0.0, device=self.device),  # Euler-only on Boltz-2
            on_epoch=True,
            sync_dist=False,
        )
        return None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.ema is not None:
            checkpoint["mscd_structure_ema"] = self.ema.state_dict()
        if self.phema is not None:
            checkpoint["mscd_structure_phema"] = self.phema.state_dict()
            checkpoint["mscd_phema_param_names"] = list(self._phema_param_names)
        # Record the active teacher grid in each training checkpoint.
        t_now = int(self.mscd_loss.current_teacher_steps())
        checkpoint["mscd_T_edges"] = t_now
        if getattr(self.trainer, "is_global_zero", True):
            print(
                f"[mscd-distill] checkpoint global_step={int(self.global_step)} "
                f"T_edges={t_now} "
                f"(anneal {self.mscd_loss.teacher_steps_start}→"
                f"{self.mscd_loss.teacher_steps_end} / "
                f"{self.mscd_loss.teacher_steps_anneal} "
                f"{self.mscd_loss.teacher_steps_anneal_shape})"
            )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Restore EMA tensors directly onto the module device.
        device = self.device
        state = checkpoint.get("mscd_structure_ema")
        if state is not None:
            self.ema = StructureModuleEMA(
                self.student.structure_module.parameters(),
                halflife_kimg=self.ema_halflife_kimg,
                rampup_ratio=self.ema_rampup_ratio,
            )
            self.ema.load_state_dict(state, device=device)
            self._ema_last_global_step = int(self.ema.num_updates)
        phema_state = checkpoint.get("mscd_structure_phema")
        if phema_state is not None and self.phema_enabled:
            named = [
                (n, p)
                for n, p in self.student.structure_module.named_parameters()
                if p.requires_grad
            ]
            self._phema_param_names = list(
                checkpoint.get("mscd_phema_param_names") or [n for n, _ in named]
            )
            self.phema = PowerFunctionEMA((p for _, p in named), stds=self.phema_stds)
            self.phema.load_state_dict(phema_state, device=device)

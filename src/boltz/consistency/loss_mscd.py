from __future__ import annotations

import math
from typing import Any

import torch

from boltz.data import const
from boltz.consistency.mscd_ops import (
    heun_hop_protein,
    inv_ddim_edm,
    make_karras_sigmas,
    partition_edges_by_sigma,
    sample_segment_and_teacher_pair,
    filter_teacher_edges_by_sigma,
    _run_structure,
)
# Boltz-2 loss path (sign(det) Kabsch fix lives in diffusionv2.weighted_rigid_align)
from boltz.model.loss.diffusionv2 import smooth_lddt_loss, weighted_rigid_align


def _huber_loss(x: torch.Tensor, delta: float) -> torch.Tensor:
    abs_x = x.abs()
    quad = torch.minimum(abs_x, torch.as_tensor(delta, device=x.device, dtype=x.dtype))
    return 0.5 * quad.square() + (abs_x - quad) * delta


def pseudo_huber_vector_norm(norm_sq: torch.Tensor, c: torch.Tensor | float) -> torch.Tensor:
    """EDM/Song pseudo-Huber on a precomputed squared L2 norm: sqrt(||e||^2 + c^2) - c.

    Reference: ``edm2/training/loss_cd.py:_pseudo_huber_vector_norm`` (fixed c=1e-4
    on ImageNet). Song et al. consistency-models scale ``c = c_base * sqrt(D)``.
    """
    c_t = torch.as_tensor(c, device=norm_sq.device, dtype=norm_sq.dtype)
    return torch.sqrt(norm_sq + c_t.square()) - c_t


def song_huber_c(data_dim: torch.Tensor | float, c_base: float = 0.00054) -> torch.Tensor | float:
    """Song et al. consistency-models Huber constant: ``c = c_base * sqrt(D)``."""
    if isinstance(data_dim, torch.Tensor):
        return float(c_base) * torch.sqrt(data_dim.to(torch.float32).clamp(min=1.0))
    return float(c_base) * math.sqrt(max(float(data_dim), 1.0))


def sqrt_karras_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """``sqrt(σ² + σ_data²) / (σ · σ_data)`` — EDM2 ``weight_mode=sqrt_karras``."""
    sigma = sigma.to(torch.float32)
    return torch.sqrt(sigma.square() + sigma_data**2) / (sigma * sigma_data).clamp(min=1e-20)


def _restore_rng_state(device: torch.device, state: torch.Tensor) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device=device)
    else:
        torch.random.set_rng_state(state)


def _capture_rng_state(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device=device)
    return torch.random.get_rng_state()


def _mscd_edge_sample_generator(device: torch.device, global_step: int, batch_idx: int) -> torch.Generator:
    """Rank-identical RNG for MSCD edge-type masks (no collectives).

    Seeding from ``global_step`` and ``batch_idx`` gives every rank the same
    masks without adding a collective operation to the training step.
    """
    gen = torch.Generator(device=device)
    seed = (0x4D534344 ^ (int(global_step) * 1_000_003) ^ (int(batch_idx) * 97)) & 0xFFFFFFFFFFFF
    gen.manual_seed(int(seed))
    return gen


class MSCDLoss:
    """Multistep consistency distillation loss for Boltz atom coordinates."""

    def __init__(
        self,
        *,
        student_steps: int = 8,
        teacher_steps_start: int = 200,
        teacher_steps_end: int = 350,
        teacher_steps_anneal: int = 3000,
        teacher_steps_anneal_shape: str = "log_linear",
        teacher_steps_log_rate_per_step: float | None = None,
        sigma_min: float = 0.004,
        sigma_max: float = 160.0,
        sigma_data: float = 16.0,
        rho: float = 40.0,
        sampling_mode: str = "edm",
        terminal_anchor: bool = True,
        terminal_teacher_hop: bool = True,
        # Fraction of edge-sampling mass moved onto the segment-top edges, which are
        # the only sigmas the student is ever queried at during inference.
        # 0.0 preserves the default edge-sampling distribution.
        segment_top_mass: float = 0.0,
        # Defaults match Boltz AtomDiffusion.noise_distribution (P_mean=-1.2, P_std=1.5)
        # interpreted in *relative* sigma space: sigma = sigma_data * exp(P_mean + P_std * N).
        # The importance sampler in mscd_ops.compute_importance_weights shifts log(sigma)
        # by log(sigma_data) so these values do not need rescaling per dataset.
        P_mean: float = -1.2,
        P_std: float = 1.5,
        loss_type: str = "pseudo_huber",
        weight_mode: str = "sqrt_karras",
        huber_eps: float = 1e-4,
        # "fixed": c = huber_eps (EDM discrete CD). "song_sqrt_d": c = huber_c_base * sqrt(D)
        # with D = 3 * sum(atom weights) per sample (Boltz atom-coordinate dim).
        huber_eps_mode: str = "fixed",
        huber_c_base: float = 0.00054,
        # "vector": EDM reference sqrt(||e||^2+c^2)-c on weighted coord L2.
        # "rms": alternate form that divides by 3*sum(w) inside the sqrt.
        pseudo_huber_reduce: str = "vector",
        smooth_lddt_loss_weight: float = 0.0,
        sync_dropout: bool = True,
        debug_invariants: bool = False,
        # Teacher Heun target hop: 1.0 = unscaled PF step; Boltz inference default is 1.5.
        teacher_step_scale: float = 1.0,
    ) -> None:
        if student_steps < 1:
            raise ValueError(f"student_steps must be >= 1, got {student_steps}")
        if teacher_steps_start < 2 or teacher_steps_end < teacher_steps_start:
            raise ValueError("teacher step schedule must satisfy 2 <= start <= end")
        if loss_type not in {"pseudo_huber", "l2_root", "huber", "l2"}:
            raise ValueError(f"Unknown loss_type {loss_type!r}")
        if float(teacher_step_scale) <= 0:
            raise ValueError(f"teacher_step_scale must be positive, got {teacher_step_scale}")
        if weight_mode not in {
            "edm",
            "vlike",
            "flat",
            "snr",
            "snr+1",
            "karras",
            "sqrt_karras",
            "truncated-snr",
            "uniform",
        }:
            raise ValueError(f"Unknown weight_mode {weight_mode!r}")
        if teacher_steps_anneal_shape not in {"log_linear", "linear"}:
            raise ValueError(
                f"teacher_steps_anneal_shape must be 'log_linear' or 'linear', "
                f"got {teacher_steps_anneal_shape!r}"
            )
        if teacher_steps_log_rate_per_step is not None:
            if teacher_steps_anneal_shape != "log_linear":
                raise ValueError(
                    "teacher_steps_log_rate_per_step requires log_linear annealing"
                )
            if (
                not math.isfinite(float(teacher_steps_log_rate_per_step))
                or float(teacher_steps_log_rate_per_step) <= 0.0
            ):
                raise ValueError(
                    "teacher_steps_log_rate_per_step must be finite and positive"
                )
        if huber_eps_mode not in {"fixed", "song_sqrt_d"}:
            raise ValueError(f"huber_eps_mode must be 'fixed' or 'song_sqrt_d', got {huber_eps_mode!r}")
        if pseudo_huber_reduce not in {"vector", "rms"}:
            raise ValueError(
                f"pseudo_huber_reduce must be 'vector' or 'rms', got {pseudo_huber_reduce!r}"
            )

        self.student_steps = int(student_steps)
        self.teacher_steps_start = int(teacher_steps_start)
        self.teacher_steps_end = int(teacher_steps_end)
        self.teacher_steps_anneal = int(teacher_steps_anneal)
        self.teacher_steps_anneal_shape = str(teacher_steps_anneal_shape)
        self.teacher_steps_log_rate_per_step = (
            None
            if teacher_steps_log_rate_per_step is None
            else float(teacher_steps_log_rate_per_step)
        )
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.sampling_mode = str(sampling_mode)
        self.terminal_anchor = bool(terminal_anchor)
        self.segment_top_mass = float(segment_top_mass)
        self.terminal_teacher_hop = bool(terminal_teacher_hop)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.loss_type = str(loss_type)
        self.weight_mode = str(weight_mode)
        self.huber_eps = float(huber_eps)
        self.huber_eps_mode = str(huber_eps_mode)
        self.huber_c_base = float(huber_c_base)
        self.pseudo_huber_reduce = str(pseudo_huber_reduce)
        self.smooth_lddt_loss_weight = float(smooth_lddt_loss_weight)
        self.teacher_step_scale = float(teacher_step_scale)
        self.sync_dropout = bool(sync_dropout)
        self.debug_invariants = bool(debug_invariants)
        self._global_step = 0
        self._batch_idx = 0
        self._filter_cache: dict[int, tuple[torch.Tensor, int]] = {}
        self.last_metrics: dict[str, torch.Tensor] = {}

    @property
    def sigma_min_abs(self) -> float:
        return self.sigma_min * self.sigma_data

    @property
    def sigma_max_abs(self) -> float:
        return self.sigma_max * self.sigma_data

    def set_global_step(self, step: int, batch_idx: int = 0) -> None:
        self._global_step = max(int(step), 0)
        self._batch_idx = max(int(batch_idx), 0)

    def current_teacher_steps(self) -> int:
        if self.teacher_steps_anneal <= 0:
            return self.teacher_steps_end
        ratio = min(max(self._global_step / float(self.teacher_steps_anneal), 0.0), 1.0)
        if self.teacher_steps_anneal_shape == "linear":
            t_now = self.teacher_steps_start + ratio * (
                self.teacher_steps_end - self.teacher_steps_start
            )
            return max(
                self.teacher_steps_start,
                min(self.teacher_steps_end, int(round(t_now))),
            )
        # log_linear (EDM2 discrete CD / prior Boltz default)
        if self.teacher_steps_log_rate_per_step is None:
            log_t = math.log(self.teacher_steps_start) + ratio * (
                math.log(self.teacher_steps_end) - math.log(self.teacher_steps_start)
            )
        else:
            capped_step = min(self._global_step, self.teacher_steps_anneal)
            log_t = (
                math.log(self.teacher_steps_start)
                + capped_step * self.teacher_steps_log_rate_per_step
            )
        return max(self.teacher_steps_start, min(self.teacher_steps_end, int(round(math.exp(log_t)))))

    def _weight(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.to(torch.float32)
        if self.weight_mode == "edm":
            return (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square().clamp(min=1e-20)
        if self.weight_mode == "vlike":
            return 1.0 / sigma.square().clamp(min=1e-20) + 1.0
        if self.weight_mode in {"flat", "uniform"}:
            return torch.ones_like(sigma)
        snr = 1.0 / sigma.square().clamp(min=1e-20)
        if self.weight_mode == "snr":
            return snr
        if self.weight_mode == "snr+1":
            return snr + 1.0
        if self.weight_mode == "karras":
            return snr + 1.0 / (self.sigma_data**2)
        if self.weight_mode == "sqrt_karras":
            return sqrt_karras_weight(sigma, self.sigma_data)
        if self.weight_mode == "truncated-snr":
            return torch.clamp(snr, min=1.0)
        raise AssertionError(f"Unhandled weight mode {self.weight_mode}")

    def _huber_c(self, data_dim: torch.Tensor) -> torch.Tensor:
        if self.huber_eps_mode == "song_sqrt_d":
            return song_huber_c(data_dim, c_base=self.huber_c_base).to(data_dim.device)
        return torch.full_like(data_dim, float(self.huber_eps), dtype=torch.float32)

    def _build_student_grid(self, device: torch.device) -> torch.Tensor:
        sigmas = make_karras_sigmas(
            num_nodes=self.student_steps,
            sigma_min=self.sigma_min_abs,
            sigma_max=self.sigma_max_abs,
            rho=self.rho,
            round_fn=None,
            device=device,
        )
        return torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)], dim=0)

    def _build_teacher_grid(self, student_sigmas: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
        target_steps = self.current_teacher_steps()
        cached = self._filter_cache.get(target_steps)
        if cached is not None:
            sigmas, terminal_k = cached
            return sigmas.to(device), terminal_k

        raw_steps = target_steps
        while True:
            sigmas = make_karras_sigmas(
                num_nodes=raw_steps,
                sigma_min=self.sigma_min_abs,
                sigma_max=self.sigma_max_abs,
                rho=self.rho,
                round_fn=None,
                device=device,
            )
            teacher_full = torch.cat(
                [sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)],
                dim=0,
            )
            teacher_filtered, terminal_k = filter_teacher_edges_by_sigma(student_sigmas, teacher_full)
            if teacher_filtered.shape[0] - 1 >= target_steps:
                break
            raw_steps += 1

        self._filter_cache[target_steps] = (teacher_filtered.detach().cpu(), terminal_k)
        return teacher_filtered, terminal_k

    @staticmethod
    def _align_prediction(
        pred: torch.Tensor,
        target: torch.Tensor,
        align_w: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Align student pred onto target with Kabsch (sign(det) reflection-safe).
        device_type = "cuda" if pred.is_cuda else "cpu"
        with torch.no_grad(), torch.autocast(device_type, enabled=False):
            _, rot, true_centroid, pred_centroid = weighted_rigid_align(
                pred.detach().float(),
                target.detach().float(),
                align_w.detach().float(),
                mask.detach().float(),
                return_transform=True,
            )
        centered = pred - true_centroid.to(pred)
        return torch.einsum("bni,bji->bnj", centered, rot.to(pred)) + pred_centroid.to(pred)

    def _reduce_loss(
        self,
        diff: torch.Tensor,
        sigma_t: torch.Tensor,
        align_w: torch.Tensor,
        resolved_mask: torch.Tensor,
    ) -> torch.Tensor:
        w = (align_w * resolved_mask).to(diff)
        # Effective coordinate dimensionality for Song c-scaling: 3 * sum(atom weights).
        # For binary masks this is 3 * n_resolved_atoms (Boltz atom xyz dim).
        data_dim = torch.clamp(3.0 * w.sum(dim=-1), min=1.0)
        denom = data_dim.clamp(min=1e-8)
        weighted_norm_sq = (diff.square().sum(dim=-1) * w).sum(dim=-1)
        if self.loss_type == "pseudo_huber":
            c = self._huber_c(data_dim)
            if self.pseudo_huber_reduce == "rms":
                # RMS inside the square root with a fixed epsilon.
                per_sample = (
                    torch.sqrt(weighted_norm_sq / denom + self.huber_eps * self.huber_eps)
                    - self.huber_eps
                )
            else:
                # EDM2 reference: sqrt(||e||^2 + c^2) - c on the weighted vector norm.
                per_sample = pseudo_huber_vector_norm(weighted_norm_sq, c)
        elif self.loss_type == "l2_root":
            if self.pseudo_huber_reduce == "rms":
                per_sample = torch.sqrt(weighted_norm_sq / denom + self.huber_eps * self.huber_eps)
            else:
                per_sample = torch.sqrt(weighted_norm_sq.clamp(min=1e-12))
        elif self.loss_type == "huber":
            per_atom = _huber_loss(diff, self.huber_eps).sum(dim=-1)
            per_sample = (per_atom * w).sum(dim=-1) / denom
        else:
            per_atom = diff.square().sum(dim=-1)
            per_sample = (per_atom * w).sum(dim=-1) / denom
        return (per_sample * self._weight(sigma_t)).mean()

    def __call__(
        self,
        *,
        student_structure: torch.nn.Module,
        teacher_structure: torch.nn.Module,
        student_trunk: dict[str, Any],
        teacher_trunk: dict[str, Any],
        y: torch.Tensor,
        align_w: torch.Tensor,
        resolved_mask: torch.Tensor,
        multiplicity: int,
        synchronize_sigmas_across_multiplicity: bool = False,
    ) -> torch.Tensor:
        device = y.device
        batch_size = y.shape[0]
        student_sigmas = self._build_student_grid(device)
        teacher_sigmas, terminal_k = self._build_teacher_grid(student_sigmas, device)
        if self.debug_invariants and student_sigmas[0] < 0.5 * self.sigma_max_abs:
            raise AssertionError("MSCD student grid is not in absolute Boltz sigma space")
        sigma_bounds = partition_edges_by_sigma(student_sigmas, teacher_sigmas)
        sample_batch_size = batch_size
        repeat_edges = bool(synchronize_sigmas_across_multiplicity and multiplicity > 1)
        if repeat_edges:
            if batch_size % int(multiplicity) != 0:
                raise ValueError(
                    "Cannot synchronize MSCD sigmas across multiplicity: "
                    f"expanded batch {batch_size} is not divisible by multiplicity {multiplicity}"
                )
            sample_batch_size = batch_size // int(multiplicity)
        # Rank-identical edge masks via seeded RNG (no broadcast / barrier).
        edge_gen = _mscd_edge_sample_generator(device, self._global_step, self._batch_idx)
        sample = sample_segment_and_teacher_pair(
            sigma_bounds=sigma_bounds,
            teacher_sigmas=teacher_sigmas,
            student_sigmas=student_sigmas,
            batch_size=sample_batch_size,
            device=device,
            generator=edge_gen,
            terminal_k=terminal_k,
            sampling_mode=self.sampling_mode,
            rho=self.rho,
            P_mean=self.P_mean,
            P_std=self.P_std,
            terminal_anchor=self.terminal_anchor,
            segment_top_mass=self.segment_top_mass,
            sigma_data=self.sigma_data,
        )
        if repeat_edges:
            sample = {
                key: value.repeat_interleave(int(multiplicity), dim=0) if torch.is_tensor(value) else value
                for key, value in sample.items()
            }

        sigma_t_vec = sample["sigma_t"]
        sigma_s_teacher_vec = sample["sigma_s"]
        sigma_bdry_vec = sample["sigma_bdry"]
        is_terminal = sample["is_terminal"].bool()
        is_boundary_snap = sample["is_boundary_snap"].bool()
        general_mask = (~is_terminal) & (~is_boundary_snap)

        sigma_s_eff = sigma_s_teacher_vec.clone()
        sigma_s_eff = torch.where(is_boundary_snap, sigma_bdry_vec, sigma_s_eff)
        sigma_s_eff = torch.where(general_mask, torch.maximum(sigma_s_teacher_vec, sigma_bdry_vec), sigma_s_eff)
        sigma_s_eff = torch.where(is_terminal, torch.zeros_like(sigma_s_eff), sigma_s_eff)

        eps = torch.randn_like(y, dtype=torch.float64)
        x_t = (y.to(torch.float64) + sigma_t_vec.to(torch.float64).reshape(-1, 1, 1) * eps).to(y.dtype)

        rng_state = _capture_rng_state(device) if self.sync_dropout else None
        x_hat_t = _run_structure(
            student_structure,
            student_trunk,
            x_t,
            sigma_t_vec.to(x_t.device),
            multiplicity=multiplicity,
        )

        # Keep every rank on the same full-batch Heun path. Terminal entries use
        # a positive placeholder sigma and are masked out below.
        sigma_s_heun = torch.where(
            is_terminal,
            (0.5 * sigma_t_vec).clamp(min=1e-4),
            sigma_s_eff.clamp(min=1e-4),
        )
        with torch.no_grad():
            x_s_teach_full = heun_hop_protein(
                teacher_structure,
                teacher_trunk,
                x_t,
                sigma_t_vec,
                sigma_s_heun,
                multiplicity=multiplicity,
                step_scale=self.teacher_step_scale,
            )
        x_s_teach = torch.where(
            is_terminal.view(-1, 1, 1),
            torch.zeros_like(x_s_teach_full),
            x_s_teach_full,
        )

        x_ref = torch.zeros_like(x_t, dtype=torch.float64)
        sigma_ref_vec = sigma_t_vec.new_zeros(batch_size).to(torch.float64)
        # Always one extra teacher forward when terminal_teacher_hop (discard if none).
        if self.terminal_teacher_hop:
            with torch.no_grad():
                t_out = _run_structure(
                    teacher_structure,
                    teacher_trunk,
                    x_t,
                    sigma_t_vec.to(x_t.device),
                    multiplicity=multiplicity,
                ).to(torch.float64)
            x_ref = torch.where(is_terminal.view(-1, 1, 1), t_out, x_ref)
        else:
            y64 = y.to(torch.float64)
            x_ref = torch.where(is_terminal.view(-1, 1, 1), y64, x_ref)
        # Boundary / general assignments are pure tensor ops (no module calls).
        x_s64 = x_s_teach.to(torch.float64)
        x_ref = torch.where(is_boundary_snap.view(-1, 1, 1), x_s64, x_ref)
        sigma_ref_vec = torch.where(
            is_boundary_snap,
            sigma_s_eff.to(torch.float64),
            sigma_ref_vec,
        )

        # Always run the second student forward on the full batch (sync_dropout
        # path). Even when sync_dropout=False, keep the full-batch call so DDP
        # ranks never diverge on subset gathers.
        with torch.no_grad():
            if self.sync_dropout and rng_state is not None:
                _restore_rng_state(device, rng_state)
            target_x = torch.where(general_mask.view(-1, 1, 1), x_s_teach, x_t)
            target_sigma = torch.where(general_mask, sigma_s_eff, sigma_t_vec)
            x_hat_full = _run_structure(
                student_structure,
                student_trunk,
                target_x,
                target_sigma.to(target_x.device),
                multiplicity=multiplicity,
            ).to(torch.float64)
        ratio = (
            sigma_bdry_vec.to(torch.float64).reshape(-1, 1, 1)
            / sigma_s_eff.to(torch.float64).reshape(-1, 1, 1).clamp(min=1e-12)
        )
        x_ref_general = x_hat_full + ratio * (x_s64 - x_hat_full)
        x_ref = torch.where(general_mask.view(-1, 1, 1), x_ref_general, x_ref)
        sigma_ref_vec = torch.where(
            general_mask,
            sigma_bdry_vec.to(torch.float64),
            sigma_ref_vec,
        )

        x_target_t = inv_ddim_edm(
            x_ref=x_ref,
            x_t=x_t.to(torch.float64),
            sigma_t=sigma_t_vec.to(torch.float64),
            sigma_ref=sigma_ref_vec,
        ).to(x_hat_t)
        x_hat_t_aligned = self._align_prediction(x_hat_t, x_target_t, align_w, resolved_mask)
        diff = x_hat_t_aligned - x_target_t
        coord_loss = self._reduce_loss(diff, sigma_t_vec, align_w, resolved_mask)
        lddt_loss = x_hat_t.new_zeros(())
        if self.smooth_lddt_loss_weight > 0.0:
            feats = student_trunk["batch"]
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            is_nucleotide = (
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
            ).to(x_hat_t_aligned)
            lddt_loss = smooth_lddt_loss(
                x_hat_t_aligned,
                x_target_t.detach(),
                is_nucleotide,
                feats["atom_resolved_mask"].to(x_hat_t_aligned),
                multiplicity=multiplicity,
            )
        loss = coord_loss + self.smooth_lddt_loss_weight * lddt_loss

        with torch.no_grad():
            ratio_ref = sigma_ref_vec.to(sigma_t_vec) / sigma_t_vec.clamp(min=1e-12)
            gain = 1.0 / (1.0 - ratio_ref).clamp(min=1e-6)
            self.last_metrics = {
                "mscd/loss": loss.detach(),
                "mscd/coord_loss": coord_loss.detach(),
                "mscd/smooth_lddt_loss": lddt_loss.detach(),
                "mscd/smooth_lddt_loss_weight": torch.as_tensor(float(self.smooth_lddt_loss_weight), device=device),
                "mscd/sigma_t_mean": sigma_t_vec.float().mean(),
                "mscd/sigma_s_mean": sigma_s_eff.float().mean(),
                "mscd/sigma_bdry_mean": sigma_bdry_vec.float().mean(),
                "mscd/T_edges": torch.as_tensor(float(teacher_sigmas.shape[0] - 1), device=device),
                "mscd/frac_terminal": is_terminal.float().mean(),
                "mscd/frac_boundary": is_boundary_snap.float().mean(),
                "mscd/frac_general": general_mask.float().mean(),
                "mscd/terminal_teacher_hop": torch.as_tensor(float(self.terminal_teacher_hop), device=device),
                "mscd/teacher_step_scale": torch.as_tensor(float(self.teacher_step_scale), device=device),
                "mscd/sync_sigmas_across_multiplicity": torch.as_tensor(float(repeat_edges), device=device),
                "mscd/gain_mean": gain.float().mean(),
                "mscd/gain_max": gain.float().max(),
                "mscd/l2_error": torch.sqrt(diff.detach().float().square().sum(dim=(-1, -2)).clamp(min=1e-12)).mean(),
                "mscd/weight_mean": self._weight(sigma_t_vec).detach().mean(),
            }
        return loss

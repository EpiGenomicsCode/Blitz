from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch


@torch.no_grad()
def make_karras_sigmas(
    num_nodes: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    round_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct a descending Karras grid in absolute EDM sigma space."""
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    step_indices = torch.arange(num_nodes, dtype=torch.float64, device=device)
    frac = step_indices / max(num_nodes - 1, 1)
    if not math.isfinite(float(rho)):
        # rho -> infinity limit of the Karras family: exact geometric spacing
        # between the SAME positive endpoints,
        #     sigma_i = sigma_max * (sigma_min/sigma_max) ** (i/(N-1)).
        # Evaluated directly rather than approximated with a large finite rho.
        #
        # 2026-09-14: without this branch rho=inf gave inv_rho = 0.0, so every
        # term became x**0 = 1 and the whole grid collapsed to 1.0 -- silently,
        # with no error.  Mirrors the AF3 fix in f07bab46 so the two codebases
        # produce identical grids for the same (sigma_min, sigma_max, N).
        sigmas = float(sigma_max) * (
            float(sigma_min) / float(sigma_max)
        ) ** frac
    else:
        inv_rho = 1.0 / float(rho)
        sigmas = (
            float(sigma_max) ** inv_rho
            + frac
            * (float(sigma_min) ** inv_rho - float(sigma_max) ** inv_rho)
        ) ** float(rho)
    if round_fn is not None:
        sigmas = round_fn(sigmas)
    return sigmas


def time_to_sigma(
    t: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    sigma_data: float,
) -> torch.Tensor:
    """Map t in [0, 1] to Boltz absolute EDM sigma."""
    t64 = t.to(torch.float64)
    if not math.isfinite(float(rho)):
        # Same rho -> infinity limit as make_karras_sigmas, in t-space:
        #   ((1-t)*smin^u + t*smax^u)^(1/u)  ->  smin^(1-t) * smax^t  as u -> 0.
        # Without this the expression degenerates to 1.0 for every t.
        sigma_rel = torch.exp(
            (1.0 - t64) * math.log(float(sigma_min))
            + t64 * math.log(float(sigma_max))
        )
    else:
        inv_rho = 1.0 / float(rho)
        sigma_rel = (
            (1.0 - t64) * (float(sigma_min) ** inv_rho)
            + t64 * (float(sigma_max) ** inv_rho)
        ) ** float(rho)
    return sigma_rel * float(sigma_data)


def filter_teacher_edges_by_sigma(
    student_sigmas: torch.Tensor,
    teacher_sigmas: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, int]:
    """Remove teacher edge starts that coincide with student interior boundaries."""
    if student_sigmas.ndim != 1 or teacher_sigmas.ndim != 1:
        raise ValueError("student_sigmas and teacher_sigmas must be 1D")
    num_edges = len(teacher_sigmas) - 1
    if num_edges < 1:
        raise ValueError("teacher_sigmas must include at least one edge and terminal zero")

    terminal_k = None
    for k in range(num_edges - 1, -1, -1):
        if teacher_sigmas[k] > 0 and teacher_sigmas[k + 1] == 0:
            terminal_k = k
            break
    if terminal_k is None:
        terminal_k = num_edges - 1

    student_interior = student_sigmas[1:-1]
    kept: list[int] = []
    for k in range(num_edges):
        if k == 0 or k == terminal_k:
            kept.append(k)
            continue
        sigma_k = teacher_sigmas[k]
        match = False
        for s in student_interior:
            if torch.isclose(
                sigma_k,
                s,
                rtol=eps,
                atol=eps * max(1.0, float(abs(sigma_k)), float(abs(s))),
            ):
                match = True
                break
        if not match:
            kept.append(k)

    kept_idx = torch.tensor(kept, dtype=torch.long, device=teacher_sigmas.device)
    teacher_sigmas_cd = torch.cat([teacher_sigmas[kept_idx], teacher_sigmas[-1:].clone()], dim=0)
    return teacher_sigmas_cd, len(kept) - 1


def partition_edges_by_sigma(
    student_sigmas: torch.Tensor,
    teacher_sigmas: torch.Tensor,
) -> torch.Tensor:
    """Assign teacher edges to student sigma segments."""
    if student_sigmas.ndim != 1 or teacher_sigmas.ndim != 1:
        raise ValueError("student_sigmas and teacher_sigmas must be 1D")
    if not torch.all(student_sigmas[:-1] >= student_sigmas[1:]):
        raise ValueError("student_sigmas must be descending")
    if not torch.all(teacher_sigmas[:-1] >= teacher_sigmas[1:]):
        raise ValueError("teacher_sigmas must be descending")

    num_segments = len(student_sigmas) - 1
    bounds: list[tuple[int, int]] = []
    for j in range(num_segments):
        upper = student_sigmas[j]
        lower = student_sigmas[j + 1]
        mask = (teacher_sigmas[:-1] <= upper) & (teacher_sigmas[:-1] > lower)
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            diffs = (teacher_sigmas[:-1] - lower).abs()
            k_near = int(torch.argmin(diffs).item())
            bounds.append((k_near, k_near))
        else:
            bounds.append((int(idx.min().item()), int(idx.max().item())))
    return torch.tensor(bounds, dtype=torch.long, device=student_sigmas.device)


def compute_importance_weights(
    teacher_sigmas: torch.Tensor,
    rho: float,
    mode: str = "vp",
    P_mean: float = -1.2,
    # 1.5, NOT EDM's image-domain 1.2. Boltz-2 trains on
    # sigma_data * exp(-1.2 + 1.5 * N(0,1)) (diffusionv2.py:188-189 and every
    # scripts/train/configs/*.yaml), which is AF3's spec verbatim (af3.md:906,
    # sigma_data=16 per af3.md:904). Matching the teacher's own training
    # distribution is the whole point of mode='edm'; 1.2 would silently narrow it.
    P_std: float = 1.5,
    terminal_anchor: bool = True,
    sigma_data: float = 1.0,
    segment_top_idx: torch.Tensor | None = None,
    segment_top_mass: float = 0.0,
) -> torch.Tensor:
    """Compute teacher-edge sampling probabilities.

    For ``mode='edm'`` the lognormal is centered in *relative* sigma space
    ``log(sigma / sigma_data)`` to match Boltz's native
    ``AtomDiffusion.noise_distribution`` (``sigma = sigma_data * exp(P_mean + P_std * N)``).
    With ``sigma_data=16`` and Boltz defaults ``P_mean=-1.2, P_std=1.5`` this places the
    mode at ``sigma ≈ 4.8`` in absolute Boltz sigma space (vs. ``sigma ≈ 0.3`` if the
    image-domain center were used unscaled).

    ``segment_top_mass`` (default 0 = off) reallocates that fraction of the total
    probability onto the ``segment_top_idx`` edges, keeping the same relative
    lognormal shape among them. Those edges are the *only* sigmas the student is
    queried at during inference (``diffusionv2.py:sample`` walks the student grid),
    so the default distribution spends ~1-1/S_edges of its mass on sigmas that only
    serve as bootstrap scaffolding.
    """
    sigmas = teacher_sigmas[:-1].float()
    num_edges = len(sigmas)
    if mode == "uniform":
        weights = torch.ones(num_edges, device=sigmas.device, dtype=torch.float32)
    elif mode == "vp":
        exponent = 1.0 - 1.0 / float(rho)
        weights = (sigmas + 1e-10) ** exponent / (1.0 + sigmas.square())
    elif mode == "edm":
        log_sigma_data = math.log(max(float(sigma_data), 1e-10))
        log_sigmas_rel = torch.log(sigmas + 1e-10) - log_sigma_data
        log_prob = -0.5 * ((log_sigmas_rel - float(P_mean)) / float(P_std)).square()
        weights = (sigmas + 1e-10) ** (-1.0 / float(rho)) * torch.exp(log_prob)
    else:
        raise ValueError(f"Unknown sampling mode {mode!r}")

    weights = weights / weights.sum().clamp(min=1e-10)
    if terminal_anchor and num_edges > 1 and mode != "uniform":
        target_p = 1.0 / num_edges
        non_term = weights[:-1]
        weights[:-1] = non_term * (1.0 - target_p) / non_term.sum().clamp(min=1e-10)
        weights[-1] = target_p
    beta = float(segment_top_mass)
    if segment_top_idx is not None and beta > 0.0 and num_edges > 1:
        beta = min(beta, 1.0)
        idx = segment_top_idx.to(weights.device).reshape(-1).long()
        idx = idx[(idx >= 0) & (idx < num_edges)]
        if idx.numel() > 0:
            top = torch.zeros_like(weights)
            top[idx] = weights[idx]
            top_sum = top.sum()
            if float(top_sum) > 0.0:
                weights = (1.0 - beta) * weights + beta * (top / top_sum)
    return weights


def sample_segment_and_teacher_pair(
    sigma_bounds: torch.Tensor,
    teacher_sigmas: torch.Tensor,
    student_sigmas: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    terminal_k: int | None = None,
    sampling_mode: str = "vp",
    rho: float = 7.0,
    P_mean: float = -1.2,
    # See compute_importance_weights: 1.5 is Boltz-2/AF3, 1.2 is EDM images.
    P_std: float = 1.5,
    terminal_anchor: bool = True,
    sigma_data: float = 1.0,
    segment_top_mass: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Sample one MSCD teacher edge per batch element."""
    sigma_bounds = sigma_bounds.to(device)
    teacher_sigmas = teacher_sigmas.to(device)
    student_sigmas = student_sigmas.to(device)
    num_segments = sigma_bounds.shape[0]
    num_edges = len(teacher_sigmas) - 1

    if sampling_mode in ("vp", "edm"):
        weights = compute_importance_weights(
            teacher_sigmas,
            rho,
            mode=sampling_mode,
            P_mean=P_mean,
            P_std=P_std,
            terminal_anchor=terminal_anchor,
            sigma_data=sigma_data,
            segment_top_idx=sigma_bounds[:, 0],
            segment_top_mass=segment_top_mass,
        ).to(device)
        k_t = torch.multinomial(weights, batch_size, replacement=True, generator=generator)
        k_starts = sigma_bounds[:, 0].contiguous()
        k_ends = sigma_bounds[:, 1]
        step_j = torch.searchsorted(k_starts, k_t, right=True) - 1
        step_j = step_j.clamp(min=0, max=num_segments - 1)
        k0 = k_starts[step_j]
        k1 = k_ends[step_j]
        seg_len = (k1 - k0 + 1).clamp(min=1)
        n_rel = (seg_len - (k_t - k0)).clamp(min=1)
    else:
        step_j = torch.randint(
            low=0,
            high=num_segments,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
            generator=generator,
        )
        k0 = sigma_bounds[step_j, 0]
        k1 = sigma_bounds[step_j, 1]
        seg_len = (k1 - k0 + 1).clamp(min=1)
        u = torch.empty(batch_size, device=device, dtype=torch.float32)
        u.uniform_(0.0, 1.0, generator=generator)
        n_rel = torch.floor(u * seg_len.float() + 1.0).to(torch.long)
        n_rel = torch.minimum(n_rel, seg_len)
        k_t = k1 - (n_rel - 1)

    k_s = (k_t + 1).clamp(max=num_edges)
    sigma_t = teacher_sigmas[k_t]
    sigma_s = teacher_sigmas[k_s]
    sigma_bdry = student_sigmas[step_j + 1]

    if terminal_k is None:
        terminal_k = num_edges - 1
    is_terminal = k_t == int(terminal_k)
    is_boundary_snap = (~is_terminal) & (n_rel == 1) & (step_j < (num_segments - 1))

    return {
        "step_j": step_j,
        "k_t": k_t,
        "k_s": k_s,
        "sigma_t": sigma_t,
        "sigma_s": sigma_s,
        "sigma_bdry": sigma_bdry,
        "n_rel": n_rel,
        "is_terminal": is_terminal,
        "is_boundary_snap": is_boundary_snap,
    }


def _expand_sigma_to_bld(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    s = torch.as_tensor(sigma, device=like.device, dtype=like.dtype)
    if s.ndim == 0:
        return s.reshape(1, 1, 1)
    if s.ndim == 1:
        return s.reshape(-1, 1, 1)
    return s


def ddim_step_edm(
    x_t: torch.Tensor,
    x_pred_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
) -> torch.Tensor:
    """EDM-space deterministic DDIM/Euler step for atom coordinates."""
    if x_t.shape != x_pred_t.shape:
        raise ValueError("x_t and x_pred_t must have the same shape")
    out_dtype = x_t.dtype
    x_t64 = x_t.to(torch.float64)
    x_pred64 = x_pred_t.to(torch.float64)
    sigma_t_b = _expand_sigma_to_bld(sigma_t, x_t64)
    sigma_s_b = _expand_sigma_to_bld(sigma_s, x_t64)
    if torch.any(sigma_t_b == 0):
        raise ValueError("ddim_step_edm received sigma_t == 0")
    x_s = x_pred64 + (sigma_s_b / sigma_t_b) * (x_t64 - x_pred64)
    return x_s.to(out_dtype)


def inv_ddim_edm(
    x_ref: torch.Tensor,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_ref: torch.Tensor,
) -> torch.Tensor:
    """Backsolve the denoised prediction whose DDIM step reaches x_ref."""
    if x_ref.shape != x_t.shape:
        raise ValueError("x_ref and x_t must have the same shape")
    out_dtype = x_t.dtype
    x_ref64 = x_ref.to(torch.float64)
    x_t64 = x_t.to(torch.float64)
    sigma_t_b = _expand_sigma_to_bld(sigma_t, x_t64)
    sigma_ref_b = _expand_sigma_to_bld(sigma_ref, x_t64)
    if torch.any(sigma_t_b == 0):
        raise ValueError("inv_ddim_edm received sigma_t == 0")
    denom = sigma_t_b - sigma_ref_b
    if torch.any(denom.abs() < 1e-12):
        raise ValueError("inv_ddim_edm denominator is near zero; sigma_ref ~= sigma_t")
    x_star = (x_ref64 * sigma_t_b - x_t64 * sigma_ref_b) / denom
    return x_star.to(out_dtype)


def sample_conditional_posterior_boltz(
    z_t: torch.Tensor,
    x_pred: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
) -> torch.Tensor:
    """Sample z_s ~ q(z_s | z_t, x_pred) for Boltz atom-coordinate tensors."""
    out_dtype = z_t.dtype
    z_t64 = z_t.to(torch.float64)
    x64 = x_pred.to(torch.float64)
    st = _expand_sigma_to_bld(sigma_t, z_t64)
    ss = _expand_sigma_to_bld(sigma_s, z_t64)
    st_sq = st.square()
    ss_sq = ss.square()
    diff_sq = torch.clamp(st_sq - ss_sq, min=0.0)
    mu = diff_sq / st_sq.clamp(min=1e-20) * x64 + ss_sq / st_sq.clamp(min=1e-20) * z_t64
    lam = torch.sqrt(torch.clamp(ss_sq * diff_sq / st_sq.clamp(min=1e-20), min=0.0))
    return (mu + lam * torch.randn_like(z_t64)).to(out_dtype)


def _gather_diffusion_conditioning(
    dc: dict[str, Any],
    idx: torch.Tensor,
    base_b: int,
) -> dict[str, Any]:
    """Subset Boltz-2 diffusion_conditioning tensors by batch index.

    ``to_keys`` is a functools.partial over a batch-independent indexing matrix
    (see encodersv2.AtomEncoder) and is shared as-is.
    """
    out: dict[str, Any] = {}
    for k, v in dc.items():
        if torch.is_tensor(v) and v.ndim > 0 and int(v.shape[0]) == base_b:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def gather_trunk_for_particles(
    trunk: dict[str, Any],
    idx: torch.Tensor,
    *,
    multiplicity: int,
) -> dict[str, Any]:
    """Subset trunk + feats for a subset of diffusion particles.

    After target augmentation, coordinates are laid out as ``[B0 * multiplicity, ...]``
    while trunk tensors still have batch ``B0``. The score model applies
    ``repeat_interleave(multiplicity)`` on trunk rows; masked calls (e.g. teacher Heun
    only on non-terminal particles) must pass a trunk whose batch dim matches ``x``
    and use ``multiplicity=1``.

    Particle ``i`` uses trunk row ``i // multiplicity``.

    Boltz-2 adaptation: also subsets ``diffusion_conditioning`` (INVENTORY_A / GEOGRAPHY_F §3).
    """
    idx = idx.long().reshape(-1)
    mult = int(multiplicity)
    if mult < 1:
        raise ValueError("multiplicity must be >= 1")
    base_b = int(trunk["s_inputs"].shape[0])
    trunk_line = idx // mult
    if base_b > 0:
        trunk_line = trunk_line.clamp(min=0, max=base_b - 1)
    feats_in = trunk["batch"]
    feats_out: dict[str, Any] = {}
    for k, v in feats_in.items():
        if torch.is_tensor(v) and v.ndim > 0 and int(v.shape[0]) == base_b:
            feats_out[k] = v[trunk_line]
        else:
            feats_out[k] = v
    result = {
        "s_inputs": trunk["s_inputs"][trunk_line],
        "s": trunk["s"][trunk_line],
        "z": trunk["z"][trunk_line],
        "relative_position_encoding": trunk["relative_position_encoding"][trunk_line],
        "batch": feats_out,
    }
    if "diffusion_conditioning" in trunk:
        result["diffusion_conditioning"] = _gather_diffusion_conditioning(
            trunk["diffusion_conditioning"], trunk_line, base_b
        )
    return result


def gather_trunk_for_base_indices(
    trunk: dict[str, Any],
    base_idx: torch.Tensor,
) -> dict[str, Any]:
    """Subset trunk + feats by base structure while preserving Boltz multiplicity semantics."""
    base_idx = base_idx.long().reshape(-1)
    base_b = int(trunk["s_inputs"].shape[0])
    if base_b > 0:
        base_idx = base_idx.clamp(min=0, max=base_b - 1)
    feats_in = trunk["batch"]
    feats_out: dict[str, Any] = {}
    for k, v in feats_in.items():
        if torch.is_tensor(v) and v.ndim > 0 and int(v.shape[0]) == base_b:
            feats_out[k] = v[base_idx]
        else:
            feats_out[k] = v
    result = {
        "s_inputs": trunk["s_inputs"][base_idx],
        "s": trunk["s"][base_idx],
        "z": trunk["z"][base_idx],
        "relative_position_encoding": trunk["relative_position_encoding"][base_idx],
        "batch": feats_out,
    }
    if "diffusion_conditioning" in trunk:
        result["diffusion_conditioning"] = _gather_diffusion_conditioning(
            trunk["diffusion_conditioning"], base_idx, base_b
        )
    return result


def _run_structure(
    structure_module: torch.nn.Module,
    trunk: dict[str, Any],
    x: torch.Tensor,
    sigma: torch.Tensor,
    *,
    multiplicity: int,
    model_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run Boltz-2 ``AtomDiffusion.preconditioned_network_forward``.

    Boltz-2 returns denoised coords only (not a ``(denoised, token_a)`` tuple) and
    expects ``diffusion_conditioning`` in ``network_condition_kwargs`` rather than
    raw ``z_trunk`` / ``relative_position_encoding`` (GEOGRAPHY_F §3).
    ``model_cache`` is accepted for API compatibility with the Boltz-1 MSCD source
    but is unused on Boltz-2.
    """
    del model_cache  # Boltz-2 AtomDiffusion has no model_cache path
    sigma_in = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
    if "diffusion_conditioning" not in trunk:
        raise KeyError(
            "Boltz-2 MSCD trunk must include diffusion_conditioning "
            "(compute via return_trunk_features=True)"
        )
    denoised = structure_module.preconditioned_network_forward(
        x,
        sigma_in,
        network_condition_kwargs=dict(
            s_inputs=trunk["s_inputs"],
            s_trunk=trunk["s"],
            feats=trunk["batch"],
            multiplicity=multiplicity,
            diffusion_conditioning=trunk["diffusion_conditioning"],
        ),
    )
    return denoised


@torch.no_grad()
def heun_hop_protein(
    structure_module: torch.nn.Module,
    trunk: dict[str, Any],
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    *,
    multiplicity: int,
    step_scale: float = 1.0,
) -> torch.Tensor:
    """Deterministic Heun hop for Boltz atom-coordinate denoisers.

    ``step_scale`` matches Boltz ``AtomDiffusion`` Heun / SDE-Euler scaling (default 1.5
    in full inference); use 1.0 for the classical unscaled probability-flow step.

    Teacher targets use this Heun hop through ``loss_mscd.py``. Student
    validation sampling stays Euler-only via
      ``AtomDiffusion.sample`` (diffusionv2.py:295; no ``integrator=`` kwarg) with
      ``gamma_0=0`` / ``step_scale=1.0`` overrides
      (``_apply_val_sampling_overrides``).
    Do **not** silently unify teacher hop → Euler (or student val → Heun). The
    inv-DDIM segment target is defined on the teacher Heun trajectory; changing
    the hop integrator would retarget distillation without a measured reason.
    """
    if step_scale <= 0:
        raise ValueError(f"step_scale must be positive, got {step_scale}")
    out_dtype = x_t.dtype
    x64 = x_t.to(torch.float64)
    sigma_t_b = _expand_sigma_to_bld(sigma_t, x64)
    sigma_s_b = _expand_sigma_to_bld(sigma_s, x64)
    if torch.any(sigma_t_b == 0) or torch.any(sigma_s_b == 0):
        raise ValueError("heun_hop_protein received sigma_t or sigma_s == 0")

    denoised_t = _run_structure(
        structure_module,
        trunk,
        x64.float(),
        torch.as_tensor(sigma_t, device=x64.device),
        multiplicity=multiplicity,
    ).to(torch.float64)
    k1 = (x64 - denoised_t) / sigma_t_b
    delta = step_scale * (sigma_s_b - sigma_t_b)
    x_eul = x64 + delta * k1

    denoised_s = _run_structure(
        structure_module,
        trunk,
        x_eul.float(),
        torch.as_tensor(sigma_s, device=x64.device),
        multiplicity=multiplicity,
    ).to(torch.float64)
    k2 = (x_eul - denoised_s) / sigma_s_b
    x_s = x64 + 0.5 * delta * (k1 + k2)
    return x_s.to(out_dtype)

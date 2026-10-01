"""EDM2-style post-hoc EMA (pHEMA) for Boltz MSCD structure_module weights.

Adapted from the power-function EMA math in NVIDIA EDM2's
``training/phema.py``. Snapshots store only trainable ``structure_module`` tensors so
reconstruction stays lightweight; the CLI merges them into a Lightning ckpt.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def exp_to_std(exp: np.ndarray | float) -> np.ndarray:
    """Convert power-function exponent → relative std (EDM2 Eq. 123)."""
    exp = np.float64(exp)
    return np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))


def std_to_exp(std: np.ndarray | float) -> np.ndarray:
    """Convert relative std → power-function exponent (EDM2 Eq. 126 / Alg. 2)."""
    std_arr = np.asarray(std, dtype=np.float64)
    tmp = std_arr.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    return np.asarray(exp, dtype=np.float64).reshape(std_arr.shape)


def power_function_correlation(
    a_ofs: np.ndarray,
    a_std: np.ndarray,
    b_ofs: np.ndarray,
    b_std: np.ndarray,
) -> np.ndarray:
    """Inner products between EMA profiles (EDM2 Eq. 151 / Alg. 3)."""
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio**t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den


def solve_posthoc_coefficients(
    in_ofs: Sequence[float] | np.ndarray,
    in_std: Sequence[float] | np.ndarray,
    out_ofs: Sequence[float] | np.ndarray | float,
    out_std: Sequence[float] | np.ndarray | float,
) -> np.ndarray:
    """Solve linear combination coeffs for post-hoc EMA (EDM2 Alg. 3).

    Returns array of shape ``[len(in), len(out)]`` that sums to 1 along axis 0.
    """
    in_ofs, in_std = np.broadcast_arrays(np.asarray(in_ofs, dtype=np.float64), np.asarray(in_std, dtype=np.float64))
    out_ofs, out_std = np.broadcast_arrays(np.asarray(out_ofs, dtype=np.float64), np.asarray(out_std, dtype=np.float64))
    rv = lambda x: np.float64(x).reshape(-1, 1)  # noqa: E731
    cv = lambda x: np.float64(x).reshape(1, -1)  # noqa: E731
    a = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    b = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    x = np.linalg.solve(a, b)
    x = x / np.sum(x, axis=0, keepdims=True)
    return x


def power_function_beta(std: float, t_next: float, t_delta: float) -> float:
    """Beta for tracking a power-function EMA profile (EDM2 Eq. 127)."""
    exp = float(std_to_exp(np.asarray(std)))
    return float((1.0 - float(t_delta) / float(t_next)) ** (exp + 1))


class PowerFunctionEMA:
    """Track one or more power-function EMA shadows of trainable parameters."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        stds: Sequence[float] = (0.050, 0.100),
    ) -> None:
        self.stds = [float(s) for s in stds]
        if not self.stds:
            raise ValueError("pHEMA stds must be non-empty")
        if any(s <= 0.0 or s >= 0.289 for s in self.stds):
            raise ValueError(f"pHEMA stds must be in (0, 0.289); got {self.stds}")
        params = [p.detach().clone() for p in parameters if p.requires_grad]
        if not params:
            raise ValueError("PowerFunctionEMA requires at least one trainable parameter")
        self.shadows: list[list[torch.Tensor]] = [
            [p.clone() for p in params] for _ in self.stds
        ]
        self.num_updates = 0

    def to(self, device: torch.device) -> None:
        self.shadows = [[t.to(device) for t in shadow] for shadow in self.shadows]

    @torch.no_grad()
    def update(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        cur_nimg: int,
        batch_size: int,
    ) -> list[float]:
        params = [p for p in parameters if p.requires_grad]
        if len(params) != len(self.shadows[0]):
            raise RuntimeError(
                f"pHEMA parameter mismatch: {len(params)} live vs {len(self.shadows[0])} shadow"
            )
        t_next = max(float(cur_nimg), 1.0)
        t_delta = float(batch_size)
        betas: list[float] = []
        for std, shadow in zip(self.stds, self.shadows):
            beta = power_function_beta(std=std, t_next=t_next, t_delta=t_delta)
            betas.append(beta)
            for s, p in zip(shadow, params):
                # EDM2: p_ema.lerp_(p_net, 1 - beta)  →  (1-beta)*net + beta*ema
                s.copy_(p.detach().lerp(s, beta))
        self.num_updates += 1
        return betas

    def state_dicts(self) -> list[dict[str, torch.Tensor]]:
        """Return flat ``structure_module.*`` state dicts, one per tracked std."""
        out: list[dict[str, torch.Tensor]] = []
        for shadow in self.shadows:
            # Caller supplies name mapping when saving; here we only return tensors
            # indexed by ordinal — distill_mscd fills names.
            out.append({str(i): t.detach().cpu() for i, t in enumerate(shadow)})
        return out

    def named_state_dicts(self, names: Sequence[str]) -> list[dict[str, torch.Tensor]]:
        if len(names) != len(self.shadows[0]):
            raise RuntimeError(f"name count {len(names)} != shadow count {len(self.shadows[0])}")
        return [
            {n: t.detach().cpu().clone() for n, t in zip(names, shadow)}
            for shadow in self.shadows
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "stds": list(self.stds),
            "num_updates": self.num_updates,
            "shadows": [[t.detach().cpu() for t in shadow] for shadow in self.shadows],
        }

    def load_state_dict(self, state: dict[str, Any], device: torch.device) -> None:
        self.stds = [float(s) for s in state["stds"]]
        self.num_updates = int(state.get("num_updates", 0))
        self.shadows = [[t.to(device) for t in shadow] for shadow in state["shadows"]]


_SNAP_RE = re.compile(r"^(?P<prefix>.+)-(?P<step>\d+)-(?P<std>\d+\.\d+)\.pt$")


def list_phema_snapshots(
    run_dir: str | Path,
    *,
    prefix: str | None = "phema",
    in_std: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """List ``{prefix}-{step:07d}-{std:.3f}.pt`` snapshots in ``run_dir``."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"pHEMA run directory does not exist: {run_dir}")
    std_filter = {float(s) for s in in_std} if in_std is not None else None
    snaps: list[dict[str, Any]] = []
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        m = _SNAP_RE.fullmatch(path.name)
        if not m:
            continue
        if prefix is not None and m.group("prefix") != prefix:
            continue
        std = float(m.group("std"))
        if std_filter is not None and std not in std_filter:
            continue
        snaps.append(
            {
                "path": path,
                "step": int(m.group("step")),
                "std": std,
                "nimg": int(m.group("step")),  # Boltz: ofs = optimizer step index
            }
        )
    snaps.sort(key=lambda s: (s["step"], s["std"]))
    return snaps


def reconstruct_structure_module_state(
    snaps: Sequence[dict[str, Any]],
    *,
    out_std: float,
    out_step: int | None = None,
) -> dict[str, torch.Tensor]:
    """Linear-combine stored pHEMA snapshots into a target-std state dict."""
    if not snaps:
        raise ValueError("No pHEMA snapshots provided")
    if out_step is None:
        out_step = max(int(s["step"]) for s in snaps)
    usable = [s for s in snaps if 0 < int(s["step"]) <= int(out_step)]
    if not usable:
        raise ValueError(f"No snapshots with 0 < step <= {out_step}")
    if not any(int(s["step"]) == int(out_step) for s in usable):
        raise ValueError(f"out_step={out_step} must match one of the input snapshot steps")

    in_ofs = np.asarray([float(s["nimg"]) for s in usable], dtype=np.float64)
    in_std = np.asarray([float(s["std"]) for s in usable], dtype=np.float64)
    coefs = solve_posthoc_coefficients(in_ofs, in_std, float(out_step), float(out_std))

    acc: dict[str, torch.Tensor] | None = None
    for i, snap in enumerate(usable):
        payload = torch.load(snap["path"], map_location="cpu", weights_only=False)
        state = payload["structure_module"] if isinstance(payload, dict) and "structure_module" in payload else payload
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected snapshot payload in {snap['path']}")
        if acc is None:
            acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state.items()}
        c = float(coefs[i, 0])
        for k, v in state.items():
            acc[k] = acc[k] + v.to(torch.float32) * c
    assert acc is not None
    return acc


def snapshot_filename(prefix: str, step: int, std: float) -> str:
    return f"{prefix}-{int(step):07d}-{float(std):.3f}.pt"

"""Deterministic K8 and K16 inference policies for Blitz."""

from __future__ import annotations

import copy
import types
from pathlib import Path
from typing import Any

import torch

from boltz.model.potentials.potentials import (
    ChiralAtomPotential,
    ConnectionsPotential,
    ContactPotentital,
    PlanarBondPotential,
    PoseBustersPotential,
    StereoBondPotential,
    SymmetricChainCOMPotential,
    TemplateReferencePotential,
    VDWOverlapPotential,
)
from boltz.model.potentials.schedules import (
    ExponentialInterpolation,
    ParameterSchedule,
    PiecewiseStepFunction,
)


BLITZ_K8 = "blitz_k8"
BLITZ_K16 = "blitz_k16"
BLITZ_PROFILES = (BLITZ_K8, BLITZ_K16)

K8_SIGMAS = (
    2559.9999999999936,
    662.0027853958692,
    163.27581055876576,
    38.27671800503477,
    8.49647074801396,
    1.7781930943808772,
    0.34920038458886976,
    0.06399999999999985,
)
K16_SIGMAS = (
    2560.000000000002,
    1437.1458036487986,
    799.999166819554,
    441.46564788973865,
    241.43980439522466,
    130.82961259276922,
    70.22044118874865,
    37.32076418185719,
    19.634951683927454,
    10.222511943229726,
    5.264812183345322,
    2.681309511025759,
    1.3498421968759302,
    0.6714527796589366,
    0.32988263548952346,
    0.15999999999999973,
)

K16_WEIGHTS = {
    "ChiralAtomPotential": (
        2.6508110523352313,
        1.683511525885819,
        1.6048114286127657,
        1.53095278098844,
        1.3312239936175618,
        1.130869460060961,
        1.26162179835882,
        1.286112272107651,
        1.2420427318734657,
        1.1236413795242304,
        1.0461530729455497,
        0.30849600158912677,
        0.07071067811865475,
        0.07692250455427756,
        0.19354931997947,
    ),
    "ConnectionsPotential": (0.15,) * 15,
    "PlanarBondPotential": (0.05,) * 15,
    "PoseBustersPotential": (
        0.6337593163321248,
        0.4747426475265231,
        0.3942350267692192,
        0.3837266267978207,
        0.37463144733515,
        0.3613011776600659,
        0.3174891107445967,
        0.2589309960123241,
        0.23558401788591446,
        0.21617612126589208,
        0.20141447946708468,
        0.08151102394641446,
        0.035355339059327376,
        0.01634602683881891,
        0.040258153195942684,
    ),
    "StereoBondPotential": (0.05,) * 15,
    "SymmetricChainCOMPotential": (0.5,) * 15,
    "VDWOverlapPotential": (0.675,) * 15,
}


class TerminalOverride(ParameterSchedule):
    """Use one raw guidance coefficient below the final-step threshold."""

    def __init__(
        self,
        ordinary: float | ParameterSchedule,
        terminal: float,
        threshold: float,
    ) -> None:
        self.ordinary = ordinary
        self.terminal = float(terminal)
        self.threshold = float(threshold)

    def compute(self, t: float) -> float:
        if float(t) < self.threshold:
            return self.terminal
        if isinstance(self.ordinary, ParameterSchedule):
            return float(self.ordinary.compute(t))
        return float(self.ordinary)


def _base_potentials(steering_args: dict[str, Any], boltz2: bool) -> list[Any]:
    physical = bool(steering_args["physical_guidance_update"])
    potentials: list[Any] = []
    if steering_args["fk_steering"] or physical:
        potentials.extend(
            [
                SymmetricChainCOMPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.5 if physical else 0.0,
                        "resampling_weight": 0.5,
                        "buffer": ExponentialInterpolation(
                            start=1.0, end=5.0, alpha=-2.0
                        ),
                    }
                ),
                VDWOverlapPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.5 if physical else 0.0,
                        "resampling_weight": PiecewiseStepFunction(
                            thresholds=[0.6], values=[0.01, 0.0]
                        ),
                        "buffer": 0.225,
                    }
                ),
                ConnectionsPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.15 if physical else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 2.0,
                    }
                ),
                PoseBustersPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05 if physical else 0.0,
                        "resampling_weight": 0.1,
                        "bond_buffer": 0.20,
                        "angle_buffer": 0.20,
                        "clash_buffer": 0.15,
                    }
                ),
                ChiralAtomPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.1 if physical else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                StereoBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05 if physical else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                PlanarBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05 if physical else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.26180,
                    }
                ),
            ]
        )
    contact = bool(steering_args["contact_guidance_update"])
    if boltz2 and (steering_args["fk_steering"] or contact):
        potentials.extend(
            [
                ContactPotentital(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": (
                            PiecewiseStepFunction(
                                thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0]
                            )
                            if contact
                            else 0.0
                        ),
                        "resampling_weight": 1.0,
                        "union_lambda": ExponentialInterpolation(
                            start=8.0, end=0.0, alpha=-2.0
                        ),
                    }
                ),
                TemplateReferencePotential(
                    parameters={
                        "guidance_interval": 2,
                        "guidance_weight": 0.1 if contact else 0.0,
                        "resampling_weight": 1.0,
                    }
                ),
            ]
        )
    return potentials


def _k16_schedule(values: tuple[float, ...]) -> PiecewiseStepFunction:
    if len(values) != 15:
        raise ValueError(f"K16 schedule requires 15 values, got {len(values)}")
    return PiecewiseStepFunction(
        thresholds=[index / 16.0 for index in range(2, 16)],
        values=[float(value) for value in reversed(values)],
    )


def get_blitz_potentials(
    steering_args: dict[str, Any], boltz2: bool = False
) -> list[Any]:
    """Build the potential schedule for a Blitz policy."""

    profile = str(steering_args.get("steering_schedule_profile"))
    if profile not in BLITZ_PROFILES:
        raise ValueError(f"unknown Blitz steering profile: {profile!r}")
    potentials = _base_potentials(steering_args, boltz2=boltz2)
    physical = bool(steering_args["physical_guidance_update"])

    if profile == BLITZ_K8:
        matched: set[str] = set()
        for potential in potentials:
            name = type(potential).__name__
            if name == "PoseBustersPotential":
                potential.parameters["guidance_weight"] = (
                    TerminalOverride(
                        potential.parameters["guidance_weight"], 0.1, 3.0 / 16.0
                    )
                    if physical
                    else 0.0
                )
                matched.add(name)
            elif name == "ChiralAtomPotential":
                potential.parameters["guidance_weight"] = (
                    TerminalOverride(
                        potential.parameters["guidance_weight"], 0.1, 3.0 / 16.0
                    )
                    if physical
                    else 0.0
                )
                matched.add(name)
        if matched != {"PoseBustersPotential", "ChiralAtomPotential"}:
            raise RuntimeError(f"K8 potential identity mismatch: {sorted(matched)}")
        return potentials

    expected = set(K16_WEIGHTS)
    matched = set()
    for potential in potentials:
        name = type(potential).__name__
        if name not in K16_WEIGHTS:
            continue
        potential.parameters["guidance_interval"] = 1
        potential.parameters["guidance_weight"] = (
            _k16_schedule(K16_WEIGHTS[name]) if physical else 0.0
        )
        if name == "PoseBustersPotential" and physical:
            potential.parameters["guidance_weight"] = TerminalOverride(
                potential.parameters["guidance_weight"], 0.135, 3.0 / 32.0
            )
        matched.add(name)
    if matched != expected:
        raise RuntimeError(f"K16 potential identity mismatch: {sorted(matched)}")
    return potentials


def profile_details(profile: str) -> dict[str, Any]:
    """Return the sampler settings for a Blitz policy."""

    if profile == BLITZ_K8:
        return {
            "profile": profile,
            "sampling_steps": 8,
            "rho": 40.0,
            "sigma_min_relative": 0.004,
            "sigmas": list(K8_SIGMAS),
            "gamma_0": 0.0,
            "step_scale": 1.5,
            "integrator": "euler",
            "exp_exact_step": True,
            "terminal_projection_scale": 0.1,
            "terminal_projection_num_gd_steps": 10,
        }
    if profile == BLITZ_K16:
        return {
            "profile": profile,
            "sampling_steps": 16,
            "rho": 40.0,
            "sigma_min_relative": 0.01,
            "sigmas": list(K16_SIGMAS),
            "gamma_0": 0.0,
            "step_scale": 1.5,
            "integrator": "euler",
            "exp_exact_step": True,
            "terminal_projection_scale": 2.0 / 27.0,
            "terminal_projection_num_gd_steps": 10,
        }
    raise ValueError(f"unknown Blitz profile: {profile!r}")


def standalone_blitz_profile(checkpoint_path: str | Path) -> str | None:
    """Return the profile embedded in a standalone Blitz checkpoint."""
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except Exception:  # noqa: BLE001 - Lightning metadata may use OmegaConf
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
        except Exception:  # noqa: BLE001 - not a readable Blitz checkpoint
            return None
    try:
        if checkpoint.get("blitz_format") != "blitz-standalone-v1":
            return None
        profile = checkpoint.get("blitz_profile")
        if profile not in {"blitz_k8", "blitz_k16"}:
            raise ValueError(f"invalid standalone Blitz profile {profile!r}")
        return str(profile)
    finally:
        del checkpoint


def configure_blitz_model(model: Any, profile: str) -> dict[str, Any]:
    """Apply a deterministic policy to a loaded Boltz-2 student model."""

    detail = profile_details(profile)
    structure = model.structure_module
    sigmas = tuple(float(value) for value in detail["sigmas"])

    def fixed_schedule(_self: Any, num_sampling_steps: int | None = None) -> torch.Tensor:
        if num_sampling_steps not in (None, len(sigmas)):
            raise ValueError(
                f"{profile} requires {len(sigmas)} sampling steps, "
                f"got {num_sampling_steps}"
            )
        device = next(_self.score_model.parameters()).device
        return torch.tensor((*sigmas, 0.0), dtype=torch.float32, device=device)

    structure.sample_schedule = types.MethodType(fixed_schedule, structure)
    structure.num_sampling_steps = len(sigmas)
    structure.rho = float(detail["rho"])
    structure.sigma_min = float(detail["sigma_min_relative"])
    structure.gamma_0 = 0.0
    structure.step_scale = 1.5
    structure.exp_exact_step = True
    structure.steering_terminal_clean_projection = True
    structure.steering_terminal_clean_projection_scale = float(
        detail["terminal_projection_scale"]
    )
    structure.steering_terminal_clean_projection_num_gd_steps = int(
        detail["terminal_projection_num_gd_steps"]
    )

    steering = copy.deepcopy(model.steering_args)
    steering.update(
        {
            "fk_steering": True,
            "num_particles": 1,
            "fk_lambda": 4.0,
            "fk_resampling_interval": 3,
            "physical_guidance_update": True,
            "contact_guidance_update": True,
            "num_gd_steps": 20,
            "steering_schedule_profile": profile,
        }
    )
    model.steering_args = steering
    predict_args = copy.deepcopy(model.predict_args)
    predict_args["sampling_steps"] = len(sigmas)
    predict_args["integrator"] = "euler"
    model.predict_args = predict_args
    return detail

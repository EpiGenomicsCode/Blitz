"""Structure validation metrics for Boltz-2 consistency distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MeanMetric

from boltz.data import const
from boltz.model.loss.validation import (
    compute_pae_mae,
    compute_pde_mae,
    compute_plddt_mae,
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
)

if TYPE_CHECKING:
    from boltz.model.models.boltz2 import Boltz2


@dataclass
class ValidationMetricRefs:
    """References to torchmetrics used during validation aggregation."""

    lddt: Any
    disto_lddt: Any
    complex_lddt: Any
    rmsd: MeanMetric
    best_rmsd: MeanMetric
    top1_lddt: Optional[Any] = None
    iplddt_top1_lddt: Optional[Any] = None
    ipde_top1_lddt: Optional[Any] = None
    pde_top1_lddt: Optional[Any] = None
    ptm_top1_lddt: Optional[Any] = None
    iptm_top1_lddt: Optional[Any] = None
    ligand_iptm_top1_lddt: Optional[Any] = None
    protein_iptm_top1_lddt: Optional[Any] = None
    avg_lddt: Optional[Any] = None
    plddt_mae: Optional[Any] = None
    pde_mae: Optional[Any] = None
    pae_mae: Optional[Any] = None


def validation_metric_refs_from_boltz(boltz: "Boltz2") -> ValidationMetricRefs:
    """Build refs from a Boltz module (metrics live on the module itself)."""
    if boltz.confidence_prediction:
        return ValidationMetricRefs(
            lddt=boltz.lddt,
            disto_lddt=boltz.disto_lddt,
            complex_lddt=boltz.complex_lddt,
            rmsd=boltz.rmsd,
            best_rmsd=boltz.best_rmsd,
            top1_lddt=boltz.top1_lddt,
            iplddt_top1_lddt=boltz.iplddt_top1_lddt,
            ipde_top1_lddt=boltz.ipde_top1_lddt,
            pde_top1_lddt=boltz.pde_top1_lddt,
            ptm_top1_lddt=boltz.ptm_top1_lddt,
            iptm_top1_lddt=boltz.iptm_top1_lddt,
            ligand_iptm_top1_lddt=boltz.ligand_iptm_top1_lddt,
            protein_iptm_top1_lddt=boltz.protein_iptm_top1_lddt,
            avg_lddt=boltz.avg_lddt,
            plddt_mae=boltz.plddt_mae,
            pde_mae=boltz.pde_mae,
            pae_mae=boltz.pae_mae,
        )
    return ValidationMetricRefs(
        lddt=boltz.lddt,
        disto_lddt=boltz.disto_lddt,
        complex_lddt=boltz.complex_lddt,
        rmsd=boltz.rmsd,
        best_rmsd=boltz.best_rmsd,
    )


def accumulate_structure_validation_batch(
    refs: ValidationMetricRefs,
    boltz: Any,
    batch: dict[str, Tensor],
    out: dict[str, Tensor],
    *,
    diffusion_samples: int,
    symmetry_correction: bool,
    confidence_prediction: bool,
) -> None:
    """Update ``refs`` metrics from one validation batch (post-forward)."""
    n_samples = diffusion_samples

    boundaries = torch.linspace(2, 22.0, 63)
    lower = torch.tensor([1.0])
    upper = torch.tensor([22.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(out["pdistogram"])

    preds = out["pdistogram"]
    pred_softmax = torch.softmax(preds, dim=-1)
    pred_softmax = pred_softmax.argmax(dim=-1)
    pred_softmax = torch.nn.functional.one_hot(
        pred_softmax, num_classes=preds.shape[-1]
    )
    pred_dist = (pred_softmax * mid_points).sum(dim=-1)
    true_center = batch["disto_center"]
    true_dists = torch.cdist(true_center, true_center)

    disto_lddt_dict, disto_total_dict = factored_token_lddt_dist_loss(
        feats=batch,
        true_d=true_dists,
        pred_d=pred_dist,
    )

    # Boltz-2 returns named coordinate tensors.
    true_ret = boltz.get_true_coordinates(
        batch=batch,
        out=out,
        diffusion_samples=n_samples,
        symmetry_correction=symmetry_correction,
    )
    if isinstance(true_ret, dict):
        true_coords = true_ret["true_coords"]
        true_coords_resolved_mask = true_ret["true_coords_resolved_mask"]
        rmsds = true_ret.get("rmsds", 0)
        best_rmsds = true_ret.get("best_rmsd_recall", true_ret.get("best_rmsds", 0))
        # Boltz2 may keep an ensemble/conformer axis at dim=1 (K=1 under current data).
        if true_coords.ndim == 4 and int(true_coords.shape[1]) == 1:
            true_coords = true_coords.squeeze(1)
        # Boltz2 reports zero-valued RMSD placeholders under symmetry correction;
        # lDDT remains the primary structure-validation metric here.
        # MeanMetric needs tensors — coerce scalars to zeros below.
        device = out["sample_atom_coords"].device
        if not torch.is_tensor(rmsds):
            rmsds = torch.zeros(n_samples, device=device)
        if not torch.is_tensor(best_rmsds):
            best_rmsds = torch.zeros(1, device=device)
    else:
        true_coords, rmsds, best_rmsds, true_coords_resolved_mask = true_ret

    all_lddt_dict, all_total_dict = factored_lddt_loss(
        feats=batch,
        atom_mask=true_coords_resolved_mask,
        true_atom_coords=true_coords,
        pred_atom_coords=out["sample_atom_coords"],
        multiplicity=n_samples,
    )

    best_lddt_dict, best_total_dict = {}, {}
    best_complex_lddt_dict, best_complex_total_dict = {}, {}
    bsz = true_coords.shape[0] // n_samples
    if n_samples > 1:
        complex_total = 0
        complex_lddt = 0
        for key in all_lddt_dict.keys():
            complex_lddt += all_lddt_dict[key] * all_total_dict[key]
            complex_total += all_total_dict[key]
        complex_lddt /= complex_total + 1e-7
        best_complex_idx = complex_lddt.reshape(-1, n_samples).argmax(dim=1)
        for key in all_lddt_dict:
            best_idx = all_lddt_dict[key].reshape(-1, n_samples).argmax(dim=1)
            best_lddt_dict[key] = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), best_idx
            ]
            best_total_dict[key] = all_total_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), best_idx
            ]
            best_complex_lddt_dict[key] = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), best_complex_idx
            ]
            best_complex_total_dict[key] = all_total_dict[key].reshape(
                -1, n_samples
            )[torch.arange(bsz), best_complex_idx]
    else:
        best_lddt_dict = all_lddt_dict
        best_total_dict = all_total_dict
        best_complex_lddt_dict = all_lddt_dict
        best_complex_total_dict = all_total_dict

    if confidence_prediction and n_samples > 1:
        assert refs.top1_lddt is not None
        mae_plddt_dict, total_mae_plddt_dict = compute_plddt_mae(
            pred_atom_coords=out["sample_atom_coords"],
            feats=batch,
            true_atom_coords=true_coords,
            pred_lddt=out["plddt"],
            true_coords_resolved_mask=true_coords_resolved_mask,
            multiplicity=n_samples,
        )
        mae_pde_dict, total_mae_pde_dict = compute_pde_mae(
            pred_atom_coords=out["sample_atom_coords"],
            feats=batch,
            true_atom_coords=true_coords,
            pred_pde=out["pde"],
            true_coords_resolved_mask=true_coords_resolved_mask,
            multiplicity=n_samples,
        )
        mae_pae_dict, total_mae_pae_dict = compute_pae_mae(
            pred_atom_coords=out["sample_atom_coords"],
            feats=batch,
            true_atom_coords=true_coords,
            pred_pae=out["pae"],
            true_coords_resolved_mask=true_coords_resolved_mask,
            multiplicity=n_samples,
        )

        plddt = out["complex_plddt"].reshape(-1, n_samples)
        top1_idx = plddt.argmax(dim=1)
        iplddt = out["complex_iplddt"].reshape(-1, n_samples)
        iplddt_top1_idx = iplddt.argmax(dim=1)
        pde = out["complex_pde"].reshape(-1, n_samples)
        pde_top1_idx = pde.argmin(dim=1)
        ipde = out["complex_ipde"].reshape(-1, n_samples)
        ipde_top1_idx = ipde.argmin(dim=1)
        ptm = out["ptm"].reshape(-1, n_samples)
        ptm_top1_idx = ptm.argmax(dim=1)
        iptm = out["iptm"].reshape(-1, n_samples)
        iptm_top1_idx = iptm.argmax(dim=1)
        ligand_iptm = out["ligand_iptm"].reshape(-1, n_samples)
        ligand_iptm_top1_idx = ligand_iptm.argmax(dim=1)
        protein_iptm = out["protein_iptm"].reshape(-1, n_samples)
        protein_iptm_top1_idx = protein_iptm.argmax(dim=1)

        for key in all_lddt_dict:
            top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[torch.arange(bsz), top1_idx]
            top1_total = all_total_dict[key].reshape(-1, n_samples)[torch.arange(bsz), top1_idx]
            iplddt_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), iplddt_top1_idx
            ]
            iplddt_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), iplddt_top1_idx
            ]
            pde_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), pde_top1_idx
            ]
            pde_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), pde_top1_idx
            ]
            ipde_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), ipde_top1_idx
            ]
            ipde_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), ipde_top1_idx
            ]
            ptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), ptm_top1_idx
            ]
            ptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), ptm_top1_idx
            ]
            iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), iptm_top1_idx
            ]
            iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), iptm_top1_idx
            ]
            ligand_iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), ligand_iptm_top1_idx
            ]
            ligand_iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), ligand_iptm_top1_idx
            ]
            protein_iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), protein_iptm_top1_idx
            ]
            protein_iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                torch.arange(bsz), protein_iptm_top1_idx
            ]

            refs.top1_lddt[key].update(top1_lddt, top1_total)
            refs.iplddt_top1_lddt[key].update(iplddt_top1_lddt, iplddt_top1_total)
            refs.pde_top1_lddt[key].update(pde_top1_lddt, pde_top1_total)
            refs.ipde_top1_lddt[key].update(ipde_top1_lddt, ipde_top1_total)
            refs.ptm_top1_lddt[key].update(ptm_top1_lddt, ptm_top1_total)
            refs.iptm_top1_lddt[key].update(iptm_top1_lddt, iptm_top1_total)
            refs.ligand_iptm_top1_lddt[key].update(
                ligand_iptm_top1_lddt, ligand_iptm_top1_total
            )
            refs.protein_iptm_top1_lddt[key].update(
                protein_iptm_top1_lddt, protein_iptm_top1_total
            )

            refs.avg_lddt[key].update(all_lddt_dict[key], all_total_dict[key])
            refs.pde_mae[key].update(mae_pde_dict[key], total_mae_pde_dict[key])
            refs.pae_mae[key].update(mae_pae_dict[key], total_mae_pae_dict[key])

        for key in mae_plddt_dict:
            refs.plddt_mae[key].update(
                mae_plddt_dict[key], total_mae_plddt_dict[key]
            )

    for m in const.out_types:
        if m == "ligand_protein":
            if torch.any(
                batch["pocket_feature"][:, :, const.pocket_contact_info["POCKET"]].bool()
            ):
                refs.lddt["pocket_ligand_protein"].update(
                    best_lddt_dict[m], best_total_dict[m]
                )
                refs.disto_lddt["pocket_ligand_protein"].update(
                    disto_lddt_dict[m], disto_total_dict[m]
                )
                refs.complex_lddt["pocket_ligand_protein"].update(
                    best_complex_lddt_dict[m], best_complex_total_dict[m]
                )
            else:
                refs.lddt["ligand_protein"].update(
                    best_lddt_dict[m], best_total_dict[m]
                )
                refs.disto_lddt["ligand_protein"].update(
                    disto_lddt_dict[m], disto_total_dict[m]
                )
                refs.complex_lddt["ligand_protein"].update(
                    best_complex_lddt_dict[m], best_complex_total_dict[m]
                )
        else:
            refs.lddt[m].update(best_lddt_dict[m], best_total_dict[m])
            refs.disto_lddt[m].update(disto_lddt_dict[m], disto_total_dict[m])
            refs.complex_lddt[m].update(
                best_complex_lddt_dict[m], best_complex_total_dict[m]
            )
    refs.rmsd.update(rmsds)
    refs.best_rmsd.update(best_rmsds)


def export_structure_validation_scalars(
    refs: ValidationMetricRefs,
    *,
    confidence_prediction: bool,
) -> dict[str, float]:
    """Compute epoch-level scalars from accumulated metrics (same math as logging)."""
    scalars: dict[str, float] = {}
    avg_lddt: dict[str, Any] = {}
    avg_disto_lddt: dict[str, Any] = {}
    avg_complex_lddt: dict[str, Any] = {}
    avg_top1_lddt = {}
    avg_iplddt_top1_lddt = {}
    avg_pde_top1_lddt = {}
    avg_ipde_top1_lddt = {}
    avg_ptm_top1_lddt = {}
    avg_iptm_top1_lddt = {}
    avg_ligand_iptm_top1_lddt = {}
    avg_protein_iptm_top1_lddt = {}
    avg_avg_lddt = {}
    avg_mae_plddt = {}
    avg_mae_pde = {}
    avg_mae_pae = {}

    def _store(name: str, val: float) -> None:
        scalars[name] = float(val)

    for m in const.out_types + ["pocket_ligand_protein"]:
        avg_lddt[m] = refs.lddt[m].compute()
        avg_lddt[m] = 0.0 if torch.isnan(avg_lddt[m]) else avg_lddt[m].item()
        refs.lddt[m].reset()
        _store(f"lddt_{m}", avg_lddt[m])

        avg_disto_lddt[m] = refs.disto_lddt[m].compute()
        avg_disto_lddt[m] = 0.0 if torch.isnan(avg_disto_lddt[m]) else avg_disto_lddt[m].item()
        refs.disto_lddt[m].reset()
        _store(f"disto_lddt_{m}", avg_disto_lddt[m])

        avg_complex_lddt[m] = refs.complex_lddt[m].compute()
        avg_complex_lddt[m] = (
            0.0 if torch.isnan(avg_complex_lddt[m]) else avg_complex_lddt[m].item()
        )
        refs.complex_lddt[m].reset()
        _store(f"complex_lddt_{m}", avg_complex_lddt[m])
        if confidence_prediction and refs.top1_lddt is not None:
            avg_top1_lddt[m] = refs.top1_lddt[m].compute()
            avg_top1_lddt[m] = (
                0.0 if torch.isnan(avg_top1_lddt[m]) else avg_top1_lddt[m].item()
            )
            refs.top1_lddt[m].reset()
            _store(f"top1_lddt_{m}", avg_top1_lddt[m])
            avg_iplddt_top1_lddt[m] = refs.iplddt_top1_lddt[m].compute()
            avg_iplddt_top1_lddt[m] = (
                0.0
                if torch.isnan(avg_iplddt_top1_lddt[m])
                else avg_iplddt_top1_lddt[m].item()
            )
            refs.iplddt_top1_lddt[m].reset()
            _store(f"iplddt_top1_lddt_{m}", avg_iplddt_top1_lddt[m])
            avg_pde_top1_lddt[m] = refs.pde_top1_lddt[m].compute()
            avg_pde_top1_lddt[m] = (
                0.0
                if torch.isnan(avg_pde_top1_lddt[m])
                else avg_pde_top1_lddt[m].item()
            )
            refs.pde_top1_lddt[m].reset()
            _store(f"pde_top1_lddt_{m}", avg_pde_top1_lddt[m])
            avg_ipde_top1_lddt[m] = refs.ipde_top1_lddt[m].compute()
            avg_ipde_top1_lddt[m] = (
                0.0
                if torch.isnan(avg_ipde_top1_lddt[m])
                else avg_ipde_top1_lddt[m].item()
            )
            refs.ipde_top1_lddt[m].reset()
            _store(f"ipde_top1_lddt_{m}", avg_ipde_top1_lddt[m])
            avg_ptm_top1_lddt[m] = refs.ptm_top1_lddt[m].compute()
            avg_ptm_top1_lddt[m] = (
                0.0
                if torch.isnan(avg_ptm_top1_lddt[m])
                else avg_ptm_top1_lddt[m].item()
            )
            refs.ptm_top1_lddt[m].reset()
            _store(f"ptm_top1_lddt_{m}", avg_ptm_top1_lddt[m])
            avg_iptm_top1_lddt[m] = refs.iptm_top1_lddt[m].compute()
            avg_iptm_top1_lddt[m] = (
                0.0
                if torch.isnan(avg_iptm_top1_lddt[m])
                else avg_iptm_top1_lddt[m].item()
            )
            refs.iptm_top1_lddt[m].reset()
            _store(f"iptm_top1_lddt_{m}", avg_iptm_top1_lddt[m])
            avg_ligand_iptm_top1_lddt[m] = refs.ligand_iptm_top1_lddt[m].compute()
            avg_ligand_iptm_top1_lddt[m] = (
                0.0
                if torch.isnan(avg_ligand_iptm_top1_lddt[m])
                else avg_ligand_iptm_top1_lddt[m].item()
            )
            refs.ligand_iptm_top1_lddt[m].reset()
            _store(f"ligand_iptm_top1_lddt_{m}", avg_ligand_iptm_top1_lddt[m])
            avg_protein_iptm_top1_lddt[m] = refs.protein_iptm_top1_lddt[m].compute()
            avg_protein_iptm_top1_lddt[m] = (
                0.0
                if torch.isnan(avg_protein_iptm_top1_lddt[m])
                else avg_protein_iptm_top1_lddt[m].item()
            )
            refs.protein_iptm_top1_lddt[m].reset()
            _store(f"protein_iptm_top1_lddt_{m}", avg_protein_iptm_top1_lddt[m])

            avg_avg_lddt[m] = refs.avg_lddt[m].compute()
            avg_avg_lddt[m] = (
                0.0 if torch.isnan(avg_avg_lddt[m]) else avg_avg_lddt[m].item()
            )
            refs.avg_lddt[m].reset()
            _store(f"avg_lddt_{m}", avg_avg_lddt[m])
            avg_mae_pde[m] = refs.pde_mae[m].compute().item()
            refs.pde_mae[m].reset()
            _store(f"MAE_pde_{m}", avg_mae_pde[m])
            avg_mae_pae[m] = refs.pae_mae[m].compute().item()
            refs.pae_mae[m].reset()
            _store(f"MAE_pae_{m}", avg_mae_pae[m])

    for m in const.out_single_types:
        if confidence_prediction and refs.plddt_mae is not None:
            avg_mae_plddt[m] = refs.plddt_mae[m].compute().item()
            refs.plddt_mae[m].reset()
            _store(f"MAE_plddt_{m}", avg_mae_plddt[m])

    overall_disto_lddt = sum(
        avg_disto_lddt[m] * w for (m, w) in const.out_types_weights.items()
    ) / sum(const.out_types_weights.values())
    _store("disto_lddt", overall_disto_lddt)

    overall_lddt = sum(
        avg_lddt[m] * w for (m, w) in const.out_types_weights.items()
    ) / sum(const.out_types_weights.values())
    _store("lddt", overall_lddt)

    overall_complex_lddt = sum(
        avg_complex_lddt[m] * w for (m, w) in const.out_types_weights.items()
    ) / sum(const.out_types_weights.values())
    _store("complex_lddt", overall_complex_lddt)

    if confidence_prediction and refs.top1_lddt is not None:
        overall_top1_lddt = sum(
            avg_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        _store("top1_lddt", overall_top1_lddt)

        overall_iplddt_top1_lddt = sum(
            avg_iplddt_top1_lddt[m] * w
            for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        _store("iplddt_top1_lddt", overall_iplddt_top1_lddt)

        overall_pde_top1_lddt = sum(
            avg_pde_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        _store("pde_top1_lddt", overall_pde_top1_lddt)

        overall_ipde_top1_lddt = sum(
            avg_ipde_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        _store("ipde_top1_lddt", overall_ipde_top1_lddt)

        overall_ptm_top1_lddt = sum(
            avg_ptm_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        _store("ptm_top1_lddt", overall_ptm_top1_lddt)

        overall_iptm_top1_lddt = sum(
            avg_iptm_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        _store("iptm_top1_lddt", overall_iptm_top1_lddt)

        overall_avg_lddt = sum(
            avg_avg_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        _store("avg_lddt", overall_avg_lddt)

    rmsd_v = refs.rmsd.compute()
    rmsd_v = 0.0 if torch.isnan(rmsd_v) else float(rmsd_v.item())
    refs.rmsd.reset()
    _store("rmsd", rmsd_v)

    best_v = refs.best_rmsd.compute()
    best_v = 0.0 if torch.isnan(best_v) else float(best_v.item())
    refs.best_rmsd.reset()
    _store("best_rmsd", best_v)

    return scalars


def log_structure_validation_epoch(
    refs: ValidationMetricRefs,
    pl_module: Any,
    *,
    prefix: str,
    confidence_prediction: bool,
) -> None:
    """Log aggregated validation metrics (typically at epoch end)."""
    scalars = export_structure_validation_scalars(refs, confidence_prediction=confidence_prediction)

    def p(name: str) -> str:
        return f"{prefix}/{name}" if prefix else name

    prog_keys = {
        "disto_lddt",
        "lddt",
        "complex_lddt",
        "top1_lddt",
        "iplddt_top1_lddt",
        "pde_top1_lddt",
        "ipde_top1_lddt",
        "ptm_top1_lddt",
        "iptm_top1_lddt",
        "avg_lddt",
        "rmsd",
        "best_rmsd",
    }
    for k, v in scalars.items():
        pl_module.log(
            p(k),
            v,
            prog_bar=k in prog_keys,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )


class StandaloneValidationMetrics(nn.Module):
    """Own copy of validation metrics (for Lightning modules that are not ``Boltz2``)."""

    def __init__(self, confidence_prediction: bool) -> None:
        super().__init__()
        self.confidence_prediction = confidence_prediction
        self.lddt = nn.ModuleDict()
        self.disto_lddt = nn.ModuleDict()
        self.complex_lddt = nn.ModuleDict()
        if confidence_prediction:
            self.top1_lddt = nn.ModuleDict()
            self.iplddt_top1_lddt = nn.ModuleDict()
            self.ipde_top1_lddt = nn.ModuleDict()
            self.pde_top1_lddt = nn.ModuleDict()
            self.ptm_top1_lddt = nn.ModuleDict()
            self.iptm_top1_lddt = nn.ModuleDict()
            self.ligand_iptm_top1_lddt = nn.ModuleDict()
            self.protein_iptm_top1_lddt = nn.ModuleDict()
            self.avg_lddt = nn.ModuleDict()
            self.plddt_mae = nn.ModuleDict()
            self.pde_mae = nn.ModuleDict()
            self.pae_mae = nn.ModuleDict()
        for m in const.out_types + ["pocket_ligand_protein"]:
            self.lddt[m] = MeanMetric()
            self.disto_lddt[m] = MeanMetric()
            self.complex_lddt[m] = MeanMetric()
            if confidence_prediction:
                self.top1_lddt[m] = MeanMetric()
                self.iplddt_top1_lddt[m] = MeanMetric()
                self.ipde_top1_lddt[m] = MeanMetric()
                self.pde_top1_lddt[m] = MeanMetric()
                self.ptm_top1_lddt[m] = MeanMetric()
                self.iptm_top1_lddt[m] = MeanMetric()
                self.ligand_iptm_top1_lddt[m] = MeanMetric()
                self.protein_iptm_top1_lddt[m] = MeanMetric()
                self.avg_lddt[m] = MeanMetric()
                self.pde_mae[m] = MeanMetric()
                self.pae_mae[m] = MeanMetric()
        for m in const.out_single_types:
            if confidence_prediction:
                self.plddt_mae[m] = MeanMetric()
        self.rmsd = MeanMetric()
        self.best_rmsd = MeanMetric()

    def reset_all(self) -> None:
        for mod in self.modules():
            if isinstance(mod, MeanMetric):
                mod.reset()

    def as_refs(self) -> ValidationMetricRefs:
        if self.confidence_prediction:
            return ValidationMetricRefs(
                lddt=self.lddt,
                disto_lddt=self.disto_lddt,
                complex_lddt=self.complex_lddt,
                rmsd=self.rmsd,
                best_rmsd=self.best_rmsd,
                top1_lddt=self.top1_lddt,
                iplddt_top1_lddt=self.iplddt_top1_lddt,
                ipde_top1_lddt=self.ipde_top1_lddt,
                pde_top1_lddt=self.pde_top1_lddt,
                ptm_top1_lddt=self.ptm_top1_lddt,
                iptm_top1_lddt=self.iptm_top1_lddt,
                ligand_iptm_top1_lddt=self.ligand_iptm_top1_lddt,
                protein_iptm_top1_lddt=self.protein_iptm_top1_lddt,
                avg_lddt=self.avg_lddt,
                plddt_mae=self.plddt_mae,
                pde_mae=self.pde_mae,
                pae_mae=self.pae_mae,
            )
        return ValidationMetricRefs(
            lddt=self.lddt,
            disto_lddt=self.disto_lddt,
            complex_lddt=self.complex_lddt,
            rmsd=self.rmsd,
            best_rmsd=self.best_rmsd,
        )

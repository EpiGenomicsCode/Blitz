from __future__ import annotations

from collections.abc import Iterable

import torch
from pytorch_lightning import Callback


class StructureModuleEMA:
    """EMA helper with EDM-style half-life/ramp-up parameterization."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        halflife_kimg: float = 500.0,
        rampup_ratio: float | None = 0.05,
    ) -> None:
        self.halflife_kimg = float(halflife_kimg)
        self.rampup_ratio = None if rampup_ratio is None else float(rampup_ratio)
        self.num_updates = 0
        self.shadow_params = [p.detach().clone() for p in parameters if p.requires_grad]
        self.collected_params: list[torch.Tensor] = []

    def _beta(self, *, batch_size: int, cur_nimg: int) -> float:
        halflife_nimg = self.halflife_kimg * 1000.0
        if self.rampup_ratio is not None:
            halflife_nimg = min(halflife_nimg, max(float(cur_nimg), 1.0) * self.rampup_ratio)
        return 0.5 ** (float(batch_size) / max(halflife_nimg, 1e-8))

    def update(self, parameters: Iterable[torch.nn.Parameter], *, batch_size: int, cur_nimg: int) -> float:
        params = [p for p in parameters if p.requires_grad]
        if len(params) != len(self.shadow_params):
            raise RuntimeError(
                f"EMA parameter mismatch: {len(params)} live vs {len(self.shadow_params)} shadow"
            )
        beta = self._beta(batch_size=batch_size, cur_nimg=cur_nimg)
        with torch.no_grad():
            for shadow, param in zip(self.shadow_params, params):
                shadow.copy_(param.detach().lerp(shadow, beta))
        self.num_updates += 1
        return beta

    def assert_update_cadence(self, *, global_step: int, context: str = "EMA") -> None:
        """Require one EMA update per optimizer step."""
        expected = max(int(global_step), 0)
        if int(self.num_updates) != expected:
            raise RuntimeError(
                f"[{context}] EMA update cadence mismatch: num_updates={self.num_updates} "
                f"!= global_step={expected}. EMA must update exactly once per optimizer "
                f"step (account for accumulate_grad_batches)."
            )

    def to(self, device: torch.device) -> None:
        self.shadow_params = [p.to(device) for p in self.shadow_params]

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        self.collected_params = [p.detach().clone() for p in parameters if p.requires_grad]

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        params = [p for p in parameters if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        params = [p for p in parameters if p.requires_grad]
        for collected, param in zip(self.collected_params, params):
            param.data.copy_(collected.data)
        self.collected_params = []

    def state_dict(self) -> dict[str, object]:
        return {
            "halflife_kimg": self.halflife_kimg,
            "rampup_ratio": self.rampup_ratio,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict[str, object], device: torch.device) -> None:
        self.halflife_kimg = float(state_dict["halflife_kimg"])
        rr = state_dict.get("rampup_ratio")
        self.rampup_ratio = None if rr is None else float(rr)
        self.num_updates = int(state_dict.get("num_updates", 0))
        self.shadow_params = [p.to(device) for p in state_dict["shadow_params"]]  # type: ignore[index]


class StructureModuleEMACallback(Callback):
    """Update structure-module EMA/pHEMA once per optimizer step.

    Must be registered *before* ``ModelCheckpoint`` so ``on_save_checkpoint``
    sees ``num_updates == global_step``. Lightning runs callback
    ``on_train_batch_end`` before the LightningModule hook
    (``training_epoch_loop.py``), and PL 2.4 gates accumulation via
    ``trainer.fit_loop._should_accumulate`` (not ``Trainer.should_accumulate``).
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        fit_loop = getattr(trainer, "fit_loop", None)
        should_acc = getattr(fit_loop, "_should_accumulate", None)
        if not callable(should_acc):
            raise RuntimeError(
                "[mscd-EMA] trainer.fit_loop._should_accumulate missing; "
                "cannot gate EMA on the optimizer-step boundary."
            )
        if should_acc():
            return
        if hasattr(pl_module, "update_ema"):
            pl_module.update_ema(batch)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        if hasattr(pl_module, "swap_in_ema"):
            pl_module.swap_in_ema()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if hasattr(pl_module, "restore_from_ema"):
            pl_module.restore_from_ema()

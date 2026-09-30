import os
import random
import string
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from typing_extensions import override

from boltz.data.module.training import BoltzTrainingDataModule, DataConfig
from boltz.data.module.training_mscd import Boltz2TrainingDataModule, DataConfigV2


class SameDirAtomicCheckpointIO(CheckpointIO):
    """CheckpointIO that never stages via /tmp.

    The temporary file is written next to its destination and replaced
    atomically, so an interrupted write cannot replace a good checkpoint.
    """

    @override
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: Any,
        storage_options: Optional[Any] = None,
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "`storage_options` is not supported by SameDirAtomicCheckpointIO"
            )
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
        try:
            torch.save(checkpoint, tmp)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            if not zipfile.is_zipfile(tmp):
                raise RuntimeError(
                    f"checkpoint failed zipfile validation before rename: {tmp}"
                )
            os.replace(tmp, dest)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise

    @override
    def load_checkpoint(
        self,
        path: Any,
        map_location: Optional[Callable] = lambda storage, loc: storage,
    ) -> Dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=False)

    @override
    def remove_checkpoint(self, path: Any) -> None:
        p = Path(path)
        if p.is_file():
            p.unlink()


@dataclass
class TrainConfig:
    """Train configuration.

    Attributes
    ----------
    data : DataConfig
        The data configuration.
    model : ModelConfig
        The model configuration.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    pretrained : Optional[str]
        The pretrained model.
    wandb : Optional[dict]
        The wandb configuration.
    disable_checkpoint : bool
        Disable checkpoint.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save top k checkpoints.
    validation_only : bool
        Run validation only.
    debug : bool
        Debug mode.
    strict_loading : bool
        Fail on mismatched checkpoint weights.
    load_confidence_from_trunk: Optional[bool]
        Load pre-trained confidence weights from trunk.
    v2: bool
        Use v2 model.
    checkpoint_monitor: Optional[str]
        Metric to monitor for ModelCheckpoint (default val/lddt).
    checkpoint_mode: Optional[str]
        max/min for checkpoint_monitor.
    checkpoint_every_n_train_steps: Optional[int]
        If set, save checkpoints every N train steps (for val-skipped short trains).
    """

    data: DataConfig
    model: LightningModule
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    save_top_k: Optional[int] = 1
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    load_confidence_from_trunk: Optional[bool] = False
    v2: bool = False
    checkpoint_monitor: Optional[str] = "val/lddt"
    checkpoint_mode: Optional[str] = "max"
    checkpoint_every_n_train_steps: Optional[int] = None


def train(raw_config: str, args: list[str]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.

    """
    # Load the configuration
    raw_config = omegaconf.OmegaConf.load(raw_config)

    # Apply input arguments
    args = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, args)

    # Instantiate the task
    cfg = hydra.utils.instantiate(raw_config)
    cfg = TrainConfig(**cfg)

    # Set matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Create trainer dict
    trainer = cfg.trainer
    if trainer is None:
        trainer = {}

    # Flip some arguments in debug mode
    devices = trainer.get("devices", 1)

    wandb = cfg.wandb
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer["devices"] = devices
        cfg.data.num_workers = 0
        if wandb:
            wandb = None


    model_module = cfg.model

    # BoltzMSCDistiller loads its teacher and student directly.
    is_mscd_distiller = False
    try:
        from boltz.consistency.distill_mscd import BoltzMSCDistiller

        is_mscd_distiller = isinstance(model_module, BoltzMSCDistiller)
    except ImportError:
        BoltzMSCDistiller = None  # type: ignore

    # Create objects
    if cfg.v2:
        data_config = DataConfigV2(**cfg.data)
        data_module = Boltz2TrainingDataModule(data_config)
    else:
        data_config = DataConfig(**cfg.data)
        data_module = BoltzTrainingDataModule(data_config)

    if cfg.pretrained and not cfg.resume and not is_mscd_distiller:
        # Load the pretrained weights into the confidence module
        if cfg.load_confidence_from_trunk:
            checkpoint = torch.load(cfg.pretrained, map_location="cpu")

            # Modify parameter names in the state_dict
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("structure_module") and not key.startswith(
                    "distogram_module"
                ):
                    new_key = "confidence_module." + key
                    new_state_dict[new_key] = value
            new_state_dict.update(checkpoint["state_dict"])

            # Update the checkpoint with the new state_dict
            checkpoint["state_dict"] = new_state_dict

            # Save the modified checkpoint
            random_string = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=10)
            )
            file_path = os.path.dirname(cfg.pretrained) + "/" + random_string + ".ckpt"
            print(
                f"Saving modified checkpoint to {file_path} created by broadcasting trunk of {cfg.pretrained} to confidence module."
            )
            torch.save(checkpoint, file_path)
        else:
            file_path = cfg.pretrained

        print(f"Loading model from {file_path}")
        # validators are excluded from save_hyperparameters / checkpoint hparams;
        # keep the yaml-instantiated ModuleList so setup() can map val groups.
        yaml_validators = getattr(model_module, "validators", None)
        yaml_num_val = getattr(model_module, "num_val_datasets", None)
        model_module = type(model_module).load_from_checkpoint(
            file_path, map_location="cpu", strict=False, **(model_module.hparams)
        )
        if yaml_validators is not None and getattr(
            model_module, "validate_structure", False
        ):
            model_module.validators = yaml_validators
            if yaml_num_val is not None:
                model_module.num_val_datasets = yaml_num_val

        if cfg.load_confidence_from_trunk:
            os.remove(file_path)

    # Create checkpoint callback
    callbacks = []
    dirpath = cfg.output
    # Update EMA before ModelCheckpoint saves at the end of a training batch.
    if is_mscd_distiller:
        from boltz.consistency.ema import StructureModuleEMACallback

        callbacks.append(StructureModuleEMACallback())
    if not cfg.disable_checkpoint:
        every_n_steps = cfg.checkpoint_every_n_train_steps
        if every_n_steps:
            mc = ModelCheckpoint(
                dirpath=str(dirpath),
                save_top_k=-1,
                save_last=True,
                every_n_train_steps=int(every_n_steps),
                filename="mscd-{step}",
            )
        else:
            monitor = cfg.checkpoint_monitor or "val/lddt"
            mode = cfg.checkpoint_mode or "max"
            if is_mscd_distiller and monitor == "val/lddt":
                monitor = "val/mscd/lddt"
            mc = ModelCheckpoint(
                dirpath=str(dirpath),
                monitor=monitor,
                save_top_k=cfg.save_top_k,
                save_last=True,
                mode=mode,
                every_n_epochs=1,
            )
        callbacks.append(mc)

    # Create wandb logger
    loggers = []
    if wandb:
        wdb_logger = WandbLogger(
            name=wandb["name"],
            group=wandb["name"],
            save_dir=cfg.output,
            project=wandb["project"],
            entity=wandb["entity"],
            log_model=False,
        )
        loggers.append(wdb_logger)
        # Save the config to wandb

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
            with Path.open(config_out, "w") as f:
                OmegaConf.save(raw_config, f)
            wdb_logger.experiment.save(str(config_out))

        save_config_to_wandb()

    # Persist step metrics locally even when Weights & Biases is disabled.
    try:
        from pytorch_lightning.loggers import CSVLogger

        loggers.append(
            CSVLogger(
                str(dirpath),
                name="lightning_logs",
                version=None,
                flush_logs_every_n_steps=1,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[csv-logger] WARNING: could not create CSVLogger ({exc!r}); "
            "continuing without on-disk metrics"
        )

    # Keep checkpoint writes on one filesystem so the final rename is atomic.
    ckpt_io = SameDirAtomicCheckpointIO()
    strategy: Any = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, listconfig.ListConfig)) and len(devices) > 1
    ):
        strategy = DDPStrategy(
            find_unused_parameters=cfg.find_unused_parameters,
            checkpoint_io=ckpt_io,
        )

    trainer_kwargs = dict(trainer)
    plugins = list(trainer_kwargs.pop("plugins", []) or [])
    if strategy == "auto":
        # Attach via plugins when strategy is the string "auto".
        plugins.append(ckpt_io)

    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        plugins=plugins or None,
        **trainer_kwargs,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    if cfg.validation_only:
        trainer.validate(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )
    else:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )


if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2:]
    train(arg1, arg2)

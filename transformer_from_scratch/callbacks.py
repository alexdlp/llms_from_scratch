from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol

import torch
import mlflow
from pathlib import Path
from .logger import logger


class CallbackPipeline(Protocol):
    """Pipeline attributes required by the callback system."""

    run_artifacts_dir: Path | None
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    device: torch.device
    stop_training: bool



class Callback(ABC):
    """
    Minimal callback interface for training pipelines.
    All methods are optional; override only what you need.
    """

    def on_fit_start(self, pipeline: CallbackPipeline) -> None:
        pass

    def on_fit_end(self, pipeline: CallbackPipeline) -> None:
        pass

    def on_epoch_end(self, pipeline: CallbackPipeline, epoch: int, logs: Dict[str, float]) -> None:
        pass



class MetricCallback(Callback, ABC):
    """
    Base class for metric-based callbacks.
    Provides:
    - monitor
    - mode ("min" / "max")
    - min_delta
    - patience
    - start_from_epoch
    - best
    - wait
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min", min_delta: float = 0.0, 
                 patience: int = 0, start_from_epoch: int = 0) -> None:

        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.start_from_epoch = start_from_epoch

        self.best: Optional[float] = None
        self.wait = 0


    def should_skip(self, epoch: int) -> bool:
        """
        Returns True if the callback should not evaluate yet.
        """
        return epoch < self.start_from_epoch


    def _is_improvement(self, metric: float) -> bool:
        """
        Checks whether the new metric is an improvement over `best`,
        considering mode and min_delta.
        """
        if self.best is None:
            return True

        if self.mode == "min":
            return metric < self.best - self.min_delta

        return metric > self.best + self.min_delta


    def update_best(self, metric: float) -> bool:
        """
        Updates internal state and returns True if improved.
        """
        improved = self._is_improvement(metric)

        if improved:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1

        return improved


    def exceeded_patience(self) -> bool:
        """
        Returns True if patience has been exceeded.
        """
        return self.wait >= self.patience



class ModelCheckpoint(MetricCallback):
    """
    Hybrid callback:
    - Saves LAST checkpoint (PyTorch format): usable to resume training exactly.
    - Saves BEST MLflow model (servible): mlflow models serve, reproducible, portable.

    Compatible with MLflow 2.2.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min", start_from_epoch: int = 0,
                 min_delta: float = 0.0, save_last: bool = True, save_best: bool = True) -> None:
        super().__init__(monitor=monitor, mode=mode, min_delta=min_delta,start_from_epoch=start_from_epoch)

      
        self.save_last = save_last
        self.save_best = save_best

        # Dirpath is set later (not here)
        self.dir_best: Optional[Path] = None
        self.dir_last: Optional[Path] = None


    def on_fit_start(self, pipeline: CallbackPipeline) -> None:
        if pipeline.run_artifacts_dir is None:
            raise RuntimeError("The MLflow run artifacts directory is not initialized.")

        run_dir = pipeline.run_artifacts_dir

        # DIRECTORIOS
        self.dir_best = run_dir / "best_model"
        self.dir_last = run_dir / "checkpoints"

        self.dir_best.mkdir(parents=True, exist_ok=True)
        self.dir_last.mkdir(parents=True, exist_ok=True)

        logger.info(f"[ModelCheckpoint] Using run_dir: {run_dir}")
        logger.info(f"[ModelCheckpoint] Best-model directory: {self.dir_best}")
        logger.info(f"[ModelCheckpoint] Last-checkpoint directory: {self.dir_last}")

        if not self.dir_best.exists():
            logger.warning(f"[ModelCheckpoint] WARNING: best-model directory does not exist after creation!")
        if not self.dir_last.exists():
            logger.warning(f"[ModelCheckpoint] WARNING: last-checkpoint directory does not exist after creation!")






    def on_epoch_end(self, pipeline: CallbackPipeline, epoch: int, logs: Dict[str, float]) -> None:

        if self.should_skip(epoch):
            return

        metric = logs.get(self.monitor)
        if metric is None:
            return
        

        # ----- 1) Save BEST model (MLflow) -----
        if self.save_best and self.update_best(metric):
            self._save_best_model(pipeline)

        # ----- 2) Save LAST checkpoint -----
        if self.save_last:
            self._save_last_checkpoint(pipeline, epoch, logs)



    # ------------------------------------------------------------
    # Save best model (MLflow SERVIBLE)
    # ------------------------------------------------------------
    def _save_best_model(self, pipeline: CallbackPipeline) -> None:
        """
        Logs BEST model in MLflow (servible).
        """
        
        if mlflow.active_run():
            mlflow.pytorch.log_model(
                pipeline.model,
                artifact_path="best_model"
            )


    # ------------------------------------------------------------
    # Save last checkpoint (PyTorch FULL CHECKPOINT)
    # ------------------------------------------------------------
    def _save_last_checkpoint(self, pipeline: CallbackPipeline, epoch: int, logs: Dict[str, float]) -> Path:

        assert self.dir_last is not None
        ckpt_path = self.dir_last / "last.ckpt"
        scheduler = getattr(pipeline, "scheduler", None)

        state = {
            "epoch": epoch,
            "model_state": pipeline.model.state_dict(),
            "optimizer_state": pipeline.optimizer.state_dict() if pipeline.optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics": logs,
            "device": str(pipeline.device),
        }

        torch.save(state, ckpt_path)

        return ckpt_path



class EarlyStopping(MetricCallback):
    def __init__(self, monitor: str = "val_loss", mode: str = "min", patience: int = 10, 
                 min_delta: float = 0.0, start_from_epoch: int = 0):
        super().__init__(
            monitor=monitor,
            mode=mode,
            min_delta=min_delta,
            patience=patience,
            start_from_epoch=start_from_epoch,
        )
        self.stopped = False

    def on_epoch_end(
        self,
        pipeline: CallbackPipeline,
        epoch: int,
        logs: Dict[str, float],
    ) -> None:

        if self.should_skip(epoch):
            return

        metric = logs.get(self.monitor)
        if metric is None:
            return

        self.update_best(metric)

        if self.exceeded_patience():
            self.stopped = True
            pipeline.stop_training = True
            logger.info(
                f"[EarlyStopping] Stopping at epoch {epoch+1}: "
                f"no improvement on {self.monitor}"
            )

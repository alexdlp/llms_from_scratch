from pathlib import Path
from typing import Dict, Optional

import mlflow
import torch

from ..logger import logger
from .base import CallbackPipeline, MetricCallback


class ModelCheckpoint(MetricCallback):
    """Save the best model and the latest resumable training checkpoint."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        start_from_epoch: int = 0,
        min_delta: float = 0.0,
        save_last: bool = True,
        save_best: bool = True,
    ) -> None:
        super().__init__(
            monitor=monitor,
            mode=mode,
            min_delta=min_delta,
            start_from_epoch=start_from_epoch,
        )

        self.save_last = save_last
        self.save_best = save_best
        self.dir_best: Optional[Path] = None
        self.dir_last: Optional[Path] = None

    def on_fit_start(self, pipeline: CallbackPipeline) -> None:
        if pipeline.run_artifacts_dir is None:
            raise RuntimeError(
                "The MLflow run artifacts directory is not initialized."
            )

        run_dir = pipeline.run_artifacts_dir
        self.dir_best = run_dir / "best_model"
        self.dir_last = run_dir / "checkpoints"

        self.dir_best.mkdir(parents=True, exist_ok=True)
        self.dir_last.mkdir(parents=True, exist_ok=True)

        logger.info(f"[ModelCheckpoint] Using run_dir: {run_dir}")
        logger.info(
            f"[ModelCheckpoint] Best-model directory: {self.dir_best}"
        )
        logger.info(
            f"[ModelCheckpoint] Last-checkpoint directory: {self.dir_last}"
        )

        if not self.dir_best.exists():
            logger.warning(
                "[ModelCheckpoint] WARNING: best-model directory does not "
                "exist after creation!"
            )
        if not self.dir_last.exists():
            logger.warning(
                "[ModelCheckpoint] WARNING: last-checkpoint directory does "
                "not exist after creation!"
            )

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

        if self.save_best and self.update_best(metric):
            self._save_best_model(pipeline)

        if self.save_last:
            self._save_last_checkpoint(pipeline, epoch, logs)

    def _save_best_model(self, pipeline: CallbackPipeline) -> None:
        """Log the best model in MLflow."""
        if mlflow.active_run():
            model = getattr(pipeline.model, "module", pipeline.model)
            mlflow.pytorch.log_model(model, artifact_path="best_model")

    def _save_last_checkpoint(
        self,
        pipeline: CallbackPipeline,
        epoch: int,
        logs: Dict[str, float],
    ) -> Path:
        """Save the latest model and optimizer state for resuming."""
        assert self.dir_last is not None

        checkpoint_path = self.dir_last / "last.ckpt"
        scheduler = getattr(pipeline, "scheduler", None)

        state = {
            "epoch": epoch,
            "model_state": pipeline.model.state_dict(),
            "optimizer_state": (
                pipeline.optimizer.state_dict() if pipeline.optimizer else None
            ),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics": logs,
            "device": str(pipeline.device),
        }

        torch.save(state, checkpoint_path)
        return checkpoint_path


class EarlyStopping(MetricCallback):
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        start_from_epoch: int = 0,
    ) -> None:
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
                f"[EarlyStopping] Stopping at epoch {epoch + 1}: "
                f"no improvement on {self.monitor}"
            )

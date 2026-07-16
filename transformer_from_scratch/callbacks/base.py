from abc import ABC
from pathlib import Path
from typing import Dict, Optional, Protocol

import torch


class CallbackPipeline(Protocol):
    """Pipeline attributes required by the callback system."""

    run_artifacts_dir: Path | None
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    device: torch.device
    stop_training: bool


class Callback(ABC):
    """Minimal callback interface for training pipelines."""

    def on_fit_start(self, pipeline: CallbackPipeline) -> None:
        pass

    def on_fit_end(self, pipeline: CallbackPipeline) -> None:
        pass

    def on_epoch_end(
        self,
        pipeline: CallbackPipeline,
        epoch: int,
        logs: Dict[str, float],
    ) -> None:
        pass


class MetricCallback(Callback, ABC):
    """Base callback for tracking whether a monitored metric improves."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        patience: int = 0,
        start_from_epoch: int = 0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.start_from_epoch = start_from_epoch

        self.best: Optional[float] = None
        self.wait = 0

    def should_skip(self, epoch: int) -> bool:
        """Return whether metric evaluation should be skipped this epoch."""
        return epoch < self.start_from_epoch

    def _is_improvement(self, metric: float) -> bool:
        """Return whether a metric improves on the current best value."""
        if self.best is None:
            return True

        if self.mode == "min":
            return metric < self.best - self.min_delta

        return metric > self.best + self.min_delta

    def update_best(self, metric: float) -> bool:
        """Update the tracked metric state and return whether it improved."""
        improved = self._is_improvement(metric)

        if improved:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1

        return improved

    def exceeded_patience(self) -> bool:
        """Return whether the number of unimproved epochs reached patience."""
        return self.wait >= self.patience

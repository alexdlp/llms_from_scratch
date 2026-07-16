from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast, Dict, Optional, Protocol

import torch
import mlflow
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from .dataset.bilingual_common import EOS_TOKEN, SOS_TOKEN
from .decoding import GreedyDecodingModel, greedy_decode
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
            model = getattr(pipeline.model, "module", pipeline.model)
            mlflow.pytorch.log_model(
                model,
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


class TranslationExamplesCallback(Callback):
    """Generate and log a small set of validation translations each epoch.

    Args:
        validation_dataloader: Loader that provides prepared bilingual samples.
        target_tokenizer: Tokenizer used to decode the generated target IDs.
        maximum_length: Maximum generated sequence length, including ``[SOS]``.
        num_examples: Number of translations logged after each epoch.
    """

    def __init__(
        self,
        validation_dataloader: DataLoader,
        target_tokenizer: Tokenizer,
        maximum_length: int,
        num_examples: int = 2,
    ) -> None:
        if num_examples < 1:
            raise ValueError("num_examples must be at least 1.")

        self.validation_dataloader = validation_dataloader
        self.target_tokenizer = target_tokenizer
        self.maximum_length = maximum_length
        self.num_examples = num_examples

        self.target_sos_id = target_tokenizer.token_to_id(SOS_TOKEN)
        self.target_eos_id = target_tokenizer.token_to_id(EOS_TOKEN)

        if self.target_sos_id is None or self.target_eos_id is None:
            raise ValueError("The target tokenizer must contain [SOS] and [EOS].")

    @torch.no_grad()
    def on_epoch_end(
        self,
        pipeline: CallbackPipeline,
        epoch: int,
        logs: Dict[str, float],
    ) -> None:
        """Generate examples and store them as an MLflow table."""
        model = cast(GreedyDecodingModel, pipeline.model)
        pipeline.model.eval()

        examples = {
            "source": [],
            "target": [],
            "prediction": [],
        }

        for batch in self.validation_dataloader:
            source_batch = batch["source_token_ids"].to(pipeline.device)
            source_mask_batch = batch["source_padding_mask"].to(
                pipeline.device
            )

            for sample_index in range(source_batch.size(0)):
                predicted_token_ids = greedy_decode(
                    model=model,
                    source_token_ids=source_batch[
                        sample_index : sample_index + 1
                    ],
                    source_padding_mask=source_mask_batch[
                        sample_index : sample_index + 1
                    ],
                    target_sos_id=self.target_sos_id,
                    target_eos_id=self.target_eos_id,
                    maximum_length=self.maximum_length,
                )

                source_text = batch["source_text"][sample_index]
                target_text = batch["target_text"][sample_index]
                prediction = self.target_tokenizer.decode(
                    predicted_token_ids.detach().cpu().tolist(),
                    skip_special_tokens=True,
                )

                examples["source"].append(source_text)
                examples["target"].append(target_text)
                examples["prediction"].append(prediction)

                logger.info(
                    "[Translation example] "
                    f"source={source_text!r} target={target_text!r} "
                    f"prediction={prediction!r}"
                )

                if len(examples["source"]) == self.num_examples:
                    break

            if len(examples["source"]) == self.num_examples:
                break

        if mlflow.active_run():
            mlflow.log_table(
                data=examples,
                artifact_file=(
                    f"translation_examples/epoch_{epoch + 1}.json"
                ),
            )

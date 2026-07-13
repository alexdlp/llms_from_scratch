from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import torch
from lightning.fabric import Fabric
from dotenv import load_dotenv
from torch.utils.data import DataLoader
import mlflow
from omegaconf import OmegaConf
from itp_fabadII.utils import ConfigNamespace, flatten_dict
from itp_fabadII.logger import logger
from tqdm import tqdm
import sys
from collections.abc import Mapping, Sequence

from itp_fabadII.callbacks import Callback
from itp_fabadII.utils.color_utils import bold_green, orange

class BasePipeline(ABC):
    """
    Abstract training pipeline using PyTorch.
    Handles device placement, dataloaders, model, optimizer, and training loop orchestration.
    """

    def __init__(self, cfg):

        self._inject_config(cfg)
        self._setup_mlflow()

        self.device = self._set_device(cfg.training.device)

        fabric_devices = "auto" if self.device.type == "cuda" else 1
        self.fabric = Fabric(
            accelerator=self.device.type,
            devices=fabric_devices,
            strategy="auto",
        )

        # To be defined by subclasses
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion: torch.nn.Module | None = None
        self.callbacks: list[Callback] = []
        self.run_artifacts_dir: Path | None = None

        # flag to stop the training if needed
        self.stop_training = False


    def _inject_config(self, cfg):
        """
        Injects Hydra configuration as nested ConfigNamespace objects into the pipeline.

        Each top-level section (data, model, optimizer, loss, training, etc.)
        becomes a class attribute, preserving its internal hierarchical structure.

        Example:
            self.data.train.batch_size  -> access value directly
            CubeDataLoader(**self.data.train)  -> unpack section as kwargs
        """
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.cfg = ConfigNamespace()

        #This alternative is not recomended as we can`t use same words as class attributes
        #and as config sections. (e.g. self.model)
        for section, content in cfg_dict.items():
            setattr(self.cfg, section, ConfigNamespace.from_dict(content))

    def _set_device(self, requested: str) -> torch.device:
        """
        Selects the device requested by the user if available.
        Logs the decision and falls back to CPU if the requested
        device is unsupported on the current system.

        Args:
            requested (str): Desired device name ("cuda", "mps", "cpu").

        Returns:
            torch.device: Selected PyTorch device.
        """
        req = requested.lower()

        # --- CUDA ---
        if req == "cuda":
            if torch.cuda.is_available():
                logger.info("Using CUDA device.")
                return torch.device("cuda")
            else:
                logger.warning("Requested 'cuda' but CUDA is not available. Falling back to CPU.")
                return torch.device("cpu")

        # --- MPS ---
        if req == "mps":
            if torch.backends.mps.is_available():
                logger.info("Using MPS device.")
                return torch.device("mps")
            else:
                logger.warning("Requested 'mps' but MPS is not available. Falling back to CPU.")
                return torch.device("cpu")

        # --- CPU ---
        if req == "cpu":
            logger.info("Using CPU device.")
            return torch.device("cpu")

        # --- Unknown value ---
        logger.warning(f"Unknown device '{requested}'. Falling back to CPU.")
        return torch.device("cpu")
    
    def _setup_mlflow(self):
        """Configures MLflow tracking and experiment setup."""

        # Directory for local artifacts
        self.artifacts_dir = Path(self.cfg.mlflow.artifacts.dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Tracking URI
        mlflow.set_tracking_uri(self.cfg.mlflow.artifacts.dir)

        # --- Resume logic ---
        # Check for a resume_training argument either at root or inside model
        self.resume_run_id = os.getenv("RESUME_RUN_ID", None)
        self.resuming = bool(self.resume_run_id)
        if not self.resuming:

            experiment_name = getattr(self.cfg.model.experiment, "name", None)
            run_name = getattr(self.cfg.model.experiment, "run_name", None)

            if experiment_name:
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow experiment set: {experiment_name}, run: {run_name}")

            # Guarda en el objeto
            self.experiment_name = experiment_name
            self.run_name = run_name
        else:

            # -----------------------------
            # Validate MLflow run existence
            # -----------------------------
            try:
                run = mlflow.get_run(run_id=self.resume_run_id)
            except Exception:
                raise ValueError(
                    f"resume_training references a non-existent run_id: {self.resume_run_id}"
                )

            # Retrieve experiment info
            exp_id = run.info.experiment_id
            exp = mlflow.get_experiment(exp_id)

            if exp is None:
                raise ValueError(
                    f"Run '{self.resume_run_id}' exists but its experiment_id '{exp_id}' is invalid."
                )

            # Extract experiment & run name
            self.experiment_name = exp.name
            self.run_name = run.info.run_name

            logger.info(f"Resuming MLflow run: {self.resume_run_id}")
            logger.info(f" → Experiment: {self.experiment_name}")
            logger.info(f" → Run name:  {self.run_name}")

            # Optional: restore last_epoch if you log it
            # last_epoch = int(run.data.params.get("training.last_epoch", 0))
            # self.last_epoch = last_epoch

    def _load_resume_model(self, run_id: str):
        """
        Loads full training state from MLflow checkpoint:
        - model_state
        - optimizer_state
        - scheduler_state
        - last epoch

        Returns:
            model, optimizer_state, scheduler_state, last_epoch
        """

        logger.info(f"Loading LAST checkpoint from MLflow run: {run_id}")

        client = mlflow.tracking.MlflowClient()

        # 1) Download checkpoint folder
        ckpt_local_path = client.download_artifacts(
            run_id=run_id,
            path="checkpoints/last.ckpt"
        )

        # 2) Load checkpoint
        checkpoint = torch.load(ckpt_local_path, map_location=self.device)

        # 3) Recreate model
        model = self.build_model()
        model.load_state_dict(checkpoint["model_state"])

        # 4) Restore optimizer/scheduler state (optional in setup)
        optimizer_state = checkpoint.get("optimizer_state", None)
        scheduler_state = checkpoint.get("scheduler_state", None)

        # 5) Last epoch
        last_epoch = checkpoint.get("epoch", 0)

        logger.info(f"Loaded checkpoint from epoch {last_epoch}")

        return model, optimizer_state, scheduler_state, last_epoch
         

    # ---------- user extension points ----------
    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader]: ...

    @abstractmethod
    def build_model(self) -> torch.nn.Module: ...

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor: ...

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def init_callbacks(self) -> list[Callback]: ...
    
    def build_loss(self) -> torch.nn.Module:
        name = self.cfg.loss.name.lower()
        params = {k: v for k, v in self.cfg.loss.items() if k != "name"}

        loss_classes = {
            "mse": torch.nn.MSELoss,
            "l1": torch.nn.L1Loss,
            "bce": torch.nn.BCEWithLogitsLoss,
            "cross_entropy": torch.nn.CrossEntropyLoss,
        }

        if name not in loss_classes:
            raise ValueError(f"Unknown loss: {name}")

        return loss_classes[name](**params)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        name = self.cfg.optimizer.name.lower()
        params = {k: v for k, v in self.cfg.optimizer.items() if k != "name"}

        optimizer_classes = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }

        if name not in optimizer_classes:
            raise ValueError(f"Unknown optimizer: {name}")

        return optimizer_classes[name](model.parameters(), **params)
    

    # ---------- orchestration ----------
    def setup(self) -> None:
        """Initializes model, optimizer, dataloaders, and optionally resumes state."""

        self.fabric.launch()
        
        self.train_loader, self.val_loader = self.load_data()
        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(
            self.train_loader,
            self.val_loader,
        )

        # --- Build or load model ---
        if self.resuming:
            logger.info(orange(f"Resuming model state from run: {self.resume_run_id}"))
            model, opt_state, _, last_epoch = self._load_resume_model(self.resume_run_id)
            self.last_epoch = last_epoch
        else:
            model = self.build_model()

        self.optimizer = self.build_optimizer(model)
        self.model, self.optimizer = self.fabric.setup(model, self.optimizer)

        if self.resuming and opt_state is not None:
            self.optimizer.load_state_dict(opt_state)

        self.criterion = self.build_loss().to(self.device)
        self.callbacks = self.init_callbacks()

        logger.info("Setup completed.")


    def _log_config_to_mlflow(self) -> None:
        """
        Logs all configuration parameters (self.cfg) recursively to MLflow.
        Handles nested namespaces by flattening them into dot notation.
        Example: cfg.data.train.batch_size -> "data.train.batch_size"
        """
        try:
            # Convert ConfigNamespace → dict (OmegaConf compatible)
            #cfg_dict = OmegaConf.to_container(OmegaConf.create(vars(self.cfg)), resolve=True)
            cfg_dict = ConfigNamespace.to_builtin(self.cfg)

            flat_cfg = flatten_dict(cfg_dict)
            mlflow.log_params(flat_cfg)
            logger.info(f"Logged {len(flat_cfg)} configuration parameters to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log configuration to MLflow: {e}")

    def _apply_prefix(self, metrics: dict[str, float], prefix: str | None) -> dict[str, float]:
        """
        Returns a new metrics dict where each key has the prefix applied
        (e.g. 'loss' -> 'train_loss'), unless it already starts with the prefix.
        """
        if prefix is None:
            return metrics

        prefixed = {}
        for name, value in metrics.items():
            if not name.startswith(f"{prefix}_"):
                name = f"{prefix}_{name}"
            prefixed[name] = value
        return prefixed


    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Logs metrics to MLflow. Assumes metrics are already prefixed.
        """
        if not mlflow.active_run():
            logger.warning("Attempted to log metrics outside an active MLflow run.")
            return

        for name, value in metrics.items():
            try:
                mlflow.log_metric(name, float(value), step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric '{name}': {e}")


    def _move_to_device(self, batch: Any) -> Any:
        """
        Recursively moves tensors in `batch` to `self.device`.

        Supports:
        - single tensors
        - mappings (dict-like)
        - sequences (list/tuple) of tensors or nested structures.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)

        if isinstance(batch, Mapping):
            return {k: self._move_to_device(v) for k, v in batch.items()}

        if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            return type(batch)(self._move_to_device(v) for v in batch)

        # Anything else (ints, strings, etc.) is returned as-is
        return batch
    
    def _run_callbacks(self, hook_name: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            hook = getattr(cb, hook_name, None)
            if hook is not None:
                hook(pipeline=self, **kwargs)
    
    def _get_run_artifacst_dir(self) -> str:
        """
        Returns the actual artifact directory for the current MLflow run.
        """
        artifact_uri = mlflow.get_artifact_uri()

        # Standard MLflow local behavior
        if artifact_uri.startswith("file://"):
            return artifact_uri.replace("file://", "")

        # Remote stores already return valid URIs
        return artifact_uri

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Runs one full training epoch and logs averaged training metrics."""
        self.model.train()       
        metrics_accum: dict[str, list[float]] = {}

        # tqdm solo en rank 0
        train_iter = tqdm(self.train_loader, desc=f"[Train Epoch {epoch+1}]",
        dynamic_ncols=True,     # adapta el ancho automáticamente
        position=0,             # mantiene la barra en una sola línea
        leave=True,             # deja la barra final al acabar
        file=sys.stdout,        # asegura que escribe en stdout real
        smoothing=0.3,          # suaviza el movimiento del promedio
        mininterval=0.05,       # refresco más frecuente
        colour="green",          # color moderno si el terminal lo soporta
        bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        )

        for i, batch in enumerate(train_iter):

            batch = self._move_to_device(batch)
            out = self.training_step(batch, i)

            loss = out["loss"]

            self.optimizer.zero_grad()
            self.fabric.backward(loss)
            self.optimizer.step()

            # Acumular métricas
            for key, val in out.items():
                val = float(val.detach().item())
                
                if key not in metrics_accum:
                    metrics_accum[key] = val
                else:
                    metrics_accum[key] += val

            # # Actualizar barra de progreso
            if i % 100 == 0:
                train_iter.set_postfix({"loss": f"{val:.3f}"})

        metrics_mean = {k: total / (i + 1) for k, total in metrics_accum.items()}
        metrics_mean = self._apply_prefix(metrics=metrics_mean, prefix="train")
        
        self.log_metrics(metrics_mean, step=epoch)
        return metrics_mean
        
    
    def _validate_epoch(self, epoch: int) -> dict[str, float]:
        """Run one full validation epoch with tqdm progress bar."""
        self.model.eval()
        metrics_accum: dict[str, list[float]] = {}


        val_iter = tqdm(self.val_loader, desc=f"[Val   Epoch {epoch+1}]",
            dynamic_ncols=True,     # adapta el ancho automáticamente
            position=1,             # mantiene la barra en una sola línea
            leave=True,             # deja la barra final al acabar
            file=sys.stdout,        # asegura que escribe en stdout real
            smoothing=0.3,          # suaviza el movimiento del promedio
            mininterval=0.05,       # refresco más frecuente
            colour="magenta",          # color moderno si el terminal lo soporta
            bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        )

        with torch.no_grad():
            for i, batch in enumerate(val_iter):
                
                batch = self._move_to_device(batch)
                out = self.validation_step(batch, i)
                
                for key, val in out.items():
                    val = float(val.detach().item())
                    
                    if key not in metrics_accum:
                        metrics_accum[key] = val
                    else:
                        metrics_accum[key] += val

                # # Actualizar barra de progreso
                if i % 100 == 0:
                    val_iter.set_postfix({"val_loss": f"{val:.3f}"})

        metrics_mean = {k: total / (i + 1) for k, total in metrics_accum.items()}
        metrics_mean = self._apply_prefix(metrics=metrics_mean, prefix="val")
        
        self.log_metrics(metrics_mean, step=epoch)
        return metrics_mean
  
    def fit(self) -> None:
        """
        Executes the full training and validation loop with Fabric and MLflow integration.

        - Sets up model, optimizer, and dataloaders via `setup()`.
        - Initializes a new MLflow run or resumes an existing one.
        - Runs training and validation for the configured number of epochs.
        - Logs all averaged metrics and configuration to MLflow.
        """
        epochs: int = int(self.cfg.training.epochs)

        # ------------------ RESUME OR NEW RUN ------------------
        if self.resuming:

            logger.info(f"Resuming MLflow run: {self.resume_run_id}")
            mlflow_context = mlflow.start_run(run_id=self.resume_run_id)
        else:
            logger.info(bold_green(f"Starting new training run: {self.run_name}"))
            mlflow_context = mlflow.start_run(run_name=self.run_name)

        # ------------------ MLflow CONTEXT ------------------
        with mlflow_context:
            self.run_artifacts_dir = Path(self._get_run_artifacst_dir())
            self.run_artifacts_dir.mkdir(parents=True, exist_ok=True)

            self.setup()

            start_epoch = getattr(self, "last_epoch", 0) + 1 if self.resuming else 0
            if self.resuming:
                epochs = start_epoch + epochs

            if os.getenv("FAST_DEV_RUN") == "1":
                epochs = start_epoch + 1
                logger.info(orange("Fast dev run enabled. Running for 1 epoch"))

            # Log config once per run
            self._log_config_to_mlflow()

            # Callbacks: inicio de fit
            self._run_callbacks("on_fit_start")

            # ---- Main training loop ----
            for epoch in range(start_epoch, epochs):

                train_metrics = self._train_epoch(epoch)
                val_metrics = self._validate_epoch(epoch)

                # Optional: merge for inspection or callbacks
                all_metrics = {**train_metrics, **val_metrics}

                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in all_metrics.items())
                logger.info(f"[Epoch {epoch+1}/{epochs}] → {metrics_str}")

                # Callbacks de fin de epoch
                self._run_callbacks("on_epoch_end", epoch=epoch, logs=all_metrics)

                if self.stop_training:
                    break;

                # Keep track of the latest epoch
                self.last_epoch = epoch + 1


            self._run_callbacks("on_fit_end")
            logger.info(bold_green("Training loop finished successfully"))

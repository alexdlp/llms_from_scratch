from typing import Any

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from .. import register_pipeline
from .base_pipeline import BasePipeline
from ..callbacks import Callback, EarlyStopping, ModelCheckpoint
from ..dataset.bilingual_dataloader import BilingualDataLoader, BilingualDataset
from ..dataset.bilingual_dataset import BilingualDatasetBuilder
from ..model import Transformer, build_transformer


@register_pipeline("transformer")
class TransformerPipeline(BasePipeline):
    """Train an encoder-decoder Transformer for bilingual translation.

    The pipeline coordinates the bilingual dataset builder, PyTorch
    dataloaders, Transformer construction, optimization, and training
    callbacks. The concrete training and validation steps will be implemented
    once the model's forward path has been reviewed.
    """

    def load_data(self) -> tuple[DataLoader, DataLoader]:
        """Build the bilingual datasets and their train/validation loaders."""
 
        dataset_builder = BilingualDatasetBuilder(
            artifacts_dir=self.run_artifacts_dir / "preprocessing",
            resume=self.resuming,
            **self.cfg.data.dataset
        )

        train_dataset, validation_dataset = dataset_builder.build_datasets()

        # The model needs both vocabulary sizes, so retain the tokenizers built
        # or loaded as part of dataset preparation.
        self.source_tokenizer = train_dataset.source_tokenizer
        self.target_tokenizer = train_dataset.target_tokenizer

        train_dataloader = BilingualDataLoader(
            dataset=train_dataset,
            **self.cfg.data.train,
        )

        validation_dataloader = BilingualDataLoader(
            dataset=validation_dataset,
            **self.cfg.data.val,
        )

        return train_dataloader, validation_dataloader


        

    def build_model(self) -> Transformer:
        """Build a Transformer sized for the prepared bilingual vocabulary."""
        
        return build_transformer(
            source_vocabulary_size=self.source_tokenizer.get_vocab_size(),
            target_vocabulary_size=self.target_tokenizer.get_vocab_size(),
            sequence_length=self.cfg.data.dataset.sequence_length,
            **self.cfg.model.params,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:

        return {
            "loss": loss,
            "mu_hat": mu_hat.mean(),
            "sigma_hat": sigma_hat.mean(),
        }

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
     
      
        return {
            "loss": loss,
            "mu_hat": mu_hat.mean(),
            "sigma_hat": sigma_hat.mean(),
        }

    def init_callbacks(self) -> list[Callback]:
        """Create the callbacks configured for Transformer training."""

        model_checkpoint = ModelCheckpoint(**self.cfg.callbacks.model_checkpoint)
        early_stopping = EarlyStopping(**self.cfg.callbacks.early_stopping)

        return [model_checkpoint, early_stopping]

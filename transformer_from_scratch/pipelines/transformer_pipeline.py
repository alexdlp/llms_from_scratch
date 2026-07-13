from typing import Any, Dict

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from .. import register_pipeline
from .base_pipeline import BasePipeline
from ..callbacks import Callback, EarlyStopping, ModelCheckpoint
from ..dataset.bilingual_dataloader import BilingualDataLoader, PAD_TOKEN
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

        encoder_input = batch['source_token_ids'] # (b, seq_len)
        decoder_input = batch['target_input_token_ids'] # (B, seq_len)
        encoder_mask = batch['source_padding_mask'] # (B, 1, 1, seq_len)
        decoder_mask = batch['target_attention_mask'] # (B, 1, seq_len, seq_len)
        label = batch['target_output_token_ids'] # (B, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
        proj_output = self.model.project(decoder_output) # (B, seq_len, vocab_size)

        # Compute the loss using a simple cross entropy
        loss = self.criterion(proj_output.view(-1, self.target_tokenizer.get_vocab_size()), label.view(-1))

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        
        encoder_input = batch['source_token_ids'] # (b, seq_len)
        decoder_input = batch['target_input_token_ids'] # (B, seq_len)
        encoder_mask = batch['source_padding_mask'] # (B, 1, 1, seq_len)
        decoder_mask = batch['target_attention_mask'] # (B, 1, seq_len, seq_len)
        label = batch['target_output_token_ids'] # (B, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
        proj_output = self.model.project(decoder_output) # (B, seq_len, vocab_size)

        # Compute the loss using a simple cross entropy
        loss = self.criterion(proj_output.view(-1, self.target_tokenizer.get_vocab_size()), label.view(-1))

        return {"loss": loss}
    
    def build_loss(self) -> torch.nn.Module:
        """Build cross entropy using the target tokenizer padding ID."""
        target_padding_id = self.target_tokenizer.token_to_id(PAD_TOKEN)

        if target_padding_id is None:
            raise ValueError("The target tokenizer does not contain [PAD].")

        loss_parameters = {
            key: value
            for key, value in self.cfg.loss.items()
            if key not in {"name", "ignore_index"}
        }

        return torch.nn.CrossEntropyLoss(
            ignore_index=target_padding_id,
            **loss_parameters,
        )
    
    def init_callbacks(self) -> list[Callback]:
        """Create the callbacks configured for Transformer training."""

        model_checkpoint = ModelCheckpoint(**self.cfg.callbacks.model_checkpoint)
        early_stopping = EarlyStopping(**self.cfg.callbacks.early_stopping)

        return [model_checkpoint, early_stopping]

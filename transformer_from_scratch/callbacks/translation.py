from collections.abc import Iterator
from typing import cast, Dict

import mlflow
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore, CHRFScore

from ..dataset.bilingual_common import EOS_TOKEN, SOS_TOKEN
from ..decoding import GreedyDecodingModel, greedy_decode
from ..logger import logger
from ..model import Transformer
from ..visualizations import build_attention_chart
from .base import Callback, CallbackPipeline


def _get_target_boundary_ids(target_tokenizer: Tokenizer) -> tuple[int, int]:
    """Return the target ``[SOS]`` and ``[EOS]`` token IDs."""
    target_sos_id = target_tokenizer.token_to_id(SOS_TOKEN)
    target_eos_id = target_tokenizer.token_to_id(EOS_TOKEN)

    if target_sos_id is None or target_eos_id is None:
        raise ValueError("The target tokenizer must contain [SOS] and [EOS].")

    return target_sos_id, target_eos_id


def _generate_validation_translations(
    model: GreedyDecodingModel,
    validation_dataloader: DataLoader,
    target_tokenizer: Tokenizer,
    device: torch.device,
    target_sos_id: int,
    target_eos_id: int,
    maximum_length: int,
    maximum_examples: int | None,
) -> Iterator[tuple[str, str, str]]:
    """Yield source, reference, and predicted texts from validation batches."""
    generated_examples = 0

    for batch in validation_dataloader:
        source_batch = batch["source_token_ids"].to(device)
        source_mask_batch = batch["source_padding_mask"].to(device)

        for sample_index in range(source_batch.size(0)):
            predicted_token_ids = greedy_decode(
                model=model,
                source_token_ids=source_batch[
                    sample_index : sample_index + 1
                ],
                source_padding_mask=source_mask_batch[
                    sample_index : sample_index + 1
                ],
                target_sos_id=target_sos_id,
                target_eos_id=target_eos_id,
                maximum_length=maximum_length,
            )

            prediction = target_tokenizer.decode(
                predicted_token_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )

            yield (
                batch["source_text"][sample_index],
                batch["target_text"][sample_index],
                prediction,
            )

            generated_examples += 1
            if (
                maximum_examples is not None
                and generated_examples >= maximum_examples
            ):
                return


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
        self.target_sos_id, self.target_eos_id = _get_target_boundary_ids(
            target_tokenizer
        )

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

        translations = list(
            _generate_validation_translations(
                model=model,
                validation_dataloader=self.validation_dataloader,
                target_tokenizer=self.target_tokenizer,
                device=pipeline.device,
                target_sos_id=self.target_sos_id,
                target_eos_id=self.target_eos_id,
                maximum_length=self.maximum_length,
                maximum_examples=self.num_examples,
            )
        )

        examples = {
            "source": [source for source, _, _ in translations],
            "target": [target for _, target, _ in translations],
            "prediction": [prediction for _, _, prediction in translations],
        }

        for source_text, target_text, prediction in translations:
            logger.info(
                "[Translation example] "
                f"source={source_text!r} target={target_text!r} "
                f"prediction={prediction!r}"
            )

        if mlflow.active_run():
            mlflow.log_table(
                data=examples,
                artifact_file=(
                    f"translation_examples/epoch_{epoch + 1}.json"
                ),
            )


class TranslationMetricsCallback(Callback):
    """Evaluate corpus-level BLEU and chrF from generated translations.

    Args:
        validation_dataloader: Loader that provides prepared bilingual samples.
        target_tokenizer: Tokenizer used to decode the generated target IDs.
        maximum_length: Maximum generated sequence length, including ``[SOS]``.
        num_samples: Validation samples used for periodic metric calculation.
        every_n_epochs: Number of epochs between periodic evaluations.
        evaluate_full_dataset_on_fit_end: Evaluate the complete validation set
            after training finishes.
    """

    def __init__(
        self,
        validation_dataloader: DataLoader,
        target_tokenizer: Tokenizer,
        maximum_length: int,
        num_samples: int = 200,
        every_n_epochs: int = 5,
        evaluate_full_dataset_on_fit_end: bool = True,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")

        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1.")

        self.validation_dataloader = validation_dataloader
        self.target_tokenizer = target_tokenizer
        self.maximum_length = maximum_length
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.evaluate_full_dataset_on_fit_end = (
            evaluate_full_dataset_on_fit_end
        )
        self.target_sos_id, self.target_eos_id = _get_target_boundary_ids(
            target_tokenizer
        )
        self._last_epoch = 0

    @torch.no_grad()
    def on_epoch_end(
        self,
        pipeline: CallbackPipeline,
        epoch: int,
        logs: Dict[str, float],
    ) -> None:
        """Calculate metrics periodically on a bounded validation sample."""
        self._last_epoch = epoch

        if (epoch + 1) % self.every_n_epochs != 0:
            return

        self._evaluate(
            pipeline=pipeline,
            maximum_examples=self.num_samples,
            step=epoch,
            metric_suffix="",
        )

    @torch.no_grad()
    def on_fit_end(self, pipeline: CallbackPipeline) -> None:
        """Optionally calculate final metrics on the full validation set."""
        if not self.evaluate_full_dataset_on_fit_end:
            return

        self._evaluate(
            pipeline=pipeline,
            maximum_examples=None,
            step=self._last_epoch,
            metric_suffix="_full",
        )

    def _evaluate(
        self,
        pipeline: CallbackPipeline,
        maximum_examples: int | None,
        step: int,
        metric_suffix: str,
    ) -> None:
        """Generate translations, calculate both metrics, and log them."""
        if not mlflow.active_run():
            return

        pipeline.model.eval()
        model = cast(GreedyDecodingModel, pipeline.model)
        translations = list(
            _generate_validation_translations(
                model=model,
                validation_dataloader=self.validation_dataloader,
                target_tokenizer=self.target_tokenizer,
                device=pipeline.device,
                target_sos_id=self.target_sos_id,
                target_eos_id=self.target_eos_id,
                maximum_length=self.maximum_length,
                maximum_examples=maximum_examples,
            )
        )

        if not translations:
            logger.warning("No validation translations available for metrics.")
            return

        predictions = [prediction for _, _, prediction in translations]
        references = [[target] for _, target, _ in translations]

        bleu = float(BLEUScore(smooth=True)(predictions, references))
        chrf = float(CHRFScore(n_word_order=0)(predictions, references))
        metrics = {
            f"val_bleu{metric_suffix}": bleu,
            f"val_chrf{metric_suffix}": chrf,
        }

        mlflow.log_metrics(metrics, step=step)
        logger.info(
            f"Translation metrics ({len(translations)} samples): "
            f"BLEU={bleu:.4f}, chrF={chrf:.4f}"
        )


class AttentionVisualizationCallback(Callback):
    """Save attention heatmaps for one fixed validation example.

    Args:
        validation_dataloader: Loader that provides prepared bilingual samples.
        source_tokenizer: Tokenizer used to label encoder attention axes.
        target_tokenizer: Tokenizer used to label decoder attention axes.
        maximum_length: Maximum target length used by greedy decoding.
        layer_indices: Transformer layers included in the visualizations.
        head_indices: Attention heads included in the visualizations.
        maximum_tokens: Maximum number of tokens shown on each axis.
        every_n_epochs: Number of epochs between visualizations.
    """

    def __init__(
        self,
        validation_dataloader: DataLoader,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        maximum_length: int,
        layer_indices: list[int],
        head_indices: list[int],
        maximum_tokens: int = 20,
        every_n_epochs: int = 5,
    ) -> None:
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1.")

        if maximum_tokens < 1:
            raise ValueError("maximum_tokens must be at least 1.")

        self.validation_dataloader = validation_dataloader
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.maximum_length = maximum_length
        self.layer_indices = layer_indices
        self.head_indices = head_indices
        self.maximum_tokens = maximum_tokens
        self.every_n_epochs = every_n_epochs
        self.target_sos_id, self.target_eos_id = _get_target_boundary_ids(
            target_tokenizer
        )

    @torch.no_grad()
    def on_epoch_end(
        self,
        pipeline: CallbackPipeline,
        epoch: int,
        logs: Dict[str, float],
    ) -> None:
        """Generate one translation and save its three attention charts."""
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        if not mlflow.active_run() or pipeline.run_artifacts_dir is None:
            return

        batch = next(iter(self.validation_dataloader))
        source_token_ids = batch["source_token_ids"][0:1].to(pipeline.device)
        source_padding_mask = batch["source_padding_mask"][0:1].to(
            pipeline.device
        )

        pipeline.model.eval()
        generated_token_ids = greedy_decode(
            model=cast(GreedyDecodingModel, pipeline.model),
            source_token_ids=source_token_ids,
            source_padding_mask=source_padding_mask,
            target_sos_id=self.target_sos_id,
            target_eos_id=self.target_eos_id,
            maximum_length=self.maximum_length,
        )

        source_tokens = self._decode_tokens(
            self.source_tokenizer,
            source_token_ids[0],
        )
        decoder_tokens = self._decode_tokens(
            self.target_tokenizer,
            generated_token_ids[:-1],
        )

        wrapped_model = getattr(pipeline.model, "module", pipeline.model)
        model = cast(Transformer, wrapped_model)
        output_dir = (
            pipeline.run_artifacts_dir
            / "attention"
            / f"epoch_{epoch + 1}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        charts = {
            "encoder_self_attention": build_attention_chart(
                model=model,
                attention_type="encoder",
                layer_indices=self.layer_indices,
                head_indices=self.head_indices,
                row_tokens=source_tokens,
                column_tokens=source_tokens,
                maximum_tokens=self.maximum_tokens,
            ),
            "decoder_self_attention": build_attention_chart(
                model=model,
                attention_type="decoder",
                layer_indices=self.layer_indices,
                head_indices=self.head_indices,
                row_tokens=decoder_tokens,
                column_tokens=decoder_tokens,
                maximum_tokens=self.maximum_tokens,
            ),
            "cross_attention": build_attention_chart(
                model=model,
                attention_type="cross",
                layer_indices=self.layer_indices,
                head_indices=self.head_indices,
                row_tokens=decoder_tokens,
                column_tokens=source_tokens,
                maximum_tokens=self.maximum_tokens,
            ),
        }

        for chart_name, chart in charts.items():
            chart.save(output_dir / f"{chart_name}.html")

        logger.info(f"Saved attention visualizations to: {output_dir}")

    @staticmethod
    def _decode_tokens(
        tokenizer: Tokenizer,
        token_ids: torch.Tensor,
    ) -> list[str]:
        """Convert token IDs to labels, stopping before padding."""
        tokens = []

        for token_id in token_ids.detach().cpu().tolist():
            token = tokenizer.id_to_token(token_id)
            if token == "[PAD]":
                break
            tokens.append(token or "[UNK]")

        return tokens

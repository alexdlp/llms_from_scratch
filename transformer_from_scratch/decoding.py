from typing import Protocol

import torch

from .dataset.bilingual_dataloader import causal_mask


class GreedyDecodingModel(Protocol):
    """Transformer operations required by greedy decoding."""

    def encode(
        self,
        source_token_ids: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def decode(
        self,
        encoder_output: torch.Tensor,
        source_padding_mask: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def project(self, decoder_output: torch.Tensor) -> torch.Tensor: ...


def greedy_decode(
    model: GreedyDecodingModel,
    source_token_ids: torch.Tensor,
    source_padding_mask: torch.Tensor,
    target_sos_id: int,
    target_eos_id: int,
    maximum_length: int,
) -> torch.Tensor:
    """Generate one target sequence by repeatedly selecting the best token.

    The source is encoded only once. The decoder then starts with ``[SOS]``
    and predicts one token at a time, reusing the encoder output at every
    iteration. Generation stops after producing ``[EOS]`` or reaching the
    configured maximum length.

    Args:
        model: Transformer exposing encoder, decoder, and projection methods.
        source_token_ids: One padded source sequence with shape ``(1, length)``.
        source_padding_mask: Padding mask corresponding to the source sequence.
        target_sos_id: Target tokenizer ID used to start generation.
        target_eos_id: Target tokenizer ID that ends generation.
        maximum_length: Maximum number of generated tokens, including ``[SOS]``.

    Returns:
        Generated target token IDs with shape ``(generated_length,)``.
    """
    if source_token_ids.ndim != 2 or source_token_ids.size(0) != 1:
        raise ValueError("greedy_decode expects exactly one source sequence.")

    if maximum_length < 2:
        raise ValueError("maximum_length must be at least 2.")

    encoder_output = model.encode(source_token_ids, source_padding_mask)
    decoder_input = source_token_ids.new_full((1, 1), target_sos_id)

    while decoder_input.size(1) < maximum_length:
        decoder_mask = causal_mask(decoder_input.size(1)).to(
            source_token_ids.device
        )
        decoder_output = model.decode(
            encoder_output,
            source_padding_mask,
            decoder_input,
            decoder_mask,
        )

        next_token_scores = model.project(decoder_output[:, -1])
        next_token_id = int(next_token_scores.argmax(dim=-1).item())

        next_token = source_token_ids.new_full((1, 1), next_token_id)
        decoder_input = torch.cat((decoder_input, next_token), dim=1)

        if next_token_id == target_eos_id:
            break

    return decoder_input.squeeze(0)

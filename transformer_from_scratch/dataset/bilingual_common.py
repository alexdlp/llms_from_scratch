from typing import TypedDict

import torch

SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]


class TranslationExample(TypedDict):
    """Raw bilingual example returned by the translation dataset."""

    translation: dict[str, str]


class BilingualSample(TypedDict):
    """Tensors and original texts returned for one prepared example."""

    source_token_ids: torch.Tensor
    target_input_token_ids: torch.Tensor
    source_padding_mask: torch.Tensor
    target_attention_mask: torch.Tensor
    target_output_token_ids: torch.Tensor
    source_text: str
    target_text: str

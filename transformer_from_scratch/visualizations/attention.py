from typing import Literal

import altair as alt
import torch

from ..model import Transformer

AttentionType = Literal["encoder", "decoder", "cross"]


def _get_attention_scores(
    model: Transformer,
    attention_type: AttentionType,
    layer_index: int,
) -> torch.Tensor:
    """Return the attention scores stored by one Transformer layer."""
    if attention_type == "encoder":
        attention_block = model.encoder.layers[
            layer_index
        ].self_attention_block
    elif attention_type == "decoder":
        attention_block = model.decoder.layers[
            layer_index
        ].self_attention_block
    elif attention_type == "cross":
        attention_block = model.decoder.layers[
            layer_index
        ].cross_attention_block
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

    attention_scores = getattr(attention_block, "attention_scores", None)
    if attention_scores is None:
        raise RuntimeError(
            "Attention scores are unavailable. Run a model forward pass before "
            "building attention charts."
        )

    return attention_scores.detach().cpu()


def _build_head_chart(
    attention_matrix: torch.Tensor,
    row_tokens: list[str],
    column_tokens: list[str],
    layer_index: int,
    head_index: int,
    maximum_tokens: int,
) -> alt.Chart:
    """Build one heatmap for a single attention head."""
    row_count = min(attention_matrix.size(0), len(row_tokens), maximum_tokens)
    column_count = min(
        attention_matrix.size(1),
        len(column_tokens),
        maximum_tokens,
    )

    records = [
        {
            "row": row_index,
            "column": column_index,
            "value": float(attention_matrix[row_index, column_index]),
            "row_token": f"{row_index:03d} {row_tokens[row_index]}",
            "column_token": (
                f"{column_index:03d} {column_tokens[column_index]}"
            ),
        }
        for row_index in range(row_count)
        for column_index in range(column_count)
    ]

    return (
        alt.Chart(alt.Data(values=records))
        .mark_rect()
        .encode(
            x=alt.X("column_token:N", title=None, sort=None),
            y=alt.Y("row_token:N", title=None, sort=None),
            color=alt.Color("value:Q", title="Attention"),
            tooltip=[
                alt.Tooltip("row:Q"),
                alt.Tooltip("column:Q"),
                alt.Tooltip("value:Q", format=".4f"),
                alt.Tooltip("row_token:N"),
                alt.Tooltip("column_token:N"),
            ],
        )
        .properties(
            height=300,
            width=300,
            title=f"Layer {layer_index} · Head {head_index}",
        )
        .interactive(name=f"layer_{layer_index}_head_{head_index}")
    )


def build_attention_chart(
    model: Transformer,
    attention_type: AttentionType,
    layer_indices: list[int],
    head_indices: list[int],
    row_tokens: list[str],
    column_tokens: list[str],
    maximum_tokens: int,
) -> alt.VConcatChart:
    """Build a compound heatmap for selected Transformer layers and heads."""
    if maximum_tokens < 1:
        raise ValueError("maximum_tokens must be at least 1.")

    if not layer_indices:
        raise ValueError("At least one layer index must be selected.")

    if not head_indices:
        raise ValueError("At least one head index must be selected.")

    layer_charts = []

    for layer_index in layer_indices:
        attention_scores = _get_attention_scores(
            model,
            attention_type,
            layer_index,
        )

        head_charts = []
        for head_index in head_indices:
            if head_index >= attention_scores.size(1):
                raise IndexError(
                    f"Attention head {head_index} does not exist in layer "
                    f"{layer_index}."
                )

            head_charts.append(
                _build_head_chart(
                    attention_matrix=attention_scores[0, head_index],
                    row_tokens=row_tokens,
                    column_tokens=column_tokens,
                    layer_index=layer_index,
                    head_index=head_index,
                    maximum_tokens=maximum_tokens,
                )
            )

        layer_charts.append(alt.hconcat(*head_charts))

    return alt.vconcat(*layer_charts)

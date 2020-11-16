from typing import *

import torch
from torch import nn
from allennlp.modules.span_extractors import SpanExtractor

from modules.span_graph_encoder import SpanGraphEncoder
from modules.span_typer import SpanTyper


class SpanSelector(nn.Module):
    """
    SpanSelector accepts a list of span indices with
    """

    def __init__(self,
                 span_extractor: SpanExtractor,
                 span_graph_encoder: SpanGraphEncoder,
                 span_typer: SpanTyper):
        super(SpanSelector, self).__init__()

        self.span_extractor: SpanExtractor = span_extractor  # [batch_size, seq_len, embed_dim]
        self.span_graph_encoder: SpanGraphEncoder = span_graph_encoder
        self.span_affine: nn.Linear = nn.Linear(self.span_extractor.get_output_dim(),
                                                self.span_graph_encoder.get_input_dim())

        self.span_typer: SpanTyper = span_typer

    def forward(self,
                sequence_tensor: torch.Tensor,  # [batch_size, seq_len, embed_dim]
                span_indices: torch.Tensor,  # [batch_size, num_spans, 2]
                span_indices_mask: Optional[torch.Tensor] = None,  # [batch_size, num_spans]
                type_mask: Optional[torch.Tensor] = None,  # [batch_size, num_types]
                span_types: Optional[torch.Tensor] = None,  # [batch_size, num_spans]
                sequence_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                loss_reduction: Optional[bool] = False
                ) -> Dict[str, torch.Tensor]:
        span_reprs: torch.Tensor = self.span_extractor(
            sequence_tensor=sequence_tensor,
            span_indices=span_indices,
            sequence_mask=sequence_mask,
            span_indices_mask=span_indices_mask
        )  # [batch_size, num_spans, embedded_span_size]

        transformed_span_reprs: torch.Tensor = self.span_repr_transform(self.span_affine(span_reprs))
        encoded_span_graph: torch.Tensor = self.span_graph_encoder(
            span_reprs=transformed_span_reprs,
            span_mask=span_indices_mask
        )  # [batch_size, num_spans, embedded_span_size]

        typer_results: Dict[str, torch.Tensor] = self.span_typer(span_reprs=encoded_span_graph,
                                                                 span_mask=span_indices_mask,
                                                                 type_mask=type_mask,
                                                                 span_types=span_types,
                                                                 loss_reduction=loss_reduction)

        return typer_results

from typing import *

import torch
from torch import nn


class SpanGraphEncoder(nn.Module):
    """
    SpanGraphEncoder accepts a list of span and treats each span as a node in the graph,
    then performs a series of operations among the nodes to get better representation of
    each node.
    """

    def __init__(self):
        super(SpanGraphEncoder, self).__init__()

    def get_input_dim(self) -> int:
        raise NotImplemented

    def forward(self,
                span_reprs: torch.Tensor,
                span_mask: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """ Computes new representations for nodes

        Args:
            span_reprs: [batch_size, num_spans, embedded_span_size]
            span_mask: [batch_size, num_spans]

        Returns:
            new_span_reprs: [batch_size, num_spans, embedded_span_size]
        """

        raise NotImplemented


class PseudoSpanGraphEncoder(SpanGraphEncoder):
    def get_input_dim(self) -> int:
        return 768

    def forward(self,
                span_reprs: torch.Tensor,
                span_mask: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return span_reprs


class TransformerSpanGraphEncoder(SpanGraphEncoder):
    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 norm: Optional[nn.Module] = None):
        super(TransformerSpanGraphEncoder, self).__init__()

        self.input_dim: int = d_model
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=d_model,
                                                                               nhead=nhead,
                                                                               dim_feedforward=dim_feedforward,
                                                                               dropout=dropout,
                                                                               activation=activation)
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm)

    def get_input_dim(self) -> int:
        return self.input_dim

    def forward(self,
                span_reprs: torch.Tensor,
                span_mask: torch.Tensor,
                **kwargs) -> torch.Tensor:
        transposed_span_reprs: torch.Tensor = span_reprs.transpose(0, 1)
        src_key_padding_mask: torch.Tensor = ~ span_mask
        # print(transposed_span_reprs.shape, src_key_padding_mask.shape)
        encoded_reprs: torch.Tensor = self.encoder(src=transposed_span_reprs,
                                                   src_key_padding_mask=src_key_padding_mask).transpose(0, 1)
        return encoded_reprs

from typing import *

import torch
from allennlp.data import Vocabulary
from allennlp.modules import TimeDistributed, FeedForward
from allennlp.modules.span_extractors import EndpointSpanExtractor, SpanExtractor, SelfAttentiveSpanExtractor
from torch import nn

from modules.span_graph_encoder import TransformerSpanGraphEncoder, SpanGraphEncoder, PseudoSpanGraphEncoder
from modules.span_typer import SpanTyper, SoftmaxSpanTyper


class SelectorArgLinking(nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 span_graph_encoder: SpanGraphEncoder,
                 span_typer: SpanTyper,
                 embed_size: int,
                 label_namespace: str = 'span_labels',
                 event_namespace: str = 'event_labels',
                 use_event_embedding: bool = True):
        super(SelectorArgLinking, self).__init__()

        self.vocab: Vocabulary = vocab
        self.label_namespace: str = label_namespace
        self.event_namespace: str = event_namespace

        self.use_event_embedding = use_event_embedding
        self.embed_size = embed_size
        self.event_embedding_size = 50

        # self.span_finder: SpanFinder = span_finder
        # self.span_selector: SpanSelector = span_selector
        if use_event_embedding:
            self.event_embeddings: nn.Embedding = nn.Embedding(
                num_embeddings=len(vocab.get_token_to_index_vocabulary(namespace=event_namespace)),
                embedding_dim=self.event_embedding_size
            )

        self.lexical_dropout = nn.Dropout(p=0.2)
        # self.contextualized_encoder: Seq2SeqEncoder = LstmSeq2SeqEncoder(
        #     bidirectional=True,
        #     input_size=embed_size,
        #     hidden_size=embed_size,
        #     num_layers=2,
        #     dropout=0.4
        # )
        self.span_graph_encoder: SpanGraphEncoder = span_graph_encoder
        self.span_extractor: SpanExtractor = EndpointSpanExtractor(
            # input_dim=self.contextualized_encoder.get_output_dim(),
            input_dim=self.embed_size,
            combination='x,y'
        )
        self.attentive_span_extractor: SpanExtractor = SelfAttentiveSpanExtractor(embed_size)

        self.arg_affine = TimeDistributed(FeedForward(
            input_dim=self.span_extractor.get_output_dim() + self.attentive_span_extractor.get_output_dim(),
            hidden_dims=self.span_graph_encoder.get_input_dim(),
            num_layers=2,
            activations=nn.GELU(),
            dropout=0.2
        ))
        self.trigger_affine = FeedForward(
            input_dim=self.span_extractor.get_output_dim() + self.attentive_span_extractor.get_output_dim(),
            hidden_dims=self.span_graph_encoder.get_input_dim() - (
                self.event_embedding_size if use_event_embedding else 0),
            num_layers=2,
            activations=nn.GELU(),
            dropout=0.2
        )
        # self.arg_affine: nn.Linear = nn.Linear(
        #     self.span_extractor.get_output_dim() + self.attentive_span_extractor.get_output_dim(),
        #     self.span_graph_encoder.get_input_dim()
        # )
        # self.trigger_affine: nn.Linear = nn.Linear(
        #     self.span_extractor.get_output_dim() + self.attentive_span_extractor.get_output_dim(),
        #     self.span_graph_encoder.get_input_dim()
        # )

        # self.trigger_event_infuse: nn.Sequential = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(4 * self.span_graph_encoder.get_input_dim(), 2 * self.span_graph_encoder.get_input_dim()),
        #     nn.Dropout(p=0.1),
        #     nn.GELU(),
        #     nn.Linear(2 * self.span_graph_encoder.get_input_dim(), self.span_graph_encoder.get_input_dim()),
        #     nn.Dropout(p=0.1),
        #     nn.GELU()
        # )

        self.span_typer: SpanTyper = span_typer

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module, initializer_range: float = 0.02):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_params(cls,
                    vocab: Vocabulary,
                    label_namespace: str = 'span_labels',
                    event_namespace: str = 'event_labels',
                    embed_dim: int = 768,
                    num_layers: int = 3,
                    dim_feedforward: int = 2048,
                    nhead: int = 12,
                    activation: str = 'gelu',
                    use_span_encoder: bool = True,
                    label_rescaling_weights: Optional[Dict[str, float]] = None,
                    use_event_embedding: bool = True,
                    **kwargs) -> 'SelectorArgLinking':
        # span_extractor: SpanExtractor = EndpointSpanExtractor(input_dim=embed_dim,
        #                                                       combination='x,y')
        # span_extractor: SpanExtractor = SelfAttentiveSpanExtractor(input_dim=embed_dim)

        if use_span_encoder:
            span_graph_encoder: SpanGraphEncoder = TransformerSpanGraphEncoder(
                num_layers=num_layers,
                d_model=embed_dim,
                dim_feedforward=dim_feedforward,
                nhead=nhead,
                activation=activation
            )
        else:
            span_graph_encoder = PseudoSpanGraphEncoder()

        span_typer: SpanTyper = SoftmaxSpanTyper(vocab=vocab,
                                                 namespace=label_namespace,
                                                 input_dim=embed_dim,
                                                 activation=activation,
                                                 label_rescaling_weights=label_rescaling_weights)
        # span_selector: SpanSelector = SpanSelector(span_extractor=span_extractor,
        #                                            span_graph_encoder=span_graph_encoder,
        #                                            span_typer=span_typer)
        return cls(
            span_graph_encoder=span_graph_encoder,
            span_typer=span_typer,
            embed_size=embed_dim,
            vocab=vocab,
            label_namespace=label_namespace,
            event_namespace=event_namespace,
            use_event_embedding=use_event_embedding
        )

    def forward(self,
                sequence_tensor: torch.Tensor,  # [batch_size, seq_len, embed_dim]
                span_indices: torch.Tensor,  # [batch_size, num_spans, 2]
                event_type: torch.Tensor,  # [batch_size]
                span_indices_mask: Optional[torch.Tensor] = None,  # [batch_size, num_spans]
                type_mask: Optional[torch.Tensor] = None,  # [batch_size, num_types]
                span_types: Optional[torch.Tensor] = None,  # [batch_size, num_spans]
                sequence_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                **kwargs) -> Dict[str, torch.Tensor]:
        # span_selector_outputs: Dict[str, torch.Tensor] = self.span_selector(sequence_tensor=sequence_tensor,
        #                                                                     span_indices=span_indices,
        #                                                                     span_indices_mask=span_indices_mask,
        #                                                                     type_mask=type_mask,
        #                                                                     span_types=span_types,
        #                                                                     sequence_mask=sequence_mask)

        # contexualized_embeddings = self.contextualized_encoder(self.lexical_dropout(sequence_tensor), sequence_mask)

        span_reprs: torch.Tensor = self.span_extractor(
            # sequence_tensor=contexualized_embeddings,
            sequence_tensor=sequence_tensor,
            span_indices=span_indices,
            sequence_mask=sequence_mask,
            span_indices_mask=span_indices_mask
        )  # [batch_size, num_spans, embedded_span_size]
        # [batch_size, num_spans, embedded_span_size]
        attentive_span_reprs = self.attentive_span_extractor(sequence_tensor, span_indices, span_indices_mask)

        combined_span_reprs = torch.cat([span_reprs, attentive_span_reprs], dim=2)

        # [batch_size, 1, event_embed_size]
        event_embeds: Optional[torch.Tensor] = (
            self.event_embeddings(event_type).unsqueeze(dim=1)
            if self.use_event_embedding else None
        )

        transformed_span_reprs: torch.Tensor = torch.cat([
            torch.cat([
                self.trigger_affine(combined_span_reprs[:, 0, :]).view(combined_span_reprs.shape[0], 1, -1),
                event_embeds
            ], dim=2)
            if self.use_event_embedding else
            self.trigger_affine(combined_span_reprs[:, 0, :]).view(combined_span_reprs.shape[0], 1, -1),
            self.arg_affine(combined_span_reprs[:, 1:, :])
        ], dim=1)  # [batch_size, num_spans, embedded_span_size]

        encoded_span_graph: torch.Tensor = self.span_graph_encoder(
            span_reprs=transformed_span_reprs,
            span_mask=span_indices_mask
        )  # [batch_size, num_spans, embedded_span_size]

        # encoded_span_graph = transformed_span_reprs

        span_selector_outputs: Dict[str, torch.Tensor] = self.span_typer(span_reprs=encoded_span_graph,
                                                                         span_mask=span_indices_mask,
                                                                         type_mask=type_mask,
                                                                         span_types=span_types,
                                                                         loss_reduction=False)
        res: Dict[str, Any] = {
            'type_ids': [
                lst[1:]
                for lst in span_selector_outputs['type_ids']
            ],
            'type_strs': [
                lst[1:]
                for lst in span_selector_outputs['type_strs']
            ]
        }
        if span_types is not None:
            # loss: torch.Tensor = (
            #         span_selector_outputs['loss'][:, 1:].logsumexp(-1).logsumexp(-1) / span_indices_mask[:, 1:].sum()
            # )
            # res['loss'] = loss
            res['loss'] = span_selector_outputs['loss']
            res['gold_type_ids'] = [
                span_types[i, span_indices_mask[i]][1:].tolist()
                for i in range(span_indices.shape[0])
            ]
            res['gold_type_strs'] = [
                [
                    self.vocab.get_token_from_index(index=i, namespace=self.label_namespace)
                    for i in lst
                ]
                for lst in res['gold_type_ids']
            ]

        return res

from pdb import set_trace
from typing import *

import torch
from allennlp.data import Vocabulary
from allennlp.modules import TimeDistributed, FeedForward
from allennlp.modules.span_extractors import EndpointSpanExtractor, SpanExtractor, SelfAttentiveSpanExtractor
from torch import nn

from modules.span_typer import SpanTyper, SoftmaxSpanTyper


class ArgumentSpanClassifier(nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 span_typer: SpanTyper,
                 embed_size: int,
                 label_namespace: str = 'span_labels',
                 event_namespace: str = 'event_labels'):
        super(ArgumentSpanClassifier, self).__init__()

        self.vocab: Vocabulary = vocab
        self.label_namespace: str = label_namespace
        self.event_namespace: str = event_namespace

        self.embed_size = embed_size
        self.event_embedding_size = 50

        self.event_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=len(vocab.get_token_to_index_vocabulary(namespace=event_namespace)),
            embedding_dim=self.event_embedding_size
        )

        self.lexical_dropout = nn.Dropout(p=0.2)
        self.span_extractor: SpanExtractor = EndpointSpanExtractor(
            input_dim=self.embed_size,
            combination='x,y'
        )
        self.attentive_span_extractor: SpanExtractor = SelfAttentiveSpanExtractor(embed_size)

        self.arg_affine = TimeDistributed(FeedForward(
            input_dim=self.span_extractor.get_output_dim() + self.attentive_span_extractor.get_output_dim(),
            hidden_dims=self.embed_size,
            num_layers=2,
            activations=nn.GELU(),
            dropout=0.2
        ))
        self.trigger_affine = FeedForward(
            input_dim=self.span_extractor.get_output_dim() + self.attentive_span_extractor.get_output_dim(),
            hidden_dims=self.embed_size - self.event_embedding_size,
            num_layers=2,
            activations=nn.GELU(),
            dropout=0.2
        )

        self.trigger_event_infusion = TimeDistributed(FeedForward(
            input_dim=2 * self.embed_size,
            hidden_dims=self.embed_size,
            num_layers=2,
            activations=nn.GELU(),
            dropout=0.2
        ))

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
                    activation: str = 'gelu',
                    embed_dim: int = 768,
                    label_rescaling_weights: Optional[Dict[str, float]] = None,
                    **kwargs) -> 'SelectorArgLinking':
        span_typer: SpanTyper = SoftmaxSpanTyper(vocab=vocab,
                                                 namespace=label_namespace,
                                                 input_dim=embed_dim,
                                                 activation=activation,
                                                 label_rescaling_weights=label_rescaling_weights)
        # span_selector: SpanSelector = SpanSelector(span_extractor=span_extractor,
        #                                            span_graph_encoder=span_graph_encoder,
        #                                            span_typer=span_typer)
        return cls(
            span_typer=span_typer,
            embed_size=embed_dim,
            vocab=vocab,
            label_namespace=label_namespace,
            event_namespace=event_namespace
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
        event_embeds: torch.Tensor = self.event_embeddings(event_type).unsqueeze(dim=1)
        batch_size, num_spans, embedded_span_size = combined_span_reprs.shape
        transformed_span_reprs = self.trigger_event_infusion(
            torch.cat([
                torch.cat([
                    self.trigger_affine(combined_span_reprs[:, 0, :]).view(combined_span_reprs.shape[0], 1, -1),
                    event_embeds
                ], dim=2).expand(batch_size, num_spans, -1),
                self.arg_affine(combined_span_reprs)
            ], dim=2)
        )

        span_selector_outputs: Dict[str, torch.Tensor] = self.span_typer(span_reprs=transformed_span_reprs,
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

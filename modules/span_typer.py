from typing import *

import torch
from allennlp.data import Vocabulary
from torch import nn


class SpanTyper(nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 namespace: str = 'span_labels'):
        super(SpanTyper, self).__init__()

        self.vocab: Vocabulary = vocab
        self.namespace: str = namespace

    def type_spans(self,
                   span_reprs: torch.Tensor,
                   span_mask: Optional[torch.Tensor] = None,
                   type_mask: Optional[torch.Tensor] = None,
                   **kwargs: Any) -> torch.Tensor:
        raise NotImplemented

    def decode_types(self, **kwargs: Any) -> torch.Tensor:
        raise NotImplemented

    def loss(self,
             preds: torch.Tensor,
             golds: torch.Tensor,
             **kwargs: Any) -> torch.Tensor:
        raise NotImplemented

    def forward(self,
                span_reprs: torch.Tensor,
                span_mask: torch.Tensor,
                type_mask: Optional[torch.Tensor] = None,
                span_types: Optional[torch.Tensor] = None,
                **kwargs: Any) -> Dict[str, Any]:
        raise NotImplemented


class SoftmaxSpanTyper(SpanTyper):
    def __init__(self,
                 vocab: Vocabulary,
                 namespace: str = 'span_labels',
                 input_dim: int = 768,
                 activation: str = 'relu',
                 label_rescaling_weights: Optional[Dict[str, float]] = None):
        super(SoftmaxSpanTyper, self).__init__(vocab=vocab,
                                               namespace=namespace)

        self.num_types: int = len(self.vocab.get_token_to_index_vocabulary(namespace=namespace))
        self.type_embedding_size: int = 50

        if activation == 'tanh':
            activation_layer = nn.Tanh
        elif activation == 'gelu':
            activation_layer = nn.GELU
        else:
            activation_layer = nn.ReLU

        self.type_embeddings: nn.Embedding = nn.Embedding(num_embeddings=self.num_types,
                                                          embedding_dim=self.type_embedding_size,
                                                          padding_idx=self.vocab.get_token_index(token='@@PADDING@@',
                                                                                                 namespace=namespace))
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(p=0.1),
            activation_layer(),
            nn.Linear(input_dim, self.type_embedding_size),
            nn.Dropout(p=0.1),
            activation_layer()
        )

        class_weights: List[float] = [1 for _ in range(self.num_types)]
        if label_rescaling_weights is not None:
            for k, v in label_rescaling_weights.items():
                class_weights[self.vocab.get_token_index(k, namespace=namespace)] = v
        # else:
            # class_weights[
            #     self.vocab.get_token_index('None', namespace=namespace)
            # ] = 1 / self.num_types * 2  # down scale the weight for Null type
            # class_weights[1] = 0.  # ignore `@@PADDING@@`

        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.vocab.get_token_index(token='@@PADDING@@',
                                                                                         namespace=namespace),
                                                 weight=torch.tensor(class_weights, dtype=torch.float))
                                                 # reduction='none')

    def type_spans(self,
                   span_reprs: torch.Tensor,
                   span_mask: Optional[torch.Tensor] = None,
                   type_mask: Optional[torch.Tensor] = None,
                   **kwargs: Any) -> torch.Tensor:
        device = span_reprs.device
        batch_size: int = span_reprs.shape[0]

        # type_logits: torch.Tensor = self.mlp(span_reprs)  # [batch_size, num_spans, num_types]
        type_embs: torch.Tensor = self.type_embeddings(torch.arange(self.num_types, device=device)).unsqueeze(
            dim=0).expand([batch_size, self.num_types, -1]).transpose(1, 2)  # [batch_size, embed_size, num_types]
        type_logits: torch.Tensor = self.mlp(span_reprs).matmul(type_embs)  # [batch_size, num_spans, num_types]

        type_mask: torch.Tensor = type_mask.view(batch_size, 1, -1)

        masked_type_logits: torch.Tensor = type_logits.where(type_mask, torch.tensor(-1000.0, device=device))

        # return masked_type_logits.log_softmax(dim=2)
        return masked_type_logits

    def decode_types(self, **kwargs: Any) -> torch.Tensor:
        type_logits: torch.Tensor = kwargs.get('type_logits')  # [batch_size, num_spans, num_types]
        types: torch.Tensor = type_logits.argmax(dim=2)  # [batch_size, num_spans]
        return types

    def output_types(self,
                     type_tensor: torch.Tensor,
                     span_mask: torch.Tensor) -> Tuple[List[List[int]], List[List[str]]]:
        type_ids: List[List[int]] = []
        type_strs: List[List[str]] = []
        for i in range(type_tensor.shape[0]):  # iterate over each instance within the batch
            ids: List[int] = type_tensor[i, span_mask[i]].tolist()
            type_ids.append(ids)
            type_strs.append([
                self.vocab.get_token_from_index(index=j, namespace=self.namespace)
                for j in ids
            ])

        return type_ids, type_strs

    def loss(self,
             preds: torch.Tensor,
             golds: torch.Tensor,
             **kwargs: Any) -> torch.Tensor:
        device = preds.device
        loss_reduction: bool = kwargs.get('loss_reduction', False)

        span_mask: torch.Tensor = kwargs['span_mask']
        batch_size, num_spans = preds.shape[0], preds.shape[1]

        flattened_type_logits: torch.Tensor = preds.view(-1,
                                                         self.num_types)  # [batch_size * num_spans, num_types]
        # unreduced_loss: torch.Tensor = self.nll(flattened_type_logits, golds.view(-1)).view_as(span_mask).where(
        #     span_mask,
        #     torch.zeros(1, device=device)
        # ).view(batch_size, num_spans)  # [batch_size, num_spans]

        # unreduced_loss: torch.Tensor = self.cross_entropy(flattened_type_logits, golds.view(-1)).view_as(
        #     span_mask).view(batch_size, num_spans)  # [batch_size, num_spans]
        return self.cross_entropy(flattened_type_logits, golds.view(-1))
        # if loss_reduction:
        #     # loss: torch.Tensor = unreduced_loss.logsumexp() / span_mask.sum()  # scalar
        #     loss: torch.Tensor = unreduced_loss.logsumexp(-1).logsumexp(-1) / span_mask.sum()  # scalar
        #     return loss
        # else:
        #     return unreduced_loss

    def forward(self,
                span_reprs: torch.Tensor,
                span_mask: Optional[torch.Tensor] = None,
                type_mask: Optional[torch.Tensor] = None,
                span_types: Optional[torch.Tensor] = None,
                **kwargs: Any) -> Dict[str, Any]:
        if type_mask is None:
            type_mask: torch.Tensor = torch.ones(1, dtype=torch.bool).expand(span_reprs.shape[0],
                                                                             self.num_types)  # [batch_size, num_types]
        if span_mask is None:
            span_mask: torch.Tensor = torch.ones(1, dtype=torch.bool).expand(
                span_reprs.shape[0],
                span_reprs.shape[1]
            )  # [batch_size, num_spans]

        # set_trace()

        type_logits: torch.Tensor = self.type_spans(span_reprs=span_reprs,
                                                    span_mask=span_mask,
                                                    type_mask=type_mask)  # [batch_size, num_spans, num_types]

        types: torch.Tensor = self.decode_types(type_logits=type_logits)  # [batch_size, num_spans]
        type_list: Tuple[List[List[int]], List[List[str]]] = self.output_types(type_tensor=types,
                                                                               span_mask=span_mask)

        res: Dict[str, torch.Tensor] = {
            'type_ids': type_list[0],
            'type_strs': type_list[1]
        }

        if span_types is not None:
            res['loss'] = self.loss(preds=type_logits,
                                    golds=span_types,
                                    span_mask=span_mask,
                                    kwargs=kwargs)

        return res

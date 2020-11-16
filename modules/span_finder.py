from typing import *

import torch
from torch import nn


class SpanFinder(nn.Module):
    def __init__(self,
                 input_dim: int):
        super(SpanFinder, self).__init__()
        self.input_dim: int = input_dim

    def forward(self,
                sequence_tensor: torch.Tensor,  # [batch_size, seq_len, embed_size]
                sequence_mask: torch.Tensor,  # [batch_size, seq_len]
                gold_span_indices: Optional[torch.Tensor] = None,
                gold_span_indices_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, Any]:  # [batch_size, num_spans, 2]
        # First element is reserved for tokens like [CLS], which is used as empty
        # Output a list of span indices (start, end) - left and right **inclusive**
        raise NotImplemented


class EnumSpanFinder(nn.Module):
    def __init__(self):
        raise NotImplemented

    def forward(self,
                sequence_tensor: torch.Tensor,  # [batch_size, seq_len, embed_size]
                sequence_mask: torch.Tensor,  # [batch_size, seq_len]
                span_indices: torch.Tensor,  # [batch_size, num_spans, 2]
                span
                ) -> Dict[str, Any]:
        span_mask = (span_indices[:, :, 0] >= 0).squeeze(-1).float()


class PredictiveSpanFinder(SpanFinder):
    def __init__(self,
                 input_dim: int):
        super(PredictiveSpanFinder, self).__init__(input_dim=input_dim)
        self.null_embed: nn.Parameter = nn.Parameter(
            torch.empty([self.input_dim]).data.normal_(0.0, 0.02).view(1, 1, -1),
            requires_grad=True
        )  # [1, 1, embed_size]
        self.start_affine: nn.Linear = nn.Linear(self.input_dim, 256)
        self.end_affine: nn.Linear = nn.Linear(self.input_dim, 256)
        # self.scoring: nn.Bilinear = nn.Bilinear(self.input_dim, self.input_dim, out_features=1)  # reduce to score
        self.scoring: nn.Sequential = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )  # reduce to score

        self.input_dropout: nn.Dropout = nn.Dropout(p=0.1)

        self.cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def score(self,
              input_reprs: torch.Tensor,  # [batch_size, seq_len, embed_size]
              input_mask: torch.Tensor,  # [batch_size, seq_len]
              ) -> torch.Tensor:  # [batch_size, seq_len, seq_len + 1]
        device = input_reprs.device
        batch_size, seq_len, embed_size = input_reprs.shape

        concatenated_sequence_reprs: torch.Tensor = self.input_dropout(torch.cat([
            self.null_embed.expand(batch_size, 1, -1),
            input_reprs
        ], dim=1))  # [batch_size, seq_len + 1, embed_size]
        start_matrix = self.start_affine(concatenated_sequence_reprs).unsqueeze(2).expand(
            [batch_size, seq_len + 1, seq_len + 1, -1])  # [batch_size, seq_len + 1, seq_len + 1, embed_size]
        end_matrix = self.end_affine(concatenated_sequence_reprs).unsqueeze(1).expand(
            [batch_size, seq_len + 1, seq_len + 1, -1])  # [batch_size, seq_len + 1, seq_len + 1, embed_size]
        # scores = self.scoring(start_matrix, end_matrix).squeeze(dim=3)  # [batch-size, seq_len + 1, seq_len + 1]

        # start_repr = self.start_affine(concatenated_sequence_reprs)  # [batch_size, seq_len + 1, embed_size]
        # end_repr = self.end_affine(concatenated_sequence_reprs)  # [batch_size, seq_len + 1, embed_size]
        scores = self.scoring(end_matrix - start_matrix).squeeze(dim=3)  # [batch-size, seq_len + 1, seq_len + 1]

        triu_mask_base = torch.ones([1, 1, 1], dtype=torch.bool, device=device).expand_as(scores)
        triu_mask_non_null = torch.triu(triu_mask_base)[:, 1:, 1:]
        triu_mask_null = triu_mask_base[:, 1:, 0].view(batch_size, -1, 1)
        triu_mask = torch.cat(
            [triu_mask_non_null, triu_mask_null],
            dim=2
        ) & input_mask.unsqueeze(-1).expand([batch_size, seq_len, seq_len + 1])  # [batch_size, seq_len, seq_len + 1]
        merged_scores = scores[:, 1:, :].where(
            triu_mask,
            torch.tensor(-1000., dtype=torch.float, device=device)
        )  # [batch_size, seq_len, seq_len + 1]

        return merged_scores

    def forward(self,
                sequence_tensor: torch.Tensor,  # [batch_size, seq_len, embed_size]
                sequence_mask: torch.Tensor,  # [batch_size, seq_len]
                gold_span_indices: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                gold_span_indices_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                **kwargs) -> Dict[str, Any]:  # [batch_size, num_spans, 2]
        device = sequence_tensor.device
        batch_size, seq_len, embed_size = sequence_tensor.shape

        scores = self.score(input_reprs=sequence_tensor,
                            input_mask=sequence_mask)  # [batch_size, seq_len, seq_len + 1]

        # decode span indices from scores
        end_index = scores.argmax(dim=2).unsqueeze(-1)  # [batch_size, seq_len, 1]
        start_index = torch.arange(1, seq_len + 1, dtype=torch.long, device=device).unsqueeze(dim=0).unsqueeze(
            -1).expand(batch_size, seq_len, 1)  # [batch_size, seq_len, 1]
        span_indices = torch.cat([start_index, end_index], dim=2) - 1  # [batch_size, seq_len, 2]
        valid_span = start_index <= end_index  # [batch_size, seq_len, 1]
        valid_span_indices = span_indices.where(
            valid_span.expand_as(span_indices),
            torch.zeros(1, dtype=torch.long, device=device)
        )  # [batch_size, seq_len, 2]

        num_valid_spans: List[int] = valid_span.squeeze(dim=-1).sum(dim=1).tolist()
        ret: Dict[str, Any] = {
            'span_indices': [
                list(filter(lambda t: t[0] == 0 and t[1] == 0, spans))
                for spans in valid_span_indices.tolist()
            ],
            'num_valid_spans': num_valid_spans
        }

        # compute loss, if goldens are given
        if gold_span_indices is not None:
            flattened_scores = scores.view(-1, seq_len + 1)  # [batch_size * seq_len, seq_len + 1]
            flattened_gold = gold_span_indices.view(-1)  # [batch_size * seq_len]
            unreduced_loss = self.cross_entropy(flattened_scores, flattened_gold)
            masked_unreduced_loss = unreduced_loss.where(
                gold_span_indices_mask.view(-1),
                torch.zeros(1, dtype=torch.float, device=device)
            )  # [batch_size * seq_len]
            loss = masked_unreduced_loss.sum() / gold_span_indices_mask.sum()

            ret['loss'] = loss

        return ret

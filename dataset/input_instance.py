from typing import *

import torch


class InputInstance(object):
    def __init__(self,
                 id: torch.Tensor,  # [batch_size]
                 span_indices: torch.Tensor,  # [batch_size, num_spans, 2]
                 event_type: torch.Tensor,  # [batch_size]
                 sequence_tensor: Optional[torch.Tensor] = None,  # [batch_size, seq_len, embed_dim]
                 span_indices_mask: Optional[torch.Tensor] = None,  # [batch_size, num_spans]
                 type_mask: Optional[torch.Tensor] = None,  # [batch_size, num_types]
                 span_types: Optional[torch.Tensor] = None,  # [batch_size, num_spans]
                 sequence_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                 gold_span_indices: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                 gold_span_indices_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                 ):
        self.id: torch.Tensor = id  # [batch_size]
        self.sequence_tensor: Optional[torch.Tensor] = sequence_tensor  # [batch_size, seq_len, embed_dim]
        self.span_indices: torch.Tensor = span_indices  # [batch_size, num_spans, 2]
        self.event_type: torch.Tensor = event_type  # [batch_size]
        self.span_indices_mask: Optional[torch.Tensor] = span_indices_mask  # [batch_size, num_spans]
        self.type_mask: Optional[torch.Tensor] = type_mask  # [batch_size, num_types]
        self.span_types: Optional[torch.Tensor] = span_types  # [batch_size, num_spans]
        self.sequence_mask: Optional[torch.Tensor] = sequence_mask  # [batch_size, seq_len]
        self.gold_span_indices: Optional[torch.Tensor] = gold_span_indices  # [batch_size, seq_len]
        self.gold_span_indices_mask: Optional[torch.Tensor] = gold_span_indices_mask  # [batch_size, seq_len]

    @classmethod
    def from_ins(cls, ins: 'InputInstance') -> 'InputInstance':
        return cls(**ins.__dict__)

    def to_device(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        # return {
        #     'sequence_tensor': self.sequence_tensor.to(device) if self.sequence_tensor is not None else None,
        #     'span_indices': self.span_indices.to(device),
        #     'span_indices_mask': self.span_indices_mask.to(device) if self.span_indices_mask is not None else None,
        #     'type_mask': self.type_mask.to(device) if self.type_mask is not None else None,
        #     'span_types': self.span_types.to(device) if self.span_types is not None else None,
        #     'sequence_mask': self.sequence_mask.to(device) if self.sequence_mask is not None else None,
        #     'event_type': self.event_type.to(device),
        #     'gold_span_indices': self.gold_span_indices.to(device) if self.gold_span_indices is not None else None,
        #     'gold_span_indices_mask': self.gold_span_indices_mask.to(
        #         device) if self.gold_span_indices_mask is not None else None
        # }
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else None
            for k, v in self.__dict__.items()
        }

    @classmethod
    def collate(cls, batch: List['InputInstance']) -> 'InputInstance':
        def _check_none(lst: List[Any], key: Callable) -> bool:
            for i in lst:
                if key(i) is None:
                    return True
            return False

        max_seq_len: int = max([ins.sequence_tensor.shape[1] for ins in batch])
        # max_num_spans: int = max([ins.span_indices.shape[1] for ins in batch])

        id = torch.cat([ins.id for ins in batch], dim=0)
        sequence_tensor = torch.cat([
            torch.cat([
                ins.sequence_tensor,
                torch.zeros([1, max_seq_len - ins.sequence_tensor.shape[1], ins.sequence_tensor.shape[2]],
                            dtype=torch.float)
            ], dim=1)
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.sequence_tensor) else None
        span_indices = torch.cat([
            # torch.cat([
            #     ins.span_indices,
            #     torch.zeros([1, max_num_spans - ins.span_indices.shape[1], 2], dtype=torch.long)
            # ], dim=1)
            ins.span_indices
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.span_indices) else None
        span_indices_mask = torch.cat([
            # torch.cat([
            #     torch.ones([1, ins.span_indices.shape[1]], dtype=torch.bool),
            #     torch.zeros([1, max_num_spans - ins.span_indices.shape[1]], dtype=torch.bool)
            # ], dim=1)
            ins.span_indices_mask
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.span_indices_mask) else None
        type_mask = torch.cat([
            ins.type_mask
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.type_mask) else None
        span_types = torch.cat([
            # torch.cat([
            #     ins.span_types,
            #     torch.zeros([1, max_num_spans - ins.span_types.shape[1]], dtype=torch.long)
            # ], dim=1)
            ins.span_types
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.span_types) else None
        sequence_mask = torch.cat([
            torch.cat([
                torch.ones([1, ins.sequence_tensor.shape[1]], dtype=torch.bool),
                torch.zeros([1, max_seq_len - ins.sequence_tensor.shape[1]], dtype=torch.bool)
            ], dim=1)
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.sequence_tensor) else None
        event_type = torch.cat([
            ins.event_type
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.event_type) else None
        gold_span_indices = torch.cat([
            torch.cat([
                ins.gold_span_indices,
                torch.zeros([1, max_seq_len - ins.gold_span_indices.shape[1]], dtype=torch.long)
            ], dim=1)
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.gold_span_indices) else None  # [batch_size, seq_len]
        gold_span_indices_mask = torch.cat([
            torch.cat([
                ins.gold_span_indices_mask,
                torch.zeros([1, max_seq_len - ins.gold_span_indices_mask.shape[1]], dtype=torch.bool)
            ], dim=1)
            for ins in batch
        ], dim=0) if not _check_none(batch, key=lambda t: t.gold_span_indices_mask) else None  # [batch_size, seq_len]

        return InputInstance(
            id=id,
            sequence_tensor=sequence_tensor,
            sequence_mask=sequence_mask,
            span_indices=span_indices,
            span_indices_mask=span_indices_mask,
            type_mask=type_mask,
            span_types=span_types,
            event_type=event_type,
            gold_span_indices=gold_span_indices,
            gold_span_indices_mask=gold_span_indices_mask
        )

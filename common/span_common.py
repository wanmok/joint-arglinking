"""
Various span related data structures
@author: Yunmo Chen (ychen@jhu.edu)
@date: 03/24/2020
"""

from typing import *

import numpy as np

from cement.cement_utils import global_to_local_indices


class Span(object):
    def __init__(self,
                 start: int,
                 end: int,
                 score: Optional[float] = None,
                 document: Optional['Document'] = None):
        # left and right inclusive
        self.start: int = start
        self.end: int = end
        self.score: Optional[float] = score
        self.document: Optional['Document'] = document

    def to_local_indices(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        assert self.document, 'This span is not associated with any document.'
        local_start: Tuple[int, int] = self.document.global_to_local(self.start)
        local_end: Tuple[int, int] = self.document.global_to_local(self.end)
        return local_start, local_end

    def get_tokens(self) -> List[str]:
        return self.document[self.start:self.end + 1]

    def to_slice(self) -> slice:
        return slice(self.start, self.end, None)

    def to_text(self) -> str:
        assert self.document, 'This span is not associated with any document.'
        return ' '.join(self.document[self.start:self.end + 1])

    def __str__(self) -> str:
        if self.document:
            return f'{self.to_text()} ({self.start}, {self.end})'
        return f'({self.start}, {self.end})'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: 'Span') -> bool:
        if isinstance(other, Span):
            raise TypeError(f'Invalid argument type {type(other)}.')

        if (
                (self.document is not None and other.document is not None)
                or (self.document is None and other.document is None)
        ):
            if self.document.doc_key == other.document.doc_key:
                return (self.start, self.end) == (other.start, other.end)

        return False


class Trigger(Span):
    def __str__(self) -> str:
        return super().__str__()


class Argument(Span):
    def __init__(self,
                 start: int,
                 end: int,
                 role: Optional[str] = None,
                 document: Optional['Document'] = None):
        super().__init__(start=start,
                         end=end,
                         document=document)
        self.role: Optional[str] = role

    def __str__(self) -> str:
        return (f'{self.role}: ' if self.role else '') + super().__str__()


class BaseSituationMention(object):
    def __init__(self,
                 kind: Optional[str] = None,
                 arguments: Optional[List[Argument]] = None,
                 document: Optional['Document'] = None):
        self.kind: Optional[str] = kind
        self.arguments: Optional[List[Argument]] = arguments
        self.document: Optional['Document'] = document


class Event(BaseSituationMention):
    def __init__(self,
                 kind: Optional[str] = None,
                 trigger: Optional[Union[Trigger, Span]] = None,
                 arguments: Optional[List[Argument]] = None,
                 document: Optional['Document'] = None):
        super().__init__(kind=kind,
                         arguments=arguments,
                         document=document)
        self.trigger: Optional[Union[Trigger, Span]] = trigger

    def find_arg_by_indices(self, indices: Tuple[int, int]) -> Optional[Argument]:
        for arg in self.arguments:
            if (arg.start, arg.end) == indices:
                return arg
        return None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        args: str = "\n\t\t".join([str(a) for a in self.arguments])
        return f'EVENT - kind: {self.kind}\n\t\ttrigger: {self.trigger}\n\t\targs:\n\t\t{args}'


class Document(object):
    def __init__(self,
                 sentences: List[List[str]],
                 events: List[Event],
                 doc_key: Optional[str] = None):
        self.doc_key: Optional[str] = doc_key
        self.events: List[Event] = events
        self.sentences: List[List[str]] = sentences
        self.sentence_offsets: np.ndarray = self._count_offsets(self.sentences)
        self.tokens: List[str] = self._flatten_sentences(sentences)

    def __getitem__(self, item):
        if isinstance(item, Span):
            return self.tokens[item.start:item.end + 1]
        return self.tokens[item]

    def global_to_local(self, global_idx: int) -> Tuple[int, int]:
        # sent_id: int = np.digitize(global_idx, self.sentence_offsets).item() - 1
        # return sent_id, global_idx - self.sentence_offsets[sent_id]
        sent_ids, global_indices = global_to_local_indices(indices=[global_idx], bins=self.sentence_offsets[1:])
        return sent_ids.item(), global_indices.item()

    def local_to_global(self, sent_id: int, local_idx: int) -> int:
        return self.sentence_offsets[sent_id] + local_idx

    def global_to_local_spans(self, spans: Union[List, np.ndarray]) -> Tuple[List[int], List[Tuple[int, int]]]:
        sent_ids, global_indices = global_to_local_indices(indices=spans, bins=self.sentence_offsets[1:])
        sent_ids = [t[0] for t in sent_ids.tolist()]
        return sent_ids, [tuple(x) for x in global_indices.tolist()]

    @staticmethod
    def _flatten_sentences(sentences: List[List[str]]) -> List[str]:
        tokens: List[str] = []
        for sent in sentences:
            tokens.extend(sent)
        return tokens

    @staticmethod
    def _count_offsets(sentences: List[List[str]]) -> np.ndarray:
        counts: List[int] = [0]
        for sent in sentences:
            counts.append(len(sent))
        return np.cumsum(counts)

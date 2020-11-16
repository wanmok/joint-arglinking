import json
import logging
import os
from collections import defaultdict
from itertools import groupby
from typing import *

import h5py
import torch
from allennlp.data import Vocabulary
from concrete import EntityMention, Communication
from concrete.util import read_communication_from_file, get_comm_tokenizations
from torch.utils.data import Dataset
from tqdm import tqdm

from common.span_common import Document, Event, Argument, Trigger, Span
from dataset.input_instance import InputInstance

logger = logging.getLogger(__name__)


def from_concrete_file(comm_file: str,
                       task: str = 'argidcls') -> Document:
    def _entity_mention_to_span_indices(em: EntityMention) -> Tuple[int, int]:
        sentid: int = tok_to_sentid[em.tokens.tokenizationId.uuidString]
        start: int = doc.local_to_global(sent_id=sentid, local_idx=em.tokens.tokenIndexList[0])
        end: int = doc.local_to_global(sent_id=sentid, local_idx=em.tokens.tokenIndexList[-1])
        return start, end

    def _normalize_token(t: str) -> str:
        # For ACE dataset
        if t == '\'\'':
            return '"'
        elif t == '``':
            return '"'
        elif t == '-LRB-':
            return '('
        elif t == '-RRB-':
            return ')'
        elif t == '-LSB-':
            return '['
        elif t == '-RSB-':
            return ']'
        elif t == '-LCB-':
            return '{'
        elif t == '-RCB-':
            return '}'
        else:
            return t

    def _normalize_role(r: str) -> str:
        if 'Time' in r:
            return 'Time'
        else:
            return r

    comm: Communication = read_communication_from_file(comm_file)
    tok_to_sentid: Dict[str, int] = {}
    sentences: List[List[str]] = []
    # extract tokens to form sentences
    for sent_id, tok in enumerate(get_comm_tokenizations(comm)):
        tok_to_sentid[tok.uuid.uuidString] = sent_id
        sentences.append([
            _normalize_token(t.text)
            for t in tok.tokenList.tokenList
        ])
    doc: Document = Document(doc_key=str(comm.id),
                             events=[],
                             sentences=sentences)

    # convert SituationMention into Event objects
    for sm in comm.situationMentionSetList[0].mentionList:
        if sm.situationType != 'EVENT':
            continue
        event: Event = Event(document=doc,
                             kind=sm.situationKind,
                             arguments=[])
        for arg in sm.argumentList:
            if arg.entityMentionId is not None:
                arg_entity_mention = comm.entityMentionForUUID[arg.entityMentionId.uuidString]
            elif arg.situationMentionId is not None:
                arg_entity_mention = comm.situationMentionForUUID[arg.situationMentionId.uuidString]
            else:
                raise ValueError
            start_idx, end_idx = _entity_mention_to_span_indices(em=arg_entity_mention)
            if arg.role == 'TRIGGER':
                event.trigger = Trigger(start=start_idx,
                                        end=end_idx,
                                        document=doc)
            else:
                event.arguments.append(
                    Argument(start=start_idx,
                             end=end_idx,
                             role=_normalize_role(arg.role),  # TODO(Yunmo): Ensure that there is only one Time
                             document=doc)
                )
        if event.trigger is None:
            if sm.tokens is None:
                start_idx, end_idx = (0, len(doc.to) - 1)
            else:
                start_idx, end_idx = _entity_mention_to_span_indices(sm)
            event.trigger = Trigger(start=start_idx, end=end_idx, document=doc)
        doc.events.append(event)

    doc.argument_mentions: List[Span] = []
    if task == 'argidcls-noisy':
        # add all possible for `argidcls`
        for em in comm.entityMentionSetList[0].mentionList:
            start_idx, end_idx = _entity_mention_to_span_indices(em=em)
            doc.argument_mentions.append(Span(start=start_idx,
                                              end=end_idx,
                                              document=doc))
    elif task == 'argcls' or task == 'argidcls':
        for event in doc.events:
            for arg in event.arguments:
                doc.argument_mentions.append(Span(start=arg.start,
                                                  end=arg.end,
                                                  document=doc))
    # else:
    #     raise NotImplemented

    return doc


class ConcreteDataset(Dataset):
    def __init__(self,
                 docs: List[Document],
                 ins_to_event: Dict[int, Event],
                 vocab: Vocabulary,
                 cache_file: h5py.File,
                 role_type_mask: torch.Tensor,
                 num_events: int,
                 num_roles: int,
                 instances: List[InputInstance],
                 metadata: Dict[int, Dict[str, Any]],
                 max_num_spans: int = 512,
                 sentence_mode: bool = False):
        self.sentence_mode = sentence_mode
        self.docs: List[Document] = docs
        self.ins_to_event: Dict[int, Event] = ins_to_event
        self.vocab: Vocabulary = vocab
        self.cache_file: h5py.File = cache_file
        self.role_type_mask: torch.Tensor = role_type_mask
        self.num_events: int = num_events
        self.num_roles: int = num_roles
        self.max_num_spans: int = max_num_spans
        self.instances: List[InputInstance] = instances
        self.metadata: Dict[int, Dict[str, Any]] = metadata

    @staticmethod
    def load_documents_from_concrete_dir(dir: str, task: str = 'argidcls') -> List[Document]:
        docs: List[Document] = []
        for fn in os.listdir(dir):
            if '.concrete' not in fn and '.comm' not in fn:
                continue

            # convert Concrete file to Document object
            docs.append(from_concrete_file(comm_file=os.path.join(dir, fn), task=task))

        return docs

    def _load_cache(self, doc_key: str, sent_id: int = -1) -> torch.Tensor:
        if sent_id == -1:
            return torch.cat(
                [
                    torch.tensor(self.cache_file[doc_key][k][:, :, -1])
                    if len(self.cache_file[doc_key][k].shape) == 3 else
                    torch.tensor(self.cache_file[doc_key][k])
                    for k in self.cache_file[doc_key]
                ],
                dim=0
            )  # [seq_len, embed_size]
        else:
            return (torch.tensor(self.cache_file[doc_key][str(sent_id)][:, :, -1])
                    if len(self.cache_file[doc_key][str(sent_id)].shape) == 3 else
                    torch.tensor(self.cache_file[doc_key][str(sent_id)]))

    @staticmethod
    def build_ontology_and_vocab(ontology_path: str, vocab_path: Optional[str] = None) -> Tuple[Vocabulary, Dict]:
        with open(ontology_path) as f:
            ontology = json.load(f)

        if vocab_path is None:
            vocab: Vocabulary = Vocabulary()
            vocab.add_token_to_namespace(token='None', namespace='span_labels')
            vocab.add_token_to_namespace(token='@@PADDING@@', namespace='span_labels')
            vocab.add_tokens_to_namespace([
                role
                for role in ontology['args'].keys()
            ], namespace='span_labels')
            vocab.add_tokens_to_namespace([
                event
                for event in ontology['events'].keys()
            ], namespace='event_labels')
        else:
            vocab: Vocabulary = Vocabulary.from_files(vocab_path)

        return vocab, ontology

    @classmethod
    def from_concrete(cls,
                      data_path: str,
                      cache_file: str,
                      vocab: Vocabulary,
                      ontology: Dict,
                      max_num_spans: int = 512,
                      task: str = 'argidcls',
                      sentence_mode: bool = False) -> 'ConcreteDataset':
        def _build_role_type_mask(vocab: Vocabulary) -> torch.Tensor:
            role_type_mask_list: List[List[int]] = []
            all_role_types: List[str] = [
                r
                for r, _ in sorted(vocab.get_token_to_index_vocabulary(namespace='span_labels').items(),
                                   key=lambda t: t[1])
            ]
            for event_type, _ in sorted(vocab.get_token_to_index_vocabulary(namespace='event_labels').items(),
                                        key=lambda t: t[1]):
                role_type_mask_list.append([
                    (
                        1
                        if k in ontology['events'][event_type]['roles'].keys() or (
                                i == 0 and task in ['argidcls', 'argidcls_noisy'])
                        else 0
                    )
                    for i, k in enumerate(all_role_types)
                ])

            return torch.tensor(role_type_mask_list, dtype=torch.bool)  # [num_events, num_roles]

        def _to_predictive_span_finder_gold(spans: List[Span]) -> Tuple[torch.Tensor, torch.Tensor]:
            sorted_spans: List[Span] = sorted(spans, key=lambda s: s.start)
            gold: List[int] = [0 for _ in range((sorted_spans[-1].start + 1) if len(sorted_spans) > 0 else 0)]
            gold_mask: List[int] = [0 for _ in range((sorted_spans[-1].start + 1) if len(sorted_spans) > 0 else 0)]
            for span in sorted_spans:
                gold[span.start] = span.end + 1  # shift one for null span
                gold_mask[span.start] = 1

            return (
                torch.tensor(gold, dtype=torch.long),
                torch.tensor(gold_mask, dtype=torch.bool)
            )

        def _tensorize_spans(evnt: Event) -> Tuple[torch.Tensor, torch.Tensor]:
            span_indices: List[Tuple[int, int]] = [(evnt.trigger.start, evnt.trigger.end)]
            span_types: List[int] = [vocab.get_token_index(token='None', namespace='span_labels')]
            if task in ['argidcls', 'argidcls-noisy']:
                mention_list = evnt.document.argument_mentions
            else:
                mention_list = evnt.arguments
            for mention in mention_list:
                if (mention.start, mention.end) == (evnt.trigger.start, evnt.trigger.end):
                    continue
                span_indices.append((mention.start, mention.end))
                arg: Optional[Argument] = evnt.find_arg_by_indices(indices=(mention.start, mention.end))
                span_types.append(
                    vocab.get_token_index(token='None', namespace='span_labels')
                    if arg is None else vocab.get_token_index(token=arg.role, namespace='span_labels')
                )
            return (
                torch.tensor(span_indices, dtype=torch.long).view([1, -1, 2]),
                torch.tensor(span_types, dtype=torch.long).view(1, -1)
            )

        def _tensorize_spans_sentence_level(
                evnt: Event,
                grouped_mentions: Optional[Dict[int, List[Tuple[Tuple[int, int], Span]]]]
        ) -> Tuple[int, torch.Tensor, torch.Tensor]:
            span_types: List[int] = [vocab.get_token_index(token='None', namespace='span_labels')]
            trigger_sent_ids, trigger_indices = evnt.document.global_to_local_spans(
                spans=[(evnt.trigger.start, evnt.trigger.end)]
            )
            trigger_sent_id = trigger_sent_ids[0]
            trigger_indices = trigger_indices[0]
            spans: List[Tuple[int, int]] = [trigger_indices]
            if task in ['argidcls', 'argidcls-noisy']:
                span_list = grouped_mentions[trigger_sent_id]
            else:  # argcls
                arg_sent_ids, arg_indices = evnt.document.global_to_local_spans(
                    spans=[(arg.start, arg.end) for arg in evnt.arguments]
                )
                span_list = [(t, evnt.arguments[i]) for i, t in enumerate(arg_indices)]
            for t, s in span_list:
                arg: Optional[Argument] = evnt.find_arg_by_indices(indices=(s.start, s.end))
                spans.append(t)
                span_types.append(
                    vocab.get_token_index(token='None', namespace='span_labels')
                    if arg is None else vocab.get_token_index(token=arg.role, namespace='span_labels')
                )
            return (
                trigger_sent_id,
                torch.tensor(spans, dtype=torch.long).view([1, -1, 2]),
                torch.tensor(span_types, dtype=torch.long).view(1, -1)
            )

        def _group_mentions_by_sentence(d: Document) -> Dict[int, List[Tuple[Tuple[int, int], Span]]]:
            if task == 'argidcls-noisy':
                mention_list = d.argument_mentions
            elif task == 'argidcls':
                mention_list = []
                for e in doc.events:
                    mention_list.extend(e.arguments)
            else:
                raise NotImplementedError
            sent_ids, span_indices = d.global_to_local_spans(spans=[(s.start, s.end) for s in mention_list])
            grouped_spans = defaultdict(list)
            for sent_id, t_list in groupby(zip(sent_ids, span_indices, mention_list), key=lambda k: k[0]):
                ss = [(t[1], t[2]) for t in t_list]
                grouped_spans[sent_id].extend(ss)
            return grouped_spans

        docs: List[Document] = cls.load_documents_from_concrete_dir(dir=data_path,
                                                                    task=task)
        cache: h5py.File = h5py.File(cache_file, mode='r')
        role_type_mask: torch.Tensor = _build_role_type_mask(vocab=vocab)  # [num_events, num_roles]
        num_events: int = len(vocab.get_token_to_index_vocabulary(namespace='event_labels'))
        num_roles: int = len(vocab.get_token_to_index_vocabulary(namespace='span_labels'))
        ins_to_event: Dict[int, Event] = {}
        instances: List[InputInstance] = []
        padding_tensor: torch.Tensor = torch.tensor(
            [[vocab.get_token_index(token='@@PADDING@@', namespace='span_labels')]],
            dtype=torch.long
        )  # [1, 1]

        metadata: Dict[int, Dict[str, Any]] = {}
        id: int = 0
        for doc in tqdm(docs):
            # sequence_tensor: torch.Tensor = _load_cache(doc.doc_key)
            if sentence_mode and task in ['argidcls', 'argidcls-noisy']:
                grouped_mentions: Optional[Dict[int, List[Tuple[Tuple[int, int], Span]]]] = _group_mentions_by_sentence(
                    doc)
            else:
                grouped_mentions = None
            for e in doc.events:  # but RAMS only has one event per document
                ins_metadata = {}
                if sentence_mode:
                    sent_id, span_indices, span_types = _tensorize_spans_sentence_level(
                        evnt=e,
                        grouped_mentions=grouped_mentions
                    )
                    ins_metadata['sentence_id'] = sent_id
                else:
                    span_indices, span_types = _tensorize_spans(evnt=e)
                if task == 'emd':
                    gold_span_indices, gold_span_indices_mask = _to_predictive_span_finder_gold([
                        arg for arg in e.arguments
                    ])
                if span_indices.shape[1] == 1:
                    logger.info('Example has no arguments.')
                    continue

                new_ins: InputInstance = InputInstance(
                    id=torch.tensor([id], dtype=torch.long),
                    # sequence_tensor=sequence_tensor.view(1, sequence_tensor.shape[0], sequence_tensor.shape[1]),
                    event_type=torch.tensor([vocab.get_token_index(token=e.kind, namespace='event_labels')],
                                            dtype=torch.long),
                    span_indices=torch.cat([
                        span_indices,
                        torch.zeros([1, max_num_spans - span_indices.shape[1], 2], dtype=torch.long)
                    ], dim=1) if task != 'emd' else None,
                    span_indices_mask=torch.cat([
                        torch.ones([1, span_indices.shape[1]], dtype=torch.bool),
                        torch.zeros([1, max_num_spans - span_indices.shape[1]], dtype=torch.bool)
                    ], dim=1) if task != 'emd' else None,
                    type_mask=role_type_mask[
                              vocab.get_token_index(e.kind, namespace='event_labels'), :
                              ].view(1, -1) if task != 'emd' else None,
                    span_types=torch.cat([
                        span_types,
                        padding_tensor.expand([1, max_num_spans - span_indices.shape[1]])
                    ], dim=1) if task != 'emd' else None,
                    gold_span_indices=gold_span_indices.view(1, -1) if task == 'emd' else None,
                    gold_span_indices_mask=gold_span_indices_mask.view(1, -1) if task == 'emd' else None
                )
                instances.append(new_ins)
                ins_to_event[id] = e
                metadata[id] = ins_metadata
                id += 1

        return cls(docs=docs,
                   ins_to_event=ins_to_event,
                   vocab=vocab,
                   cache_file=cache,
                   role_type_mask=role_type_mask,
                   num_events=num_events,
                   num_roles=num_roles,
                   instances=instances,
                   max_num_spans=max_num_spans,
                   sentence_mode=sentence_mode,
                   metadata=metadata)

    def __getitem__(self, item):
        ins: InputInstance = self.instances[item]
        if self.sentence_mode:
            sequence_tensor = self._load_cache(doc_key=self.ins_to_event[ins.id.item()].document.doc_key,
                                               sent_id=self.metadata[ins.id.item()]['sentence_id'])
        else:
            sequence_tensor: torch.Tensor = self._load_cache(doc_key=self.ins_to_event[ins.id.item()].document.doc_key)
        new_ins: InputInstance = InputInstance.from_ins(ins=ins)
        new_ins.sequence_tensor = sequence_tensor.view(1, sequence_tensor.shape[0], sequence_tensor.shape[1])
        return new_ins

    def __len__(self):
        return len(self.instances)


if __name__ == '__main__':
    from pdb import set_trace

    vocab, ontology = ConcreteDataset.build_ontology_and_vocab(
        ontology_path='out/buildACEOntology/Baseline.baseline/ace-ontology.json',
        vocab_path=None
    )
    dataset = ConcreteDataset.from_concrete(
        data_path='out/createACEConcreteFiles/Baseline.baseline/out/ace-events/train',
        cache_file='out/aceBERTCache/ACEDatasetV.events/out/train.hdf5',
        vocab=vocab,
        ontology=ontology,
        task='argidcls',
        max_num_spans=30,
        sentence_mode=True)

    set_trace()

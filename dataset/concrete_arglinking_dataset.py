import json
import logging
from itertools import groupby
from typing import *

import h5py
import numpy as np

from concrete import SituationMention
from overrides import overrides
from torch.utils.data import Dataset

from cement.cement_document import CementDocument
from cement.cement_entity_mention import CementEntityMention
from cement.cement_span import CementSpan

logger = logging.getLogger(__name__)


class ConcreteArglinkingDataset(Dataset):
    def __init__(self,
                 cache_files: Dict[str, str],
                 ontology_path: str,
                 max_span_width: int = -1,
                 skip_first_sentences: int = 0,
                 gold_mentions_only: bool = True,
                 sentence_mode: bool = True,
                 lazy: bool = False):
        self._gold_mentions_only = gold_mentions_only
        self._skip_first_sentences = skip_first_sentences
        self._sentence_mode = sentence_mode
        self._max_span_width: int = max_span_width

        if not sentence_mode and self._skip_first_sentences > 0:
            logger.warning(f'Not in sentence mode, but skip_first_sentences={self._skip_first_sentences}')

        self._ontology: Dict[str, Dict[str, List[str]]] = self.build_ontology(ontology_path=ontology_path)

        self._cache_handlers = {
            k: h5py.File(v, mode='r')
            for k, v in cache_files.items()
        }

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_names: List[str] = os.listdir(file_path)
        for fn in file_names:
            if '.concrete' not in fn and '.comm' not in fn:
                continue
            doc: CementDocument = CementDocument.from_communication_file(os.path.join(file_path, fn))
            yield from self.extract_instances_from_doc(doc=doc, cache_target=file_path.split('/')[-1])

    def _access_cache_by_key(self, key: List[str], target: str = 'train') -> Union[h5py.Group, np.ndarray]:
        t = self._cache_handlers[target]['/'.join(key)]
        return t if isinstance(t, h5py.Group) else np.array(t)

    def _read_context_array_from_cache(self, key: List[str], target: str = 'train') -> np.ndarray:
        return np.concatenate([
            np.array(self._access_cache_by_key(key, target)[str(i)])
            for i in range(len(self._access_cache_by_key(key=key, target=target)))
        ], axis=0)

    def build_ontology(self, ontology_path: str) -> Dict[str, Dict[str, List[str]]]:
        with open(ontology_path) as f:
            ontology = json.load(f)
        return {
            'events': {
                event_type:
                    (
                        [] if self._gold_mentions_only else ['None']
                    ) + [role for role in event['roles'].keys()]  # add `None` for none argument spans
                for event_type, event in ontology['events'].items()
            }
        }

    @overrides
    def extract_instances_from_doc(self,
                                   doc: CementDocument,
                                   cache_target: Optional[str] = None) -> Iterable[Instance]:
        def _iterate_mention_arguments_as_spans(
                sms: List[SituationMention]
        ) -> Iterable[Union[CementSpan, CementEntityMention]]:
            for sm in sms:
                for arg in sm.argumentList:
                    if arg.situationMentionId is not None:
                        ref_sm: SituationMention = doc.comm.situationMentionForUUID[arg.situationMentionId.uuidString]
                        span = CementSpan.from_token_ref_sequence(ref_sm.tokens, document=doc)
                    elif arg.tokens is not None:
                        span = CementSpan.from_token_ref_sequence(arg.tokens, document=doc)
                    elif arg.entityMentionId is not None:
                        span = CementEntityMention.from_entity_mention(
                            mention=doc.comm.entityMentionForUUID[arg.entityMentionId.uuidString],
                            document=doc
                        )
                    else:
                        logger.info(f'MentionArgument={arg} - does not have any span information.')
                        continue
                    span.attrs.add('role', arg.role)
                    yield span

        entity_mentions: List[CementEntityMention] = [
            CementEntityMention.from_entity_mention(mention, doc)
            for mention in doc.iterate_entity_mentions()
        ]
        event_mentions: List[SituationMention] = list(doc.iterate_event_mentions())

        if self._sentence_mode:
            local_mention_indices: Dict[Tuple[int, int], Union[CementSpan, CementEntityMention]] = {}
            for em in entity_mentions:
                sent_ids, indices = zip(*em.to_local_indices())
                if sent_ids[0] != sent_ids[1]:
                    logger.info(f'Mention span crosses sentence boundary: {sent_ids}')
                    continue
                else:
                    local_mention_indices[(indices[0], indices[-1])] = em
            sent_to_ems: Dict[int, Dict[Tuple[int, int], Union[CementSpan, CementEntityMention]]] = {
                sent_id: {k: v for k, v in mention_group}
                for sent_id, mention_group in groupby(local_mention_indices.items(), key=lambda t: t[0][0])
            }
            for event in event_mentions:
                # if this event has a trigger
                assert event.tokens is not None, 'This event does not have a trigger'
                trigger_span = CementSpan.from_token_ref_sequence(event.tokens, document=doc)
                sent_id, (trigger_start, trigger_end) = trigger_span.to_local_indices()
                sequence_array: np.ndarray = self._access_cache_by_key(key=[doc.comm.id, str(sent_id)],
                                                                       target=cache_target)
                sequence = [doc.get_sentence(sent_id=sent_id)]
                event_type: str = event.situationKind
                argument_mentions: Dict[Tuple[int, int], Tuple[str, Union[CementSpan, CementEntityMention]]] = {}
                for arg in _iterate_mention_arguments_as_spans([event]):
                    sid, arg_span = arg.to_local_indices()
                    assert sid == sent_id, f'Arguments cross sentences - cannot process with sentence mode.'
                    argument_mentions[arg_span] = (arg.attrs.role, arg)
                if not self._gold_mentions_only:
                    for em_indices, em in sent_to_ems[sent_id].items():
                        if em_indices not in argument_mentions:
                            argument_mentions[em_indices] = ('None', em)

                yield self.text_to_instance(doc=doc,
                                            sequence=sequence,
                                            sequence_array=sequence_array,
                                            event_type=event_type,
                                            trigger=(trigger_start, trigger_end),
                                            mention_spans=argument_mentions)
        else:
            sequence_array: np.ndarray = self._read_context_array_from_cache(key=[doc.comm.id],
                                                                             target=cache_target)
            sequence = doc.iterate_sentences()
            for event in event_mentions:
                if event.tokens is not None:
                    trigger_span = CementSpan.from_token_ref_sequence(event.tokens, document=doc)
                else:  # use the whole document as the trigger
                    trigger_span = CementSpan(start=0, end=len(doc) - 1, document=doc)
                event_type: str = event.situationKind
                argument_mentions: Dict[Tuple[int, int], Tuple[str, Union[CementSpan, CementEntityMention]]] = {
                    arg.to_index_tuple(): (arg.attrs.role, arg)
                    for arg in _iterate_mention_arguments_as_spans([event])
                }
                if not self._gold_mentions_only:
                    for em in entity_mentions:
                        em_indices = em.to_index_tuple()
                        if em_indices not in argument_mentions:
                            argument_mentions[em_indices] = ('None', em)

                yield self.text_to_instance(doc=doc,
                                            sequence=sequence,
                                            sequence_array=sequence_array,
                                            event_type=event_type,
                                            trigger=trigger_span.to_index_tuple(),
                                            mention_spans=argument_mentions)

    @overrides
    def text_to_instance(self,
                         doc: CementDocument,
                         sequence: Iterable[List[str]],
                         sequence_array: np.ndarray,
                         event_type: str,
                         trigger: Tuple[int, int],
                         mention_spans: Dict[Tuple[int, int], Tuple[str, Union[CementSpan, CementEntityMention]]]
                         ) -> Instance:
        metadata = {
            'doc': doc,
            'mentions': mention_spans
        }

        sequence_field: ArraySequenceField = ArraySequenceField(sequence_array)
        spans: List[Field] = [SpanField(trigger[0], trigger[1], sequence_field)]  # put trigger span into the span list
        span_labels: List[str] = ['None']  # for trigger span

        if self._max_span_width == -1:  # turn off span enumeration
            for (start, end), (role, _) in mention_spans.items():
                spans.append(SpanField(start, end, sequence_field))
                span_labels.append(role)
        else:
            sentence_offset = 0
            for sentence in sequence:
                for start, end in enumerate_spans(sentence,
                                                  offset=sentence_offset,
                                                  max_span_width=self._max_span_width):
                    if (start, end) in mention_spans:
                        span_labels.append(mention_spans[(start, end)][0])  # use roles as span labels
                    else:
                        span_labels.append('None')

                    spans.append(SpanField(start, end, sequence_field))
                sentence_offset += len(sentence)

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        event_roles_field = ListField([
            LabelField(r, label_namespace='span_labels')
            for r in self._ontology['events'][event_type]
        ])
        event_type_field = LabelField(event_type, label_namespace='event_labels')

        fields: Dict[str, Field] = {'sequence': sequence_field,
                                    'spans': span_field,
                                    'event_type': event_type_field,
                                    'event_roles': event_roles_field,
                                    'metadata': metadata_field}
        if span_labels is not None:
            fields['span_labels'] = SequenceLabelField(span_labels, span_field, label_namespace='span_labels')

        return Instance(fields)


if __name__ == '__main__':
    dataset = ConcreteArglinkingDataset(max_span_width=-1,
                                        cache_files={
                                            'train': 'out/aidaBERTCache/Baseline.baseline/out/train.hdf5'
                                        },
                                        ontology_path='out/convertAIDAOnotology/Baseline.baseline/aida-ontology.json',
                                        sentence_mode=False,
                                        lazy=True)
    a = dataset.read('out/convertAIDAToConcrete/Baseline.baseline/out/train')
    # from allennlp.data import Batch
    import pdb

    for x in a:
        pdb.set_trace()
    pass

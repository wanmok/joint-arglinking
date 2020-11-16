import logging
import os
from typing import *
import argparse
from collections import Counter, defaultdict
from functools import reduce

from concrete import SituationMention

from cement.cement_document import CementDocument
from cement.cement_entity_mention import CementEntityMention
from cement.cement_span import CementSpan

logger = logging.getLogger(__name__)


def read_documents(dir_path: str) -> Iterable[CementDocument]:
    file_names: List[str] = os.listdir(dir_path)
    for fn in file_names:
        if '.concrete' not in fn and '.comm' not in fn:
            continue
        yield CementDocument.from_communication_file(file_path=os.path.join(dir_path, fn))


def stats_on_documents(doc_stream: Iterable[CementDocument]) -> NoReturn:
    raw_descriptions: List[Dict] = []
    for doc in doc_stream:
        raw_descriptions.append(collect_raw_document_data(doc=doc))

    # aggregate to dataset level stats
    dataset_counter = Counter()
    ner = set()
    event_types = set()
    event_roles: defaultdict = defaultdict(set)
    relation_types = set()
    for desc in raw_descriptions:
        ner.update(desc['ontology']['ner'])
        event_types.update(desc['ontology']['event_types'])
        relation_types.update(desc['ontology']['relation_types'])
        # event_roles.update(desc['ontology']['event_roles'])

        for k, v in desc['ontology']['event_roles'].items():
            event_roles[k].update(v)

        for k, v in desc['counter'].items():
            dataset_counter[k] += v

    # print results
    logger.info('Overall dataset split description')
    logger.info(f'# of documents: {len(raw_descriptions)}')
    logger.info(f'# of tokens: {dataset_counter["Tokens"]}')
    logger.info(f'# of sentences: {dataset_counter["Sentence"]}')
    logger.info(f'# of entity mentions: {dataset_counter["EntityMention"]}')
    logger.info(f'# of event mentions: {dataset_counter["EventMention"]}')
    logger.info(f'# of relation mentions: {dataset_counter["RelationMention"]}')
    logger.info(f'# of named entity types: {len(ner)}')
    logger.info(f'# of event types: {len(event_types)}')
    logger.info(f'# of relation types: {len(relation_types)}')
    logger.info('Training relevant statistics')
    logger.info(f'Entity mentions per token: {round(dataset_counter["EntityMention"] / dataset_counter["Tokens"], 4)}')
    logger.info(
        f'Event mentions per sentence: {round(dataset_counter["EventMention"] / dataset_counter["Sentence"], 4)}')
    logger.info(f'Relation mentions per sentence: '
                f'{round(dataset_counter["RelationMention"] / dataset_counter["Sentence"], 4)}')
    logger.info('Event Role Counts')
    num_args = 0
    for k, v in filter(lambda t: t[0][0] == 'EventRole', dataset_counter.items()):
        logger.info(f'# of role {k[1]}: {v}')
        num_args += v
    logger.info(f'# of args: {num_args}')
    logger.info('Induced Ontology')
    logger.info('Named Entity')
    logger.info(ner)
    logger.info('Event')
    for k, v in event_roles.items():
        logger.info(f'Event type: {k}, roles: {v}')
    logger.info('Relation')
    logger.info(relation_types)


def collect_raw_document_data(doc: CementDocument) -> Dict[str, Union[Counter, Set, List]]:
    # check meta for sentence level iterator pre-generated
    # has_ner_sentence_iterator = bool(doc.read_kv_map(prefix='meta', key='ner-iterator', suffix='sentence'))
    # has_event_sentence_iterator = bool(doc.read_kv_map(prefix='meta', key='events-iterator', suffix='sentence'))
    # has_relation_sentence_iterator = bool(doc.read_kv_map(prefix='meta', key='relations-iterator', suffix='sentence'))
    document_counter = Counter()
    document_desc = {
        'doc_key': doc.comm.id,
        'num_sentences': doc.num_sentences()
    }
    ner = set()
    event_types = set()
    event_roles: defaultdict = defaultdict(set)
    relation_types = set()

    document_counter['Tokens'] += len(doc)
    for i in range(doc.num_sentences()):
        document_counter[('Sentence', i)] += len(doc.get_sentence(sent_id=i))
        document_counter['Sentence'] += 1

    # count `EntityMention`s
    # if has_ner_sentence_iterator and doc.read_kv_map(prefix='ner', key=str(i), suffix='sentence') != '':
    #     em_uuids = doc.read_kv_map(prefix='ner', key=str(i), suffix='sentence').split(',')
    ems = list(doc.iterate_entity_mentions())
    document_counter['EntityMention'] += len(ems)
    for em in ems:
        cem = CementEntityMention.from_entity_mention(mention=em, document=doc)
        document_counter[('EntityMention', 'tokens')] += len(cem)
        document_counter[('NER', cem.attrs.entity_type)] += 1
        ner.add(cem.attrs.entity_type)

    # count `SituationMention`s
    # Event Mention
    # if has_event_sentence_iterator and doc.read_kv_map(prefix='event', key=str(i), suffix='sentence') != '':
    #     event_uuids = doc.read_kv_map(prefix='event', key=str(i), suffix='sentence').split(',')
    events = list(doc.iterate_event_mentions())
    document_counter['EventMention'] += len(events)
    for event in events:
        document_counter[('EventMention', 'args')] += len(event.argumentList)
        document_counter[('EventType', event.situationKind)] += 1
        event_types.add(event.situationKind)
        participated_roles = set()
        for arg in event.argumentList:
            participated_roles.add(arg.role)
            document_counter[('EventRole', arg.role)] += 1
        event_roles[event.situationKind].update(participated_roles)
    # Relation Mention
    # if has_relation_sentence_iterator and doc.read_kv_map(prefix='relation', key=str(i), suffix='sentence') != '':
    #     relation_uuids = doc.read_kv_map(prefix='relation', key=str(i), suffix='sentence').split(',')
    relations = list(doc.iterate_relation_mentions())
    document_counter['RelationMention'] += len(relations)
    for relation in relations:
        document_counter[('RelationType', relation.situationKind)] += 1
        relation_types.add(relation.situationKind)

    # if not has_ner_sentence_iterator:
    #     logger.warning(f'doc_key: {doc.comm.id} - does not have sentence iterator info for NER.')
    # if not has_event_sentence_iterator:
    #     logger.warning(f'doc_key: {doc.comm.id} - does not have sentence iterator info for Event Mention.')
    # if not has_relation_sentence_iterator:
    #     logger.warning(f'doc_key: {doc.comm.id} - does not have sentence iterator info for Relation Mention.')

    document_desc['counter'] = document_counter
    document_desc['ontology'] = {
        'ner': ner,
        'event_types': event_types,
        'event_roles': event_roles,
        'relation_types': relation_types
    }

    return document_desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'dev', 'test'])
    # parser.add_argument('--report-path', type=str, required=False)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info(f'Dataset path: {args.dataset_path}')
    for split in args.splits:
        logger.info(f'Dataset split: {split}')
        stats_on_documents(doc_stream=read_documents(dir_path=os.path.join(args.dataset_path, split)))

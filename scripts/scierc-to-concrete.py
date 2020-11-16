import json
import logging
import os
from typing import *
import argparse

from tqdm import tqdm

from cement.cement_document import CementDocument
from cement.cement_entity_mention import CementEntityMention
from cement.cement_span import CementSpan

logger = logging.getLogger(__name__)


def read_json(input_path: str, use_dir: bool = False) -> Generator[Dict, None, None]:
    if use_dir:
        file_names: List[str] = os.listdir(input_path)
        for fn in file_names:
            if '.json' not in fn:
                continue
            with open(os.path.join(input_path, fn)) as f:
                yield json.load(f)
    else:
        with open(input_path) as f:
            for line in f:
                yield json.loads(line)


def to_cement_doc_stream(json_stream: Iterable[Dict]) -> Iterable[CementDocument]:
    for json_obj in json_stream:
        # create a `CementDocument`
        doc = CementDocument.from_tokens(tokens={'passage': json_obj['sentences']},
                                         doc_id=json_obj['doc_key'])

        # extract entity mentions (EMD or NER)
        doc.write_kv_map(prefix='meta', key='ner-iterator', suffix='sentence', value='True')
        for line_id, ems in enumerate(json_obj['ner']):
            uuids = []
            for em in ems:
                cem = CementEntityMention(start=em[0],
                                          end=em[1],
                                          entity_type=em[2],
                                          document=doc)
                em_id = doc.add_entity_mention(mention=cem)
                uuids.append(em_id.uuidString)
            doc.write_kv_map(prefix='ner', key=str(line_id), suffix='sentence', value=','.join(uuids))

        # extract event mentions
        if 'events' in json_obj:
            doc.write_kv_map(prefix='meta', key='relations-iterator', suffix='sentence', value='True')
            for line_id, events in enumerate(json_obj['events']):
                uuids = []
                for event in events:
                    trigger = CementSpan(start=event[0][0], end=event[0][0], document=doc)
                    arguments = [
                        CementEntityMention(start=start, end=end, role=role, document=doc)
                        for start, end, role in event[1:]
                    ]
                    sm_id = doc.add_event_mention(trigger=trigger, arguments=arguments, event_type=event[0][1])
                    uuids.append(sm_id.uuidString)
                doc.write_kv_map(prefix='event', key=str(line_id), suffix='sentence', value=','.join(uuids))
        else:
            logger.info(f'doc_key: {json_obj["doc_key"]} - does not have events.')

        # extract relation mentions
        if 'relations' in json_obj:
            doc.write_kv_map(prefix='meta', key='events-iterator', suffix='sentence', value='True')
            for line_id, relations in enumerate(json_obj['relations']):
                uuids = []
                for relation in relations:
                    sub = CementEntityMention(start=relation[0], end=relation[1], document=doc)
                    obj = CementEntityMention(start=relation[2], end=relation[3], document=doc)
                    sm_id = doc.add_relation_mention(arguments=[sub, obj],
                                                     relation_type=relation[4])
                    uuids.append(sm_id.uuidString)
                doc.write_kv_map(prefix='relation', key=str(line_id), suffix='sentence', value=','.join(uuids))
        else:
            logger.info(f'doc_key: {json_obj["doc_key"]} - does not have relations.')

        yield doc


def serialize_doc(doc_stream: Iterable[CementDocument], base_path: str) -> NoReturn:
    for doc in tqdm(doc_stream):
        doc.to_communication_file(file_path=os.path.join(base_path, f'{doc.comm.id}.concrete'))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--use-dir', action='store_true')
    args = parser.parse_args()

    serialize_doc(doc_stream=to_cement_doc_stream(json_stream=read_json(input_path=args.input_path,
                                                                        use_dir=args.use_dir)),
                  base_path=args.output_path)

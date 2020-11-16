import logging
import os
from typing import *
import argparse
import json

from nltk import WordNetLemmatizer
from tqdm import tqdm

from cement.cement_document import CementDocument
from cement.cement_entity_mention import CementEntityMention
from cement.cement_span import CementSpan

logger = logging.getLogger(__name__)


def read_json() -> Generator[Dict, None, None]:
    if args.use_dir:
        file_names: List[str] = os.listdir(args.input_json)
        for fn in file_names:
            if '.json' not in fn or '.jsonlines' in fn:
                continue
            with open(os.path.join(args.input_json, fn)) as f:
                logger.info(f'Processing {fn}')
                yield json.load(f)
    else:
        with open(args.input_json) as f:
            for line_no, line in enumerate(f):
                logger.info(f'Processing line {line_no}')
                yield json.loads(line)


def get_rams_event(doc: Dict) -> Tuple[List[Tuple[int, int, str]], str]:
    # This function assumes that there is only one event in the JSON doc
    spans: List[Tuple[int, int, str]] = [
        (mention[0], mention[1], ontology['mappings']['args'][mention[2][0][0]])
        for mention in doc['ent_spans']
    ]
    spans.append((doc['evt_triggers'][0][0], doc['evt_triggers'][0][1], 'TRIGGER'))
    return spans, doc['evt_triggers'][0][2][0][0]


def get_predicted_mentions(doc: Dict) -> List[Tuple[int, int]]:
    return [
        (t[0], t[1])
        for t in doc['predicted_clusters'][0]
    ]


def rams_json_to_concrete(documents: Iterable[Dict], has_additional_mention: bool = True) -> Iterable[CementDocument]:
    for i, doc in tqdm(enumerate(documents)):
        cement_doc = CementDocument.from_tokens(tokens={'passage': doc['sentences']},
                                                doc_id=doc['doc_key'])
        raw_args, event_type = get_rams_event(doc)
        cement_doc.add_event_mention(
            trigger=CementSpan(start=raw_args[-1][0], end=raw_args[-1][1], document=cement_doc),
            arguments=[
                CementEntityMention(start=start, end=end, document=cement_doc, role=role)
                for start, end, role in raw_args if role != 'TRIGGER'
            ],
            event_type=ontology['mappings']['events'][event_type],
        )
        if has_additional_mention:
            additional_mentions = get_predicted_mentions(doc)
            for mention in additional_mentions:
                cement_doc.add_entity_mention(mention=CementEntityMention(start=mention[0],
                                                                          end=mention[-1],
                                                                          document=cement_doc))
        yield cement_doc


def gvdb_json_to_concrete(documents: Iterable[Dict]) -> Iterable[CementDocument]:
    for i, doc in tqdm(enumerate(documents)):
        cement_doc = CementDocument.from_tokens(tokens={'passage': doc['full_text']},
                                                doc_id=doc['doc_key'])
        cement_doc.add_event_mention(
            arguments=[
                CementEntityMention(start=start, end=end - 1, document=cement_doc, role=role)
                for start, end, role, _, _ in doc['spans']
            ],
            event_type='Shooting',
        )
        yield cement_doc


def aida_json_to_concrete(documents: Iterable[Dict]) -> Iterable[CementDocument]:
    def _get_events(jdoc: Dict, cdoc: CementDocument) -> Iterable[Tuple[str, CementSpan, List[CementSpan]]]:
        events: Dict[Tuple[int, int], List] = {}
        for trigger in jdoc['evt_triggers']:
            trigger_start = trigger[0]
            trigger_end = trigger[1]
            etype = trigger[2][0][0] if 'unspecified' not in trigger[2][0][0] else trigger[2][0][0].replace(
                'unspecified', 'n/a')
            events[(trigger_start, trigger_end)] = [
                ontology['mappings']['events'][etype],
                CementSpan(start=trigger_start, end=trigger_end, document=cdoc)
            ]
        for link in jdoc['gold_evt_links']:
            # logger.info(f'Role: {link[2]} - Normalized Role: {ontology["mappings"]["args"][link[2]]}')
            events[(link[0][0], link[0][1])].append(
                CementEntityMention(
                    start=link[1][0], end=link[1][1], role=ontology['mappings']['args'][link[2]], document=cdoc
                )
            )
        for event in events.values():
            yield event[0], event[1], event[2:]

    for i, doc in tqdm(enumerate(documents)):
        cement_doc = CementDocument.from_tokens(tokens={'passage': doc['sentences']},
                                                doc_id=doc['doc_key'])
        for event_type, event_trigger, event_args in _get_events(jdoc=doc, cdoc=cement_doc):
            cement_doc.add_event_mention(
                trigger=event_trigger,
                arguments=event_args,
                event_type=event_type
            )
        yield cement_doc


def bnb_json_to_concrete(documents: Iterable[Dict]) -> Iterable[CementDocument]:
    wnl = WordNetLemmatizer()
    for i, doc in tqdm(enumerate(documents)):
        cement_doc = CementDocument.from_tokens(tokens={'passage': doc['sentences']},
                                                doc_id=doc['doc_key'])
        arguments = []
        event_type = wnl.lemmatize(word=doc['trigger']['text'][0].lower())
        event_type = event_type if 'loss' not in event_type else 'loss'
        for k, v in doc['arguments'].items():
            if len(v) > 1:
                logger.info(f'Arg has more than one span: {k} - {v}')
            for arg in v:
                arguments.append(
                    CementEntityMention(start=arg['span'][0],
                                        end=arg['span'][1],
                                        document=cement_doc,
                                        role=k)
                                        # role=f'{event_type}-{k}' if 'Arg' in k else k)
                )
        # special handling for `tax-loss` predicate
        cement_doc.add_event_mention(
            trigger=CementSpan(start=doc['trigger']['span'][0], end=doc['trigger']['span'][1], document=cement_doc),
            arguments=arguments,
            event_type=event_type,
        )
        yield cement_doc


def write_to_file(documents: Iterable[CementDocument], base_url: str):
    for doc in documents:
        doc.to_communication_file(file_path=os.path.join(base_url, f'{doc.comm.id}.concrete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', type=str)
    parser.add_argument('--dataset', type=str, choices=['rams', 'aida', 'gvdb', 'bnb'])
    parser.add_argument('--ontology-path', type=str, required=False)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--use-dir', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    if args.dataset in ['rams', 'aida']:
        with open(args.ontology_path) as f:
            ontology: Optional[Dict] = json.load(f)
    else:
        ontology = None

    if args.dataset == 'rams':
        json_to_concrete = rams_json_to_concrete
    elif args.dataset == 'aida':
        json_to_concrete = aida_json_to_concrete
    elif args.dataset == 'gvdb':
        json_to_concrete = gvdb_json_to_concrete
    else:  # BNB
        json_to_concrete = bnb_json_to_concrete

    json_doc_stream: Generator[Dict, None, None] = read_json()
    write_to_file(documents=json_to_concrete(documents=json_doc_stream),
                  base_url=args.output_dir)

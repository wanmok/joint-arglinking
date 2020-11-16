import datetime
import os
from typing import *
import argparse
import json

from concrete import Communication, AnnotationMetadata, LanguageIdentification, Sentence, Tokenization, \
    TokenizationKind, TokenList, Token, Section, EntityMention, TokenRefSequence, SituationMention, MentionArgument, \
    EntityMentionSet, SituationMentionSet
from concrete.util import AnalyticUUIDGeneratorFactory, write_communication_to_file
from concrete.validate import validate_communication
from tqdm import tqdm


def check_duplicate_mention(comm: Communication, span: Tuple[int, int]) -> Optional[EntityMention]:
    for mention in comm.entityMentionSetList[0].mentionList:
        mention_span: Tuple[int, int] = (mention.tokens.tokenIndexList[0], mention.tokens.tokenIndexList[-1] + 1)
        if span == mention_span:
            return mention
    return None


def read_json() -> Generator[Dict, None, None]:
    if args.use_dir:
        file_names: List[str] = os.listdir(args.input_json)
        for fn in file_names:
            if '.json' not in fn:
                continue
            with open(os.path.join(args.input_json, fn)) as f:
                yield json.load(f)
    else:
        with open(args.input_json) as f:
            for line in f:
                yield json.loads(line)


def get_flatten_sentence(doc: Dict) -> List[str]:
    # This function assumes that the whole JSON doc is one data example
    sent: List[str] = []
    for s in doc['sentences']:
        sent.extend(s)
    return sent


def get_event(doc: Dict) -> Tuple[List[Tuple[int, int, str]], str]:
    # This function assumes that there is only one event in the JSON doc
    spans: List[Tuple[int, int, str]] = [
        (mention[0], mention[1] + 1, ontology['mappings']['args'][mention[2][0][0]])
        for mention in doc['ent_spans']
    ]
    spans.append((doc['evt_triggers'][0][0], doc['evt_triggers'][0][1] + 1, 'TRIGGER'))
    return spans, doc['evt_triggers'][0][2][0][0]


def get_predicted_mentions(doc: Dict) -> List[Tuple[int, int]]:
    return [
        (t[0], t[1] + 1)
        for t in doc['predicted_clusters'][0]
    ]


def add_additional_mentions_to_comm(mentions: List[Tuple[int, int]],
                                    comm: Communication):
    tokenization_id = comm.sectionList[0].sentenceList[0].tokenization.uuid
    for mention in mentions:
        if check_duplicate_mention(comm, (mention[0], mention[1])) is not None:
            continue

        new_entity_mention = EntityMention(
            uuid=augf.next(),
            tokens=TokenRefSequence(
                tokenIndexList=list(range(mention[0], mention[1])),
                tokenizationId=tokenization_id
            )
        )
        comm.entityMentionSetList[0].mentionList.append(new_entity_mention)


def add_event_to_comm(mentions: List[Tuple[int, int, str]],
                      event_type: str,
                      comm: Communication):
    # This function assumes that the comm only has one sentence
    tokenization_id = comm.sectionList[0].sentenceList[0].tokenization.uuid
    comm.situationMentionSetList[0].mentionList.append(
        SituationMention(uuid=augf.next(),
                         situationType='EVENT',
                         situationKind=ontology['mappings']['events'][event_type],
                         argumentList=[])
    )
    for mention in mentions:
        new_entity_mention: Optional[EntityMention] = check_duplicate_mention(comm, (mention[0], mention[1]))
        if new_entity_mention is None:
            new_entity_mention = EntityMention(
                uuid=augf.next(),
                tokens=TokenRefSequence(
                    tokenIndexList=list(range(mention[0], mention[1])),
                    tokenizationId=tokenization_id
                )
            )
        comm.entityMentionSetList[0].mentionList.append(new_entity_mention)
        comm.situationMentionSetList[0].mentionList[0].argumentList.append(
            MentionArgument(
                role=mention[2],
                entityMentionId=new_entity_mention.uuid
            )
        )


def json_to_concrete(doc: Dict) -> Communication:
    metadata = AnnotationMetadata(
        tool="BlingBLing",
        timestamp=int(datetime.datetime.now().timestamp())
    )
    comm: Communication = Communication(
        uuid=augf.next(),
        id=doc['doc_key'],
        type="aida",
        metadata=metadata,
        lidList=[LanguageIdentification(
            uuid=augf.next(),
            metadata=metadata,
            languageToProbabilityMap={doc['language_id']: 1.0}
        )],
        sectionList=[Section(
            uuid=augf.next(),
            kind="passage",
            sentenceList=[
                Sentence(
                    uuid=augf.next(),
                    tokenization=Tokenization(
                        uuid=augf.next(),
                        kind=TokenizationKind.TOKEN_LIST,
                        metadata=metadata,
                        tokenList=TokenList(
                            tokenList=[
                                Token(
                                    tokenIndex=i,
                                    text=t
                                )
                                for i, t in enumerate(get_flatten_sentence(doc))
                            ]
                        )
                    )
                )
            ]
        )],
        entityMentionSetList=[EntityMentionSet(
            uuid=augf.next(),
            metadata=metadata,
            mentionList=[]
        )],
        situationMentionSetList=[SituationMentionSet(
            uuid=augf.next(),
            metadata=metadata,
            mentionList=[]
        )]
    )

    return comm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', type=str)
    parser.add_argument('--aida-ontology', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--use-dir', action='store_true')
    args = parser.parse_args()

    with open(args.aida_ontology) as f:
        ontology: Dict = json.load(f)

    json_doc_stream: Generator[Dict, None, None] = read_json()
    augf = AnalyticUUIDGeneratorFactory().create()
    for doc in tqdm(json_doc_stream):
        new_mentions, new_event_type = get_event(doc)
        new_comm = json_to_concrete(doc)
        add_event_to_comm(new_mentions, new_event_type, new_comm)

        predicted_mention_spans: List[Tuple[int, int]] = get_predicted_mentions(doc)
        add_additional_mentions_to_comm(mentions=predicted_mention_spans,
                                        comm=new_comm)

        assert validate_communication(new_comm)
        write_communication_to_file(new_comm, os.path.join(args.output_dir, f'{new_comm.id}.concrete'))

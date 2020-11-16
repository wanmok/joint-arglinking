import logging
import os
from typing import *
import argparse

import spacy
from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp_models.syntax import BiaffineDependencyParserPredictor
from overrides import overrides
from spacy.tokens.doc import Doc

from cement.cement_document import CementDocument
from cement.cement_entity_mention import CementEntityMention
from allennlp.predictors.predictor import Predictor
import allennlp_models.syntax.biaffine_dependency_parser

from cement.cement_span import CementSpan

logger = logging.getLogger(__name__)


@Predictor.register('dep_parsing_pred')
class DependencyParsingPredictor(BiaffineDependencyParserPredictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        doc = Doc(spacy_model.vocab,
                  words=json_dict['sentence'],
                  spaces=[True for _ in range(len(json_dict['sentence']))])
        spacy_tokens = spacy_model.pipeline[0][1](doc)
        if self._dataset_reader.use_language_specific_pos:  # type: ignore
            # fine-grained part of speech
            pos_tags = [token.tag_ for token in spacy_tokens]
        else:
            # coarse-grained part of speech (Universal Depdendencies format)
            pos_tags = [token.pos_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(json_dict['sentence'], pos_tags)

    @overrides
    def predict(self, sentence: List[str]) -> JsonDict:
        return self.predict_json({'sentence': sentence})


def docment_stream(file_dir: str) -> Iterable[CementDocument]:
    file_names = os.listdir(file_dir)
    for fn in file_names:
        if '.comm' not in fn and '.concrete' not in fn:
            continue
        yield CementDocument.from_communication_file(file_path=os.path.join(file_dir, fn))


def process_mentions(doc_stream: Iterable[CementDocument],
                     predictor: Predictor) -> Iterable[CementDocument]:
    for doc in doc_stream:
        # process EntityMention
        for em in doc.iterate_entity_mentions():
            cem = CementEntityMention.from_entity_mention(mention=em, document=doc)
            head_token_offset = find_mention_head(mention=cem, predictor=predictor)
            cem.attrs.head = cem.start + head_token_offset
            cem.write_em_head_to_comm()
        # process trigger
        for sm in doc.iterate_situation_mentions():
            if sm.tokens is not None:
                trigger_span = CementSpan.from_token_ref_sequence(token_ref_sequence=sm.tokens,
                                                                  document=doc)
                head_token_offset = find_mention_head(mention=trigger_span, predictor=predictor)
                trigger_span.write_span_kv(value=str(trigger_span.start + head_token_offset),
                                           suffix='head',
                                           key=sm.uuid.uuidString,
                                           key_prefix='trigger')
        yield doc


def serialize_concrete(doc_stream: Iterable[CementDocument],
                       output_path: str) -> NoReturn:
    for doc in doc_stream:
        # doc.to_communication_file(file_path=os.path.join(output_path, f'{doc.comm.id}.concrete'))
        pass


def find_mention_head(mention: CementSpan,
                      predictor: Predictor) -> Optional[int]:
    tokens = mention.get_tokens()
    predicted_results = predictor.predict(sentence=tokens)
    # sanity check
    if (
            len(tokens) != len(predicted_results['words'])
            or not all([a == b for a, b in zip(tokens, predicted_results['words'])])
    ):
        logger.warning(f'Tokenizations do not match: {tokens} - {predicted_results["words"]}')
        return None

    for i in range(len(tokens)):
        if predicted_results['predicted_dependencies'][i] == 'root':
            logger.info(f'Mention: {mention} - has head: {predicted_results["words"][i]}')
            return i
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    dependency_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
        predictor_name='dep_parsing_pred'
    )

    spacy_model = spacy.load('en_core_web_sm')

    serialize_concrete(doc_stream=process_mentions(doc_stream=docment_stream(file_dir=args.input_path),
                                                   predictor=dependency_predictor),
                       output_path=args.output_path)

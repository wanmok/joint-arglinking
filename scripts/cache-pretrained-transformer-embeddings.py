import json
import os
from itertools import groupby
from typing import *
import argparse
import logging

import h5py
import torch
import numpy as np
from allennlp.data import Instance, Token, Field, Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from tqdm import tqdm

from cement.cement_document import CementDocument

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def instance_stream(file_path: str,
                    tokenizer: PretrainedTransformerTokenizer,
                    token_indexers: Dict[str, PretrainedTransformerIndexer],
                    model_input_size: int = 512,
                    normalize: bool = False) -> Generator[Instance, None, None]:
    file_names = os.listdir(file_path)
    for fn in file_names:
        if args.file_suffix not in fn:  # '.json', '.comm'
            continue

        if args.file_suffix == '.json':
            with open(os.path.join(file_path, fn)) as f:
                data = json.load(f)
            sentences = data['sentences']
            doc_key = data['doc_key']
        else:
            doc = CementDocument.from_communication_file(file_path=os.path.join(file_path, fn))
            sentences = list(doc.iterate_sentences())
            doc_key = str(doc.comm.id)

        if normalize:
            sentences = [
                [normalize_token(t) for t in sent]
                for sent in sentences
            ]

        tokenized_context_sentences: List[Tuple[List[List[Token]], List[Tuple[int, int]], List[str]]] = []
        for sent in sentences:
            tokenized_sent, offsets = tokenizer.intra_word_tokenize(sent)
            if len(tokenized_sent) > model_input_size:
                logger.info('Segmented long sentence.')
                tokenized_context_sentences.append(
                    (segment_long_sentence(tokenized_sent, model_input_size), offsets, sent)
                )
            else:
                tokenized_context_sentences.append(([tokenized_sent], offsets, sent))

        for sent_id, (sent_token_list, sent_offsets, sent) in enumerate(tokenized_context_sentences):
            for i, sent_tokens in enumerate(sent_token_list):
                # print(f'{[doc_key, str(sent_id)]}')
                yield construct_instance(tokens=sent_tokens,
                                         offsets=sent_offsets,
                                         key=[doc_key, str(sent_id)],
                                         segment=i,
                                         raw_sentence=sent,
                                         token_indexers=token_indexers)


def normalize_token(t: str) -> str:
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

def segment_long_sentence(sent: List[Token],
                          model_input_size: int = 512) -> List[List[Token]]:
    segment_size: int = model_input_size - 2
    sent, start_token, end_token = sent[1:-1], sent[0], sent[-1]
    num_segments: int = len(sent) // segment_size + 1
    segments: List[List[Token]] = []
    for i in range(num_segments):
        new_segment = [start_token]
        new_segment.extend(sent[i * segment_size:(i + 1) * segment_size])
        new_segment.append(end_token)
        segments.append(new_segment)
    return segments


def construct_instance(tokens: List[Token],
                       offsets: List[Tuple[int, int]],
                       key: List[str],
                       segment: int,
                       raw_sentence: List[str],
                       token_indexers: Dict[str, PretrainedTransformerIndexer]) -> Instance:
    text_field: TextField = TextField(tokens=tokens,
                                      token_indexers=token_indexers)
    metadata_field: MetadataField = MetadataField({
        'offsets': offsets,
        'key': key,
        'segment': segment,
        'raw_sentence': raw_sentence,
    })
    fields: Dict[str, Field] = {
        'text': text_field,
        'metadata': metadata_field
    }
    return Instance(fields)


def compose_batch_stream(ins_stream: Generator[Instance, None, None],
                         batch_size: int = 12) -> Generator[Batch, None, None]:
    buffer = []
    while True:
        try:
            buffer.append(next(ins_stream))
            if len(buffer) == batch_size:
                yield Batch(buffer)
                buffer.clear()
        except StopIteration:
            break
    if len(buffer) != 0:
        yield Batch(buffer)


def get_embedding_stream(embedder: PretrainedTransformerEmbedder,
                         batch_stream: Generator[Batch, None, None],
                         ) -> Generator[Tuple[Instance, torch.Tensor], None, None]:
    vocab = Vocabulary()
    for batch in batch_stream:
        batch.index_instances(vocab)
        tensors = batch.as_tensor_dict()
        for k, v in tensors['text']['token'].items():
            tensors['text']['token'][k] = v.to(device)
        embeddings: torch.Tensor = embedder(
            **tensors['text']['token']).detach().cpu()  # [batch_size, seq_len, embed_size]
        for ins_id, ins in enumerate(batch):
            curr_embed = embeddings[ins_id].squeeze(0)
            yield ins, curr_embed
            # array_data = np.concatenate([
            #     curr_embed[offset[0]:offset[1] + 1, :].mean(dim=0).view(1, -1).numpy()  # [wordpiece_len, embed_size]
            #     for offset in ins.fields['metadata']['offsets']
            # ], axis=0)  # [sent_len, embed_size]
            # yield ins.fields['metadata']['key'], array_data


def debatch_embedding_stream(
        embed_stream: Iterable[Tuple[Instance, torch.Tensor]]
) -> Generator[Tuple[List[str], np.ndarray], None, None]:
    for sent_key, segments in groupby(embed_stream, key=lambda k: tuple(k[0].fields['metadata']['key'])):
        segments = sorted(segments, key=lambda k: k[0].fields['metadata']['segment'])
        ins = segments[0][0]
        # if sent_key[0] == 'XIN_ENG_20030324.0191':
        #     print(f'{sent_key} - {[seg[1].shape for seg in segments]} - {len(ins["metadata"]["raw_sentence"])}')
        #     set_trace()
        if len(segments) > 1:
            curr_embed = torch.cat(
                [
                    embed[:-1, :] if i == 0 else (embed[1:, :] if i == len(segments) - 1 else embed[1:-1, :])
                    for i, (instance, embed) in enumerate(segments)
                ],
                dim=0
            )
        else:
            curr_embed = segments[0][1]

        array_data = np.concatenate([
            curr_embed[offset[0]:offset[1] + 1, :].mean(dim=0).view(1, -1).numpy()  # [wordpiece_len, embed_size]
            for offset in ins.fields['metadata']['offsets']
        ], axis=0)  # [sent_len, embed_size]

        assert len(ins.fields['metadata']['raw_sentence']) == array_data.shape[0]

        yield ins.fields['metadata']['key'], array_data


def write_to_hdf5(cache_handler: h5py.File,
                  embed_stream: Generator[Tuple[List[str], np.ndarray], None, None]):
    for key, array_data in tqdm(embed_stream):
        group_key = '/'.join(key[:-1])
        dataset_key = key[-1]
        group_obj = cache_handler.require_group(group_key)
        group_obj.create_dataset(name=dataset_key, data=array_data, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--cache-path', type=str)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--model', type=str, default='bert-base-cased')
    parser.add_argument('--file-suffix', type=str, choices=['.json', '.comm', '.concrete'], default='.json')
    parser.add_argument('--normalize-token', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    tokenizer = PretrainedTransformerTokenizer(model_name=args.model)
    token_indexers = {'token': PretrainedTransformerIndexer(model_name=args.model)}
    model = PretrainedTransformerEmbedder(model_name=args.model)
    if args.cuda:
        device = 'cuda'
        model.to('cuda')
    else:
        device = 'cpu'

    o = h5py.File(args.cache_path, 'w')
    write_to_hdf5(
        cache_handler=o,
        embed_stream=debatch_embedding_stream(embed_stream=get_embedding_stream(
            embedder=model,
            batch_stream=compose_batch_stream(
                ins_stream=instance_stream(
                    file_path=args.input_path,
                    tokenizer=tokenizer,
                    token_indexers=token_indexers
                ),
                batch_size=args.batch_size
            )
        ))
    )

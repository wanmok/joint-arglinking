import argparse
import json
import logging
import os
from typing import *

import h5py
import numpy as np
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Features(NamedTuple):
    tokens: List[str]
    ids_to_original: torch.Tensor
    input_ids: torch.Tensor


def convert_tokens_to_features(tokens: List[str],
                               tokenizer: BertTokenizer,
                               do_lower_case: bool = True) -> Features:
    input_tokens: List[str] = ['[CLS]']
    ids_to_original: List[int] = [-1]
    for k, token in enumerate(tokens):
        for wp in tokenizer.wordpiece_tokenizer.tokenize(
                token.lower() if do_lower_case else token
        ):  # lower_case
            input_tokens.append(wp)
            ids_to_original.append(k)
    input_tokens.append('[SEP]')
    ids_to_original.append(-1)

    features: Features = Features(
        tokens=input_tokens,
        ids_to_original=torch.tensor(ids_to_original, dtype=torch.long),
        input_ids=torch.tensor(tokenizer.convert_tokens_to_ids(input_tokens), dtype=torch.long),
    )

    return features


def convert_sentences_to_features_list(tokens_list: List[List[str]],
                                       tokenizer: BertTokenizer,
                                       do_lower_case: bool = True) -> List[Features]:
    return [
        convert_tokens_to_features(tokens=sent,
                                   tokenizer=tokenizer,
                                   do_lower_case=do_lower_case)
        for sent in tokens_list
    ]


def get_embeddings(input_ids: torch.Tensor,
                   layers: List[int],
                   model: BertModel) -> torch.Tensor:
    with torch.no_grad():
        # assuming pass *1* sentence per call
        input_ids: torch.Tensor = input_ids.view(-1, input_ids.shape[0])
        all_encoder_layers, _ = model(input_ids)

        filtered_encoder_layers: torch.Tensor = torch.stack([
            all_encoder_layers[i].detach().cpu()
            for i in layers
        ], dim=1)

    return filtered_encoder_layers.squeeze(dim=0)  # [num_layer, num_token, embedding_size]


def merge_embeddings(ids_to_original: torch.Tensor,
                     embeddings: torch.Tensor) -> torch.Tensor:
    layer_embeddings: torch.Tensor = embeddings.detach().cpu()
    embedding_size: int = layer_embeddings.shape[2]
    num_layers: int = layer_embeddings.shape[0]
    num_original_tokens: int = ids_to_original.max().item()
    if num_original_tokens < 0:
        return torch.zeros(1).expand(num_layers, 1, embedding_size)

    merged_embeddings: torch.Tensor = torch.cat([
        layer_embeddings[:, ids_to_original == i].mean(dim=1).view(num_layers, 1, -1)
        for i in range(0, num_original_tokens + 1)
    ], dim=1)

    return merged_embeddings  # [num_layer, num_token, embedding_size]


def bert_embeds(features: Features,
                model: BertModel,
                layer_indexes: List[int],
                device: str) -> torch.Tensor:
    unmerged_embeddings: torch.Tensor = get_embeddings(
        input_ids=features.input_ids.to(device),
        layers=layer_indexes,
        model=model
    )
    merged_embeddings: torch.Tensor = merge_embeddings(
        ids_to_original=features.ids_to_original,
        embeddings=unmerged_embeddings
    )  # [num_layer, num_token, embedding_size]

    return merged_embeddings


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json-path", type=str, default="", help="Path to json file")
    parser.add_argument("--output-path", type=str, default="", help="Path to output dump")
    parser.add_argument("--cuda", action='store_true', help="Whether to use CUDA")
    parser.add_argument("--layers", default="0,1,-1,-2,-3,-4", type=str)
    parser.add_argument("--models", default="bert-base-cased", type=str,
                        help="A list of pretrained models separated by \":\". Consider GPU memory constraints!")
    parser.add_argument("--langs", default="eng", type=str,
                        help="A list of corresponding languages separated by :")
    parser.add_argument("--dirty-input-log", default="dirty_inputs.log", type=str)
    # parser.add_argument("--auto_langid", action="store_false", help="Whether to use langID to select BERT model")
    ARGS = parser.parse_args()

    def _get_document(file: str, is_dir: bool = True):
        if is_dir:
            file_names = os.listdir(file)
            all_jsons = []
            for fn in file_names:
                if '.json' in fn:
                    with open(os.path.join(file, fn)) as f:
                        all_jsons.append(json.load(f))
            for j in all_jsons:
                yield j
        else:
            with open(file) as f:
                for line in f:
                    yield json.loads(line)

    # process parameters
    device: str = 'cpu'
    if ARGS.cuda:
        device: str = 'cuda'
    layer_indexes: List[int] = [int(x) for x in ARGS.layers.split(",")]
    langs: List[str] = ARGS.langs.split(':')
    model_names: List[str] = ARGS.models.split(':')

    # Load pre-trained models and  tokenizers (vocabulary)
    models: Dict[str, BertModel] = {
        l: BertModel.from_pretrained(m).to(device)
        for l, m in zip(langs, model_names)
    }
    tokenizers: Dict[str, BertTokenizer] = {
        # TODO: whether cased is indicated in the model name
        l: BertTokenizer.from_pretrained(m, do_lower_case=True if 'uncased' in m else False)
        for l, m in zip(langs, model_names)
    }
    do_lower_case: Dict[str, bool] = {
        l: True if 'uncased' in m else False
        for l, m in zip(langs, model_names)
    }
    for model in models.values():
        model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    # tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
    #     'bert-base-uncased',
    #     do_lower_case=True
    # )
    # model: BertModel = BertModel.from_pretrained('bert-base-uncased').to(device)

    o = h5py.File(ARGS.output_path, 'w')
    if ARGS.dirty_input_log:
        log_file = open(ARGS.dirty_input_log, 'w')
    for comm in tqdm(_get_document(file=ARGS.json_path)):
        # if not f.endswith('.json'):
        #     continue
        #
        # try:
        #     with open(f"{ARGS.json_path}/{f}", encoding='utf8') as file:
        #         comm: Dict = json.load(file)
        # except UnicodeDecodeError:
        #     logging.info(f'Cannot decode file: {ARGS.json_path}/{f}')

        comm_id: str = comm['doc_key'].replace("/", ":")

        # language ID
        # language_id: str = comm['language_id']
        # TODO: ???
        language_id: str = langs[0]
        if language_id not in langs:
            logging.info(f'Unrecognized langID: {language_id}')
            continue

        logging.info(f'comm_id: {comm_id}, langID: {language_id}')

        # which Tokenizer and Model to use
        tokenizer: BertTokenizer = tokenizers[language_id]
        model: BertModel = models[language_id]

        tokens_list: List[List[str]] = comm['sentences']

        features_list: List[Features] = convert_sentences_to_features_list(tokens_list=tokens_list,
                                                                           tokenizer=tokenizer,
                                                                           do_lower_case=do_lower_case[language_id])

        group = o.create_group(f"{comm_id}")
        for i, features in enumerate(features_list):
            if features.input_ids.shape[0] > 500:
                logging.info(f'Chunked {comm_id} - {i} sentence which is too long for BERT')
                if ARGS.dirty_input_log:
                    log_file.write(f'{comm_id}-{i}\n')
                # disgusting code.......
                # I hate writing things twice
                len_word_pices: int = features.input_ids.shape[0]
                num_chunks: int = int(np.ceil(len_word_pices / 500))
                input_ids: torch.Tensor = features.input_ids
                logging.info(f'\t# chunks: {num_chunks}')
                concated_embeddings: torch.Tensor = torch.cat([
                    get_embeddings(input_ids=input_ids[i * 500:(i + 1) * 500].clone().to(device),
                                   layers=layer_indexes,
                                   model=model)
                    for i in range(num_chunks)
                ], dim=1)  # [Layers, Tokens, Embeds]
                merged_embeddings: torch.Tensor = merge_embeddings(ids_to_original=features.ids_to_original,
                                                                   embeddings=concated_embeddings)
                assert merged_embeddings.shape[1] == len(tokens_list[i])

                group.create_dataset(name=str(i),
                                     data=merged_embeddings.numpy().transpose([1, 2, 0]),
                                     dtype=np.float32)
            else:

                embeddings: torch.Tensor = bert_embeds(features=features,
                                                       model=model,
                                                       device=device,
                                                       layer_indexes=layer_indexes)
                assert embeddings.shape[1] == len(tokens_list[i]) \
                       or (len(tokens_list[i]) == 0 and embeddings.shape[1] == 1), \
                    f'Embedding size: {embeddings.shape[1]}, Token list size: {len(tokens_list[i])}'

                group.create_dataset(name=str(i),
                                     data=embeddings.numpy().transpose([1, 2, 0]),
                                     dtype=np.float32)

    o.close()
    if ARGS.dirty_input_log:
        log_file.close()


if __name__ == '__main__':
    main()

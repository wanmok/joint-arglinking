import json
import argparse

from allennlp.data import Vocabulary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ontology-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    with open(args.ontology_path) as f:
        ontology = json.load(f)

    vocab = Vocabulary()
    vocab.add_token_to_namespace(token='None', namespace='span_labels')
    vocab.add_token_to_namespace(token='@@PADDING@@', namespace='span_labels')
    vocab.add_tokens_to_namespace(tokens=list(ontology['args'].keys()), namespace='span_labels')
    vocab.add_tokens_to_namespace(tokens=list(ontology['events'].keys()), namespace='event_labels')
    vocab.save_to_files(args.output_path)

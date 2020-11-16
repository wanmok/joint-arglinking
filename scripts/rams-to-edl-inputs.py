import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str)
parser.add_argument('--output-dir', type=str)
args = parser.parse_args()

with open(args.input_file) as f:
    for line in f:
        doc = json.loads(line)
        filename = doc['doc_key']
        # flatten the context
        # new_sentences = []
        # for s in doc['sentences']:
        #     new_sentences.extend(s)
        # doc['sentences'] = [new_sentences]
        # doc['doc_key'] = f'nw_{filename}'
        doc['clusters'] = []
        doc['all_predicted_spans'] = []
        doc['speakers'] = [
            [
                ''
                for _ in range(len(s))
            ]
            for s in doc['sentences']
        ]
        with open(os.path.join(args.output_dir, f'{filename}.json'), 'w') as out:
            json.dump(doc, out, indent=2)

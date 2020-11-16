import collections
import json
import os
from typing import *
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--rspan-output', type=str)
parser.add_argument('--report-output', type=str, required=False)
args = parser.parse_args()

gold_spans: collections.Counter = collections.Counter()
pred_spans: collections.Counter = collections.Counter()

file_names: List[str] = os.listdir(args.rspan_output)
for fn in file_names:
    if '.json' not in fn:
        continue

    with open(os.path.join(args.rspan_output, fn)) as f:
        doc = json.load(f)

    for t in doc['predicted_clusters'][0]:
        pred_spans[(doc['doc_key'], t[0], t[1] + 1)] = 1

    for t in doc['ent_spans']:
        gold_spans[(doc['doc_key'], t[0], t[1] + 1)] = 1

common: collections.Counter = gold_spans & pred_spans
scores: collections.Counter = collections.Counter()
scores['true_positive'] = sum(common.values())
scores['false_positive'] = sum(gold_spans.values()) - scores['true_positive']
scores['false_negative'] = sum(pred_spans.values()) - scores['true_positive']

precision: float = scores['true_positive'] / (scores['true_positive'] + scores['false_positive'])
recall: float = scores['true_positive'] / (scores['true_positive'] + scores['false_negative'])
f1: float = 2 * precision * recall / (precision + recall)

to_print = [
    f'Precision: {precision}',
    f'Recall: {recall}',
    f'F1: {f1}'
]
if args.report_output is None:
    print('\n'.join(to_print))
else:
    with open(args.report_output, 'w') as f:
        f.writelines('\n'.join(to_print))

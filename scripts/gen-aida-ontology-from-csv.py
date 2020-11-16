import collections
import csv
import json
from typing import *
import argparse


def process_gloss(gloss: str, arguments: List[Tuple[str, str]]) -> str:
    normalized_gloss: str = gloss.lower()

    for i, arg in enumerate(arguments):
        insert_id = normalized_gloss.find(f']{i + 1}')
        normalized_gloss = normalized_gloss[:insert_id + 1] + arg[1].lower() + normalized_gloss[insert_id + 1:]

    return normalized_gloss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aida-csv', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--gloss', action='store_true')
    args = parser.parse_args()

    result_json: Dict = {
        'events': {},
        'args': collections.defaultdict(dict),
        'mappings': {  # to normalize strings
            'events': {},
            'args': {}
        }
    }

    with open(args.aida_csv) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            event_type: str = row['Type']
            event_subtype: str = row['Subtype']
            event_subsubtype: str = row['Sub-subtype']
            event_gloss: str = row['Bleached Gloss']

            full_event_type: str = f'{event_type}.{event_subtype}.{event_subsubtype}'
            result_json['mappings']['events'][full_event_type.lower()] = full_event_type

            result_json['events'][full_event_type] = {}
            result_json['events'][full_event_type]['roles'] = {}
            arguments: List[Tuple[str, str]] = []
            for i in range(1, 6):  # hardcoded in csv file
                label: str = row[f'arg{i} label']
                output_value: str = row[f'Output value for arg{i}']
                if label == '' and output_value == '':
                    break
                result_json['mappings']['args'][output_value] = label
                result_json['events'][full_event_type]['roles'][label] = {
                    'pos': i,
                    'constraints': row[f'arg{i} type constraints'].split(', '),
                    'output_value': output_value
                }
                arguments.append((output_value, label))
                if 'events' in result_json['args'][label]:
                    result_json['args'][label]['events'].append(full_event_type)
                else:
                    result_json['args'][label]['events'] = [full_event_type]

            if args.gloss:
                result_json['events'][full_event_type]['gloss'] = process_gloss(event_gloss, arguments)

    with open(args.output, 'w') as f:
        json.dump(result_json, f, indent=2)

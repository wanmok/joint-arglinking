import collections
from typing import *
import json
import argparse
import logging

logger = logging.getLogger(__name__)


def build_gvdb_ontology():
    ontology = {
        'events': {
            'Shooting': {
                'roles': {}
            }
        },
        'args': {}
    }
    roles = [
        'SHO-AGE', 'DAT-CLOCK', 'DAT-LOC', 'VIC-RACE', 'VIC-NAME', 'SHO-RACE', 'CIR-WEAPON', 'VIC-AGE',
        'DAT-CITY', 'DAT-TIME', 'CIR-NUM-SHOTS', 'SHO-NAME'
    ]
    for rid, r in enumerate(roles):
        ontology['events']['Shooting']['roles'][r] = {
            'pos': rid,
            'constraints': [],
            'output_value': r.lower()
        }
        ontology['args'][r] = {'events': ['Shooting']}
    return ontology


def build_ace_ontology():
    ontology = {
        'events': {
            'Life.Be-Born': {
                'Person': {},
                'Time': {},
                'Place': {},
            },
            'Life.Injure': {
                'Agent': {},
                'Victim': {},
                'Instrument': {},
                'Time': {},
                'Place': {},
            },
            'Life.Die': {
                'Agent': {},
                'Victim': {},
                'Instrument': {},
                'Time': {},
                'Place': {},
            },
            'Life.Divorce': {
                'Person': {},
                'Time': {},
                'Place': {},
            },
            'Life.Marry': {
                'Person': {},
                'Time': {},
                'Place': {},
            },
            'Movement.Transport': {
                'Agent': {},
                'Artifact': {},
                'Vehicle': {},
                'Price': {},
                'Origin': {},
                'Destination': {},
                'Time': {},
            },
            'Transaction.Transfer-Ownership': {
                'Buyer': {},
                'Seller': {},
                'Beneficiary': {},
                'Artifact': {},
                'Price': {},
                'Time': {},
                'Place': {},
            },
            'Transaction.Transfer-Money': {
                'Giver': {},
                'Recipient': {},
                'Beneficiary': {},
                'Money': {},
                'Time': {},
                'Place': {},
            },
            'Business.Merge-Org': {
                'Org': {},
                'Time': {},
                'Place': {},
            },
            'Business.Declare-Bankruptcy': {
                'Org': {},
                'Time': {},
                'Place': {},
            },
            'Business.End-Org': {
                'Org': {},
                'Time': {},
                'Place': {},
            },
            'Business.Start-Org': {
                'Agent': {},
                'Org': {},
                'Time': {},
                'Place': {},
            },
            'Conflict.Attack': {
                'Attacker': {},
                'Target': {},
                'Instrument': {},
                'Time': {},
                'Place': {},
            },
            'Conflict.Demonstrate': {
                'Entity': {},
                'Time': {},
                'Place': {},
            },
            'Contact.Meet': {
                'Entity': {},
                'Time': {},
                'Place': {},
            },
            'Contact.Phone-Write': {
                'Entity': {},
                'Time': {},
            },
            'Personnel.Start-Position': {
                'Person': {},
                'Entity': {},
                'Position': {},
                'Time': {},
                'Place': {},
            },
            'Personnel.End-Position': {
                'Person': {},
                'Entity': {},
                'Position': {},
                'Time': {},
                'Place': {},
            },
            'Personnel.Elect': {
                'Person': {},
                'Entity': {},
                'Position': {},
                'Time': {},
                'Place': {},
            },
            'Personnel.Nominate': {
                'Person': {},
                'Agent': {},
                'Position': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Charge-Indict': {
                'Defendant': {},
                'Prosecutor': {},
                'Adjudicator': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Convict': {
                'Defendant': {},
                'Adjudicator': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Trial-Hearing': {
                'Defendant': {},
                'Prosecutor': {},
                'Adjudicator': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Appeal': {
                'Defendant': {},
                'Prosecutor': {},
                'Adjudicator': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Sentence': {
                'Defendant': {},
                'Adjudicator': {},
                'Crime': {},
                'Sentence': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Release-Parole': {
                'Person': {},
                'Entity': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Acquit': {
                'Defendant': {},
                'Adjudicator': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Sue': {
                'Plaintiff': {},
                'Defendant': {},
                'Adjudicator': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Arrest-Jail': {
                'Person': {},
                'Agent': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Pardon': {
                'Defendant': {},
                'Adjudicator': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Fine': {
                'Entity': {},
                'Adjudicator': {},
                'Money': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Execute': {
                'Person': {},
                'Agent': {},
                'Crime': {},
                'Time': {},
                'Place': {},
            },
            'Justice.Extradite': {
                'Agent': {},
                'Person': {},
                'Destination': {},
                'Origin': {},
                'Crime': {},
                'Time': {},
            },
        },
        'args': {}
    }

    role_to_events = {}
    for event in ontology['events'].keys():
        roles = ontology['events'][event]
        ontology['events'][event] = {'roles': {}}
        for rid, r in enumerate(roles.keys()):
            ontology['events'][event]['roles'][r] = {
                'pos': rid,
                'output_value': r.lower(),
                'constraints': []
            }
            if r not in role_to_events:
                role_to_events[r] = {'events': []}
            role_to_events[r]['events'].append(event)
    ontology['args'] = role_to_events

    return ontology


def build_bnb_ontology():
    ontology = {
        'events': {
            'loss': {
                'roles': {
                    'loss-Arg0': {'pos': 0, 'constraints': []},
                    'loss-Arg1': {'pos': 1, 'constraints': []},
                    'loss-Arg2': {'pos': 2, 'constraints': []},
                    'loss-Arg3': {'pos': 3, 'constraints': []},
                    'Extent': {'pos': 4, 'constraints': []},
                    'Location': {'pos': 5, 'constraints': []},
                    'Manner': {'pos': 6, 'constraints': []},
                    'Temporal': {'pos': 7, 'constraints': []},
                }
            },
            'fund': {
                'roles': {
                    'fund-Arg0': {'pos': 0, 'constraints': []},
                    'fund-Arg1': {'pos': 1, 'constraints': []},
                    'fund-Arg2': {'pos': 2, 'constraints': []},
                }
            },
            'loan': {
                'roles': {
                    'loan-Arg0': {'pos': 0, 'constraints': []},
                    'loan-Arg1': {'pos': 1, 'constraints': []},
                    'loan-Arg2': {'pos': 2, 'constraints': []},
                    'loan-Arg3': {'pos': 3, 'constraints': []},
                    'loan-Arg4': {'pos': 4, 'constraints': []},
                    'Purpose': {'pos': 5, 'constraints': []},
                    'Location': {'pos': 6, 'constraints': []},
                    'Manner': {'pos': 7, 'constraints': []},
                }
            },
            'plan': {
                'roles': {
                    'plan-Arg0': {'pos': 0, 'constraints': []},
                    'plan-Arg1': {'pos': 1, 'constraints': []},
                    'plan-Arg2': {'pos': 2, 'constraints': []},
                    'plan-Arg3': {'pos': 3, 'constraints': []},
                    'Negation': {'pos': 4, 'constraints': []},
                }
            },
            'price': {
                'roles': {
                    'price-Arg0': {'pos': 0, 'constraints': []},
                    'price-Arg1': {'pos': 1, 'constraints': []},
                    'price-Arg2': {'pos': 2, 'constraints': []},
                    'Manner': {'pos': 3, 'constraints': []},
                }
            },
            'cost': {
                'roles': {
                    'cost-Arg0': {'pos': 0, 'constraints': []},
                    'cost-Arg1': {'pos': 1, 'constraints': []},
                    'cost-Arg2': {'pos': 2, 'constraints': []},
                    'cost-Arg3': {'pos': 3, 'constraints': []},
                    'Location': {'pos': 4, 'constraints': []},
                    'Manner': {'pos': 5, 'constraints': []},
                }
            },
            'sale': {
                'roles': {
                    'sale-Arg0': {'pos': 0, 'constraints': []},
                    'sale-Arg1': {'pos': 1, 'constraints': []},
                    'sale-Arg2': {'pos': 2, 'constraints': []},
                    'sale-Arg3': {'pos': 3, 'constraints': []},
                    'sale-Arg4': {'pos': 4, 'constraints': []},
                    'Temporal': {'pos': 5, 'constraints': []},
                    'Location': {'pos': 6, 'constraints': []},
                    'Manner': {'pos': 7, 'constraints': []},
                }
            },
            'bid': {
                'roles': {
                    'bid-Arg0': {'pos': 0, 'constraints': []},
                    'bid-Arg1': {'pos': 1, 'constraints': []},
                    'bid-Arg2': {'pos': 2, 'constraints': []},
                    'Purpose': {'pos': 3, 'constraints': []},
                    'Manner': {'pos': 4, 'constraints': []},
                }
            },
            'investor': {
                'roles': {
                    'investor-Arg0': {'pos': 0, 'constraints': []},
                    'investor-Arg1': {'pos': 1, 'constraints': []},
                    'investor-Arg2': {'pos': 2, 'constraints': []},
                }
            },
            'investment': {
                'roles': {
                    'investment-Arg0': {'pos': 0, 'constraints': []},
                    'investment-Arg1': {'pos': 1, 'constraints': []},
                    'investment-Arg2': {'pos': 2, 'constraints': []},
                    'Temporal': {'pos': 3, 'constraints': []},
                    'Location': {'pos': 4, 'constraints': []},
                    'Manner': {'pos': 5, 'constraints': []},
                }
            },
        },
        'args': {}
    }

    # hacky workaround
    ontology['events'] = {
        event_type: {
            'roles': {
                role_label.replace(f'{event_type}-', ''): role
                for role_label, role in event['roles'].items()
            }
        }
        for event_type, event in ontology['events'].items()
    }

    role_to_events = {}
    for event in ontology['events'].keys():
        for r in ontology['events'][event]['roles'].keys():
            ontology['events'][event]['roles'][r]['output_value'] = r.lower()
            if r not in role_to_events:
                role_to_events[r] = {'events': []}
            role_to_events[r]['events'].append(event)
    ontology['args'] = role_to_events

    return ontology


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['gvdb', 'bnb', 'ace'], required=True)
    parser.add_argument('--ontology-path', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'gvdb':
        ontology = build_gvdb_ontology()
    elif args.dataset == 'bnb':
        ontology = build_bnb_ontology()
    elif args.dataset == 'ace':
        ontology = build_ace_ontology()
    else:
        raise NotImplementedError

    with open(args.ontology_path, 'w') as f:
        json.dump(ontology, f, indent=2)

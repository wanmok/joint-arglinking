import collections
from typing import *
import copy

from common.span_common import Event, Span


def scores_to_metric(scores: collections.Counter) -> Dict[str, Union[float, int]]:
    if (scores['true_positive'] + scores['false_positive']) == 0 and (
            scores['true_positive'] + scores['false_negative']) == 0:
        precision: float = 1.
        recall: float = 1.
        f1: float = 1.
    elif scores['true_positive'] == 0:
        precision: float = 0
        recall: float = 0
        f1: float = 0
    else:
        precision: float = scores['true_positive'] / (scores['true_positive'] + scores['false_positive'])
        recall: float = scores['true_positive'] / (scores['true_positive'] + scores['false_negative'])
        f1: float = 2 * precision * recall / (precision + recall)
    return {
        'prec': precision,
        'rec': recall,
        'f1': f1
    }


def micro_average_scores(counters: collections.defaultdict) -> collections.Counter:
    results: collections.Counter = collections.Counter()
    for counter in counters.values():  # collections.Counter
        results.update(counter)

    return results


def aggregate_raw_scores(
        raw_scores: List[Dict[str, Union[collections.Counter, collections.defaultdict]]]
) -> Dict[str, Union[collections.Counter, collections.defaultdict]]:
    results: Dict[str, Union[collections.Counter, collections.defaultdict]] = {
        'trigger_id': collections.Counter(),
        'trigger_id_cls': collections.defaultdict(collections.Counter),
        'argument_id': collections.Counter(),
        'argument_id_cls': collections.defaultdict(collections.Counter)
    }

    for scores in raw_scores:
        # aggregate Trigger ID
        results['trigger_id'].update(scores['trigger_id'])

        # aggregate Trigger ID + Classification
        for event_kind in scores['trigger_id_cls'].keys():
            results['trigger_id_cls'][event_kind].update(scores['trigger_id_cls'][event_kind])

        # aggregate Argument ID
        results['argument_id'].update(scores['argument_id'])

        # aggregate Trigger ID + Classification
        for role in scores['argument_id_cls'].keys():
            results['argument_id_cls'][role].update(scores['argument_id_cls'][role])

    return results


def intersect_count(a: collections.Counter, b: collections.Counter) -> collections.Counter:
    common: collections.Counter = a & b
    res: collections.Counter = collections.Counter()
    res['true_positive'] = sum(common.values())
    res['false_positive'] = sum(a.values()) - res['true_positive']
    res['false_negative'] = sum(b.values()) - res['true_positive']
    return res


def evaluate_on_span(pred_spans: List[Span],
                     gold_spans: List[Span]) -> collections.Counter:
    pred_span_counter: collections.Counter = collections.Counter(
        [(s.document.doc_key, s.start, s.end) for s in pred_spans]
    )
    gold_span_counter: collections.Counter = collections.Counter(
        [(s.document.doc_key, s.start, s.end) for s in gold_spans]
    )
    return intersect_count(pred_span_counter, gold_span_counter)


def evaluate_on_document(pred_events: List[Event],
                         gold_events: List[Event],
                         fuzzy_match: bool = False
                         ) -> Dict[str, Union[collections.Counter, collections.defaultdict]]:
    def fuzzy_process(preds: List[Event], golds: List[Event]) -> List[Event]:
        def _overlapped(a: Span, b: Span) -> bool:
            if (
                    (a.start <= b.start <= a.end <= b.end)
                    or (a.start <= b.start <= b.end <= a.end)
                    or (b.start <= a.start <= a.end <= b.end)
                    or (b.start <= a.start <= b.end <= a.end)
            ):
                return True
            else:
                return False

        new_preds: List[Event] = []
        for e in preds:
            ne: Event = copy.deepcopy(e)
            for ge in golds:
                if ne.document.doc_key == ge.document.doc_key:
                    if ne.kind == ge.kind:
                        if _overlapped(ne.trigger, ge.trigger):
                            ne.trigger.start = ge.trigger.start
                            ne.trigger.end = ge.trigger.end
                        for a in ne.arguments:
                            for ga in ge.arguments:
                                if a.role == ga.role and (_overlapped(a, ga)):
                                    a.start = ga.start
                                    a.end = ga.end
            new_preds.append(ne)
        return new_preds

    if fuzzy_match:
        pred_events = fuzzy_process(preds=pred_events, golds=gold_events)

    # Trigger ID
    pred_trigger_spans: collections.Counter = collections.Counter(
        [(e.trigger.start, e.trigger.end) for e in pred_events]
    )
    gold_trigger_spans: collections.Counter = collections.Counter(
        [(e.trigger.start, e.trigger.end) for e in gold_events]
    )
    trigger_id_counter: collections.Counter = intersect_count(pred_trigger_spans, gold_trigger_spans)

    # Trigger ID + Classification
    trigger_id_cls_counter: collections.defaultdict = collections.defaultdict(collections.Counter)
    # group Event by kind
    pred_type_events: collections.defaultdict = collections.defaultdict(list)  # Dict[str, List[Event]]
    gold_type_events: collections.defaultdict = collections.defaultdict(list)  # Dict[str, List[Event]]
    for e in pred_events:
        pred_type_events[e.kind].append(e)
    for e in gold_events:
        gold_type_events[e.kind].append(e)
    all_event_types: Set[str] = set(pred_type_events.keys()) | set(gold_type_events.keys())

    for etype in all_event_types:
        pred_trigger_spans: collections.Counter = collections.Counter(
            [(e.trigger.start, e.trigger.end) for e in pred_type_events[etype]]
        )
        gold_trigger_spans: collections.Counter = collections.Counter(
            [(e.trigger.start, e.trigger.end) for e in gold_type_events[etype]]
        )
        trigger_id_cls_counter[etype] = intersect_count(pred_trigger_spans, gold_trigger_spans)

        # common_trigger_spans: collections.Counter = pred_trigger_spans & gold_trigger_spans
        #
        # trigger_id_cls_counter[etype]['true_positive'] = sum(common_trigger_spans.values())
        # trigger_id_cls_counter[etype]['false_positive'] = (
        #         sum(pred_trigger_spans.values()) - trigger_id_cls_counter[etype]['true_positive']
        # )
        # trigger_id_cls_counter[etype]['false_negative'] = (
        #         sum(gold_trigger_spans.values()) - trigger_id_cls_counter[etype]['true_positive']
        # )

    # Argument ID
    pred_arg_spans: collections.Counter = collections.Counter(
        [
            (e.kind, arg.start, arg.end)
            # (e.kind, str(arg))
            for e in pred_events
            for arg in e.arguments
        ]
    )
    gold_arg_spans: collections.Counter = collections.Counter(
        [
            (e.kind, arg.start, arg.end)
            # (e.kind, str(arg))
            for e in gold_events
            for arg in e.arguments
        ]
    )
    argument_id_counter: collections.Counter = intersect_count(pred_arg_spans, gold_arg_spans)

    # Argument ID + Classification
    argument_id_cls_counter: collections.defaultdict = collections.defaultdict(collections.Counter)
    # group Arguments by role
    pred_role_arg_spans: collections.defaultdict = collections.defaultdict(list)
    gold_role_arg_spans: collections.defaultdict = collections.defaultdict(list)
    for e in pred_events:
        for arg in e.arguments:
            pred_role_arg_spans[arg.role.lower()].append((e.kind, arg.start, arg.end))
            # pred_role_arg_spans[arg.role].append((e.kind, str(arg)))
    for e in gold_events:
        for arg in e.arguments:
            gold_role_arg_spans[arg.role.lower()].append((e.kind, arg.start, arg.end))
            # gold_role_arg_spans[arg.role].append((e.kind, str(arg)))
    all_role_types: Set[str] = set(pred_role_arg_spans.keys()) | set(gold_role_arg_spans.keys())

    for role in all_role_types:
        pred_arg_spans: collections.Counter = collections.Counter(pred_role_arg_spans[role])
        gold_arg_spans: collections.Counter = collections.Counter(gold_role_arg_spans[role])
        argument_id_cls_counter[role] = intersect_count(pred_arg_spans, gold_arg_spans)

    return {
        'trigger_id': trigger_id_counter,
        'trigger_id_cls': trigger_id_cls_counter,
        'argument_id': argument_id_counter,
        'argument_id_cls': argument_id_cls_counter
    }


def event_metric_to_text(trigger_id, trigger_id_cls, arg_id, arg_id_cls) -> str:
    header: str = '| Metric | Precision | Recall | F1 |'
    table_h: str = '|-------|-----------|--------|----|'
    tid: str = f'| Trigger ID | {trigger_id["prec"]} | {trigger_id["rec"]} | {trigger_id["f1"]} |'
    tidcls: str = f'| Trigger ID + CLS | {trigger_id_cls["prec"]} | {trigger_id_cls["rec"]} | {trigger_id_cls["f1"]} |'
    aid: str = f'| Argument ID | {arg_id["prec"]} | {arg_id["rec"]} | {arg_id["f1"]} |'
    aidcls: str = f'| Argument ID + CLS | {arg_id_cls["prec"]} | {arg_id_cls["rec"]} | {arg_id_cls["f1"]} |'

    return '\n'.join([header, table_h, tid, tidcls, aid, aidcls])


def emd_metric_to_text(emd) -> str:
    header: str = '| Metric | Precision | Recall | F1 |'
    table_h: str = '|-------|-----------|--------|----|'
    eid: str = f'| Entity Mention Detection | {emd["prec"]} | {emd["rec"]} | {emd["f1"]} |'

    return '\n'.join([header, table_h, eid])


def micro_scores_to_text(scores) -> str:
    header: str = '| Metric | Precision | Recall | F1 |'
    table_h: str = '|-------|-----------|--------|----|'
    lines: List[str] = [header, table_h]
    for k, v in scores.items():
        lines.append(
            f'| {k} | {v["prec"]} | {v["rec"]} | {v["f1"]}'
        )
    return '\n'.join(lines)


def result_text_to_summary(scores: Dict) -> str:
    desc: str = (
            f'Dev results:\n\n' +
            f'Total:\t{scores["total"]}\n\n' +
            f'ExactMatch:\t{scores["exact"]}\n\n' +
            f'F1:\t{scores["f1"]}, Precission:\t{scores["prec"]}, Recall:\t{scores["rec"]}\n\n'
    )
    return desc

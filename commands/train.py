import argparse
import collections
import copy
import json
import logging
import os
import pickle
from itertools import groupby
from typing import *

import numpy as np
import torch
from allennlp.data import Vocabulary
from torch import nn
from torch.nn import Parameter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup

from common.span_common import Event, Argument, Span
from dataset.concrete_event_dataset import ConcreteDataset
from dataset.input_instance import InputInstance
from metrics.event_metric import scores_to_metric, micro_average_scores, aggregate_raw_scores, evaluate_on_document, \
    event_metric_to_text, micro_scores_to_text, evaluate_on_span, emd_metric_to_text
from models.argument_span_classifier import ArgumentSpanClassifier
from models.selector_arglinking import SelectorArgLinking
from modules.span_finder import SpanFinder, PredictiveSpanFinder

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def serialize_model(model: nn.Module,
                    vocab: Vocabulary,
                    output_dir: str):
    output_model_file = os.path.join(output_dir, 'model.pickle')
    vocab.save_to_files(os.path.join(output_dir, 'vocab'))
    torch.save(model.state_dict(), output_model_file)


def compare_model(scores: Dict[str, Union[int, float]],
                  best_scores: Dict[str, Union[int, float]]) -> bool:
    return scores['f1'] > best_scores['f1']


def eval_on_rams(
        pred_events: List[Event],
        gold_events: List[Event],
        num_epoch: int,
        writer: Optional['SummaryWriter'] = None,
        save_raw_scores: bool = False
) -> Dict[str, Union[int, float]]:
    pred_events: Dict[str, List[Event]] = {
        k: list(events)
        for k, events in groupby(pred_events, key=lambda e: e.document.doc_key)
    }
    gold_events: Dict[str, List[Event]] = {
        k: list(events)
        for k, events in groupby(gold_events, key=lambda e: e.document.doc_key)
    }

    # evaluate on the single document and get raw scores
    raw_scores: Dict[str, Dict[str, Union[collections.Counter, collections.defaultdict]]] = {}
    for doc_key in pred_events.keys():
        raw_scores[doc_key] = evaluate_on_document(pred_events=pred_events[doc_key],
                                                   gold_events=gold_events[doc_key],
                                                   fuzzy_match=fuzzy_match)

    # aggregated raw scores from individual documents
    aggregated_scores: Dict[str, Union[collections.Counter, collections.defaultdict]] = aggregate_raw_scores(
        list(raw_scores.values())
    )

    if save_raw_scores:
        with open(os.path.join(args.model_serialization, f'raw_scores_{num_epoch}.pickle'), 'wb') as f:
            pickle.dump(aggregated_scores, f)

    # convert raw scores to metric
    trigger_id_metric: Dict[str, Union[int, float]] = scores_to_metric(aggregated_scores['trigger_id'])
    argument_id_metric: Dict[str, Union[int, float]] = scores_to_metric(aggregated_scores['argument_id'])

    # compute micro-averaged scores
    averaged_trigger_id_cls: Dict[str, Union[int, float]] = scores_to_metric(
        micro_average_scores(aggregated_scores['trigger_id_cls'])
    )
    averaged_argument_id_cls: Dict[str, Union[int, float]] = scores_to_metric(
        micro_average_scores(aggregated_scores['argument_id_cls'])
    )

    logger.info(f'Evaluated on the dataset:')
    logger.info('OVERALL')
    logger.info('Metric\tPrecision\tRecall\tF1')
    logger.info(f'TriggerID\t{trigger_id_metric["prec"]}\t{trigger_id_metric["rec"]}\t{trigger_id_metric["f1"]}')
    logger.info(f'TriggerCLS\t{averaged_trigger_id_cls["prec"]}' +
                f'\t{averaged_trigger_id_cls["rec"]}\t{averaged_trigger_id_cls["f1"]}')
    logger.info(f'ArgID\t{argument_id_metric["prec"]}\t{argument_id_metric["rec"]}\t{argument_id_metric["f1"]}')
    logger.info(f'ArgCLS\t{averaged_argument_id_cls["prec"]}' +
                f'\t{averaged_argument_id_cls["rec"]}\t{averaged_argument_id_cls["f1"]}')

    logger.info('TRIGGER ID CLS')
    logger.info('Metric\t\tPrecision\tRecall\tF1')
    micro_trigger_id_cls_scores: Dict[str, Dict[str, Union[float, int]]] = {}
    for etype, counter in aggregated_scores['trigger_id_cls'].items():
        micro_trigger_id_cls_scores[etype] = scores_to_metric(counter)
        logger.info(f'{etype}\t\t{micro_trigger_id_cls_scores[etype]["prec"]}' +
                    f'\t{micro_trigger_id_cls_scores[etype]["rec"]}\t{micro_trigger_id_cls_scores[etype]["f1"]}')

    logger.info('ARGUMENT ID CLS')
    logger.info('Metric\t\tPrecision\tRecall\tF1')
    micro_argument_id_cls_scores: Dict[str, Dict[str, Union[float, int]]] = {}
    for role, counter in aggregated_scores['argument_id_cls'].items():
        micro_argument_id_cls_scores[role] = scores_to_metric(counter)
        logger.info(f'{role}\t\t{micro_argument_id_cls_scores[role]["prec"]}' +
                    f'\t{micro_argument_id_cls_scores[role]["rec"]}\t{micro_argument_id_cls_scores[role]["f1"]}')

    if writer:
        overall_metric_text: str = event_metric_to_text(trigger_id_metric,
                                                        averaged_trigger_id_cls,
                                                        argument_id_metric,
                                                        averaged_argument_id_cls)
        writer.add_text(f'Metric/Overall',
                        overall_metric_text,
                        num_epoch)

        writer.add_scalar('Trigger_ID/Precision', trigger_id_metric['prec'], num_epoch)
        writer.add_scalar('Trigger_ID/Recall', trigger_id_metric['rec'], num_epoch)
        writer.add_scalar('Trigger_ID/F1', trigger_id_metric['f1'], num_epoch)

        writer.add_scalar('Trigger_ID_CLS/Precision', averaged_trigger_id_cls['prec'], num_epoch)
        writer.add_scalar('Trigger_ID_CLS/Recall', averaged_trigger_id_cls['rec'], num_epoch)
        writer.add_scalar('Trigger_ID_CLS/F1', averaged_trigger_id_cls['f1'], num_epoch)

        writer.add_scalar('Argument_ID/Precision', argument_id_metric['prec'], num_epoch)
        writer.add_scalar('Argument_ID/Recall', argument_id_metric['rec'], num_epoch)
        writer.add_scalar('Argument_ID/F1', argument_id_metric['f1'], num_epoch)

        writer.add_scalar('Argument_ID_CLS/Precision', averaged_argument_id_cls['prec'], num_epoch)
        writer.add_scalar('Argument_ID_CLS/Recall', averaged_argument_id_cls['rec'], num_epoch)
        writer.add_scalar('Argument_ID_CLS/F1', averaged_argument_id_cls['f1'], num_epoch)

        writer.add_text('Metric/Trigger_ID_CLS',
                        micro_scores_to_text(micro_trigger_id_cls_scores),
                        num_epoch)

        writer.add_text('Metric/Argument_ID_CLS',
                        micro_scores_to_text(micro_argument_id_cls_scores),
                        num_epoch)

    return averaged_argument_id_cls


def eval_on_emd(
        pred_spans: List[Span],
        gold_spans: List[Span],
        num_epoch: int,
        writer: Optional['SummaryWriter'] = None,
        save_raw_scores: bool = False
) -> Dict[str, Union[int, float]]:
    raw_scores: collections.Counter = evaluate_on_span(pred_spans=pred_spans,
                                                       gold_spans=gold_spans)

    if save_raw_scores:
        with open(os.path.join(args.model_serialization, f'raw_scores_{num_epoch}_emd.pickle'), 'wb') as f:
            pickle.dump(raw_scores, f)

    # convert raw scores to metric
    emd_metric: Dict[str, Union[int, float]] = scores_to_metric(raw_scores)

    logger.info(f'Evaluated on the dataset:')
    logger.info('OVERALL')
    logger.info('Metric\tPrecision\tRecall\tF1')
    logger.info(f'EMD\t{emd_metric["prec"]}\t{emd_metric["rec"]}\t{emd_metric["f1"]}')

    if writer:
        overall_metric_text: str = emd_metric_to_text(emd_metric)
        writer.add_text(f'Metric/Overall',
                        overall_metric_text,
                        num_epoch)

        writer.add_scalar('EMD/Precision', emd_metric['prec'], num_epoch)
        writer.add_scalar('EMD/Recall', emd_metric['rec'], num_epoch)
        writer.add_scalar('EMD/F1', emd_metric['f1'], num_epoch)

    return emd_metric


def build_events_from_inputs_outputs(dataset: ConcreteDataset,
                                     inputs: InputInstance,
                                     outputs: Dict[str, Union[torch.Tensor, list]]) -> Tuple[List[Event], List[Event]]:
    gold_events: List[Event] = [
        dataset.ins_to_event[i]
        for i in inputs.id.tolist()
    ]
    pred_events: List[Event] = [
        Event(kind=gold_events[i].kind,
              trigger=gold_events[i].trigger,
              document=gold_events[i].document,
              arguments=[
                  Argument(
                      start=inputs.span_indices[i, j + 1, 0].item() + (gold_events[i].document.sentence_offsets[
                                                                           dataset.metadata[id][
                                                                               'sentence_id']] if sentence_mode else 0),
                      end=inputs.span_indices[i, j + 1, 1].item() + (gold_events[i].document.sentence_offsets[
                                                                         dataset.metadata[id][
                                                                             'sentence_id']] if sentence_mode else 0),
                      role=outputs['type_strs'][i][j],
                      document=gold_events[i].document
                  )
                  for j in range(len(outputs['type_strs'][i]))
                  if outputs['type_strs'][i][j] != 'None'
              ])
        for i, id in enumerate(inputs.id.tolist())
    ]

    return gold_events, pred_events


def build_spans_from_inputs_outputs(dataset: ConcreteDataset,
                                    inputs: InputInstance,
                                    outputs: Dict[str, Union[torch.Tensor, list]]) -> Tuple[List[Span], List[Span]]:
    input_ids: List[int] = inputs.id.tolist()
    gold_spans: List[Span] = []
    for i in input_ids:
        gold_spans.extend(dataset.ins_to_event[i].arguments)

    pred_spans: List[Span] = []
    for i, spans in enumerate(outputs['span_indices']):
        doc = dataset.ins_to_event[input_ids[i]].document
        pred_spans.extend([
            Span(start=span[0],
                 end=span[1],
                 document=doc)
            for span in spans
        ])

    return gold_spans, pred_spans


def output_as_jsonline(pred_events: List[Event], path: str):
    buffer = []
    for event in pred_events:
        doc = event.document
        jdoc = {
            'doc_key': doc.doc_key,
            'predictions': [[
                [int(event.trigger.start), int(event.trigger.end)],
            ]]
        }
        for arg in event.arguments:
            jdoc['predictions'][0].append([
                int(arg.start),
                int(arg.end),
                # ontology['events'][event.kind]['roles'][arg.role]['output_value'],
                arg.role.lower(),
                1.0
            ])

        buffer.append(jdoc)

    with open(path, 'w') as f:
        f.writelines('\n'.join([json.dumps(b) for b in buffer]))


def train_or_test():
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.model_serialization)
    else:
        writer = None

    if not args.test_mode:
        train_data: ConcreteDataset = ConcreteDataset.from_concrete(data_path=args.train_data,
                                                                    cache_file=args.train_cache_file,
                                                                    vocab=vocab,
                                                                    ontology=ontology,
                                                                    task=args.task,
                                                                    max_num_spans=max_num_spans,
                                                                    sentence_mode=sentence_mode)
        dev_data: ConcreteDataset = ConcreteDataset.from_concrete(data_path=args.dev_data,
                                                                  cache_file=args.dev_cache_file,
                                                                  vocab=vocab,
                                                                  ontology=ontology,
                                                                  task=args.task,
                                                                  max_num_spans=max_num_spans,
                                                                  sentence_mode=sentence_mode)
        dataloaders: Dict[str, DataLoader] = {
            'train': DataLoader(train_data,
                                collate_fn=InputInstance.collate,
                                sampler=RandomSampler(train_data),
                                batch_size=args.train_batch_size),
            'dev': DataLoader(dev_data,
                              collate_fn=InputInstance.collate,
                              batch_size=args.dev_batch_size)
        }

        num_train_optimization_steps = len(
            dataloaders['train']) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay: List[str] = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters: List[Dict[str, List[Parameter]]] = [
            {'params': [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
                'weight_decay': args.weight_decay},
            {'params': [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
                'weight_decay': 0.0}
        ]

        t_total: int = len(dataloaders['train']) // args.gradient_accumulation_steps * args.num_train_epochs

        num_warmup_steps: int = args.warmup_steps * t_total

        optimizer: AdamW = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler: LambdaLR = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )

        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
    else:
        test_data: ConcreteDataset = ConcreteDataset.from_concrete(data_path=args.test_data,
                                                                   cache_file=args.test_cache_file,
                                                                   vocab=vocab,
                                                                   ontology=ontology,
                                                                   task=args.task,
                                                                   max_num_spans=max_num_spans,
                                                                   sentence_mode=sentence_mode)
        dataloaders: Dict[str, DataLoader] = {
            'test': DataLoader(test_data,
                               collate_fn=InputInstance.collate,
                               batch_size=args.test_batch_size)
        }

    best_model: Optional[nn.Module] = None
    best_scores: Optional[Dict[str, Union[int, float]]] = None
    global_step: int = 1
    val_global_step: int = 1

    early_stopping_flags: List[float] = []

    phases: List[str] = ['train', 'dev'] if not args.test_mode else ['test']
    model.zero_grad()
    for num_epoch in trange(int(args.num_train_epochs) if not args.test_mode else 1, desc="Epoch"):
        for phase in phases:
            # for phase in ['train']:
            # for phase in ['dev']:

            if phase == 'train':
                dataloader: DataLoader = dataloaders[phase]
                iteration_tqdm = tqdm(dataloader, desc='Iteration')

                model.train()

                for step, batch in enumerate(iteration_tqdm):
                    # if step == 10:
                    #     break

                    model_inputs: Dict[str, torch.Tensor] = batch.to_device(device)

                    with torch.enable_grad():
                        outputs: Dict[str, Union[torch.Tensor, List]] = model(**model_inputs)

                        loss: torch.Tensor = outputs['loss']  # scalar

                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        if torch.isnan(loss):
                            logger.info(f'loss becomes {loss.item()}, step: {step}')

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            optimizer.step()
                            scheduler.step()  # Update learning rate schedule
                            model.zero_grad()

                            iteration_tqdm.set_postfix(loss=loss.item())

                            if writer:
                                writer.add_scalar('Loss/Train',
                                                  loss.item(),
                                                  global_step)

                            global_step += 1
            else:
                model.eval()
                dataloader: DataLoader = dataloaders[phase]
                iteration_tqdm = tqdm(dataloader, desc='Iteration')

                if args.task in ['argid', 'argidcls', 'argcls', 'argidcls-noisy']:
                    gold_events: List[Event] = []
                    pred_events: List[Event] = []
                else:  # EMD
                    gold_spans: List[Span] = []
                    pred_spans: List[Span] = []

                with torch.no_grad():
                    for step, batch in enumerate(iteration_tqdm):
                        # if step == 5:
                        #     break

                        model_inputs: Dict[str, torch.Tensor] = batch.to_device(device)

                        with torch.no_grad():
                            outputs: Dict[str, Union[torch.Tensor, List]] = model(**model_inputs)

                            loss: torch.Tensor = outputs['loss']  # scalar

                            iteration_tqdm.set_postfix(loss=loss.item())

                            if writer:
                                writer.add_scalar('Loss/Dev',
                                                  loss.item(),
                                                  val_global_step)

                                val_global_step += 1

                            from_dataset = dev_data if phase == 'dev' else test_data

                            if args.task in ['argid', 'argidcls', 'argcls', 'argidcls-noisy']:
                                new_golds, new_preds = build_events_from_inputs_outputs(inputs=batch,
                                                                                        outputs=outputs,
                                                                                        dataset=from_dataset)
                                gold_events.extend(new_golds)
                                pred_events.extend(new_preds)

                                for i in range(len(new_golds)):
                                    logger.info(
                                        f'doc_key: {new_golds[i].document.doc_key}, event_kid: {new_golds[i].kind}')
                                    seen_arg = set()
                                    for arg in new_golds[i].arguments:
                                        for parg in new_preds[i].arguments:
                                            if (arg.start, arg.end) == (parg.start, parg.end):
                                                logger.info(f'Gold: {arg}\tPred: {parg}')
                                                seen_arg.add((arg.start, arg.end))
                                                break
                                        if (arg.start, arg.end) not in seen_arg:
                                            logger.info(f'Gold: {arg}\tPred: NULL')
                                            seen_arg.add((arg.start, arg.end))
                                    for parg in new_preds[i].arguments:
                                        if (parg.start, parg.end) not in seen_arg:
                                            logger.info(f'Gold: NULL\tPred: {parg}')
                            else:  # EMD
                                new_golds, new_preds = build_spans_from_inputs_outputs(inputs=batch,
                                                                                       outputs=outputs,
                                                                                       dataset=from_dataset)
                                gold_spans.extend(new_golds)
                                pred_spans.extend(new_preds)
                                for doc, gspans in groupby(new_golds, key=lambda s: s.document.doc_key):
                                    logger.info(f'doc_key: {doc.doc_key}')
                                    seen_arg = set()
                                    for gspan in gspans:
                                        for pspan in new_preds:
                                            if pspan == gspan:
                                                logger.info(f'Gold: {gspan}\tPred: {pspan}')
                                                seen_arg.add((gspan.start, gspan.end))
                                        if (gspan.start, gspan.end) not in seen_arg:
                                            logger.info(f'Gold: {gspan}\tPred: NULL')
                                    for pspan in new_preds:
                                        if (pspan.document.doc_key == doc.doc_key and (
                                                pspan.start, pspan.end) not in seen_arg):
                                            logger.info(f'Gold: NULL\tPred: {pspan}')

                if args.task in ['argid', 'argidcls', 'argcls', 'argidcls-noisy']:
                    scores: Dict[str, Union[int, float]] = eval_on_rams(pred_events=pred_events,
                                                                        gold_events=gold_events,
                                                                        num_epoch=num_epoch,
                                                                        writer=writer)
                else:
                    scores = eval_on_emd(pred_spans=pred_spans,
                                         gold_spans=gold_spans,
                                         num_epoch=num_epoch,
                                         writer=writer)

                if phase == 'test' and output_path_json is not None:
                    output_as_jsonline(pred_events=pred_events, path=output_path_json)

                if phase != 'test' and (best_model is None or compare_model(
                        scores=scores,
                        best_scores=best_scores)):
                    best_scores = scores
                    best_model: nn.Module = copy.deepcopy(model)
                    model_path = args.model_serialization
                    serialize_model(model=best_model,
                                    vocab=vocab,
                                    output_dir=model_path)

                if phase != 'test' and args.early_stopping != -1:
                    if len(early_stopping_flags) < 2:
                        early_stopping_flags.append(scores['f1'])
                    else:
                        if (abs(early_stopping_flags[-1] - scores['f1']) < args.early_stopping and
                                abs(early_stopping_flags[-2] - scores['f1']) < args.early_stopping
                        ):
                            logger.info('Early stopping triggered, stop training.')

                            if writer:
                                writer.close()

                            return

    if writer:
        writer.close()


def read_label_weights(weights_path: str) -> Dict[str, float]:
    with open(weights_path) as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, help='Train data file path')
    parser.add_argument('--train-cache-file', type=str, help='Train cache file path')
    parser.add_argument('--dev-data', type=str, help='Dev data file path')
    parser.add_argument('--dev-cache-file', type=str, help='Dev cache file path')
    parser.add_argument('--test-data', type=str, help='Test data file path')
    parser.add_argument('--test-cache-file', type=str, help='Test cache file path')
    parser.add_argument('--ontology-path', type=str, help='The path to ontology')
    parser.add_argument('--label-weights-path', type=str, help='The path to label weights rescaling file')
    parser.add_argument('--model-serialization', type=str, help='The path used to store the model')
    parser.add_argument('--train-batch-size', type=int, default=6, help='Training batch size')
    parser.add_argument('--dev-batch-size', type=int, default=6, help='Dev batch size')
    parser.add_argument('--test-batch-size', type=int, default=6, help='Test batch size')
    parser.add_argument('--num-train-epochs', default=50.0, type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--early-stopping', type=float, default=0.01, help='-1 means no early stopping.')
    parser.add_argument('--learning-rate', default=5e-5, type=float, help='The initial learning rate for Adam.')
    parser.add_argument("--weight-decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument('--warmup-steps', default=0, type=float,
                        help='Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% '
                             'of training.')
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=5.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--max-num-spans', type=int, default=512, help='Maximum number of spans')
    parser.add_argument('--fuzzy-match', type=str, help='Whether to do fuzzy match during eval',
                        choices=['True', 'False'], default='False')
    parser.add_argument('--use-span-encoder', type=str, help='Whether to enable span encoder',
                        choices=['True', 'False'], default='True')
    parser.add_argument('--test-mode', action='store_true', help='Using test mode')
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--task', type=str, help='Choose which task to perform',
                        choices=['argidcls-noisy', 'argidcls', 'argcls', 'emd'], default='argidcls')

    parser.add_argument('--num-transformer-layers', type=int, default=3, help='Number of Transformer layers')
    parser.add_argument('--num-transformer-heads', type=int, default=12, help='Number of Transformer heads')
    parser.add_argument('--seed', type=int, default=7168)
    parser.add_argument('--tensorboard', action='store_true', help='Whether to use TensorBoard')
    parser.add_argument('--sentence-mode', action='store_true', help='Whether to use sentence mode')
    parser.add_argument('--use-event-embedding', help='Whether to use event embedding',
                        choices=['True', 'False'], default='True', type=str)
    parser.add_argument('--output-path-json', type=str, required=False)
    parser.add_argument('--use-asc-model', action='store_true')
    parser.add_argument('--cuda', action='store_true', help='Whether to use CUDA')
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parse control arguments
    device: str = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    fuzzy_match: bool = True if args.fuzzy_match == 'True' else False
    use_span_encoder: bool = True if args.use_span_encoder == 'True' else False
    max_num_spans: int = args.max_num_spans
    sentence_mode: bool = args.sentence_mode
    use_event_embedding: bool = True if args.use_event_embedding == 'True' else False
    use_asc_model: bool = args.use_asc_model
    output_path_json: Optional[str] = args.output_path_json

    logger.info(f'Using device: {device}')

    # load Vocabulary and ontology, build model
    vocab, ontology = ConcreteDataset.build_ontology_and_vocab(
        ontology_path=args.ontology_path,
        vocab_path=None if not args.test_mode else os.path.join(args.model_serialization, 'vocab')
    )
    model_params: Dict[str, Any] = {
        'vocab': vocab,
        'label_namespace': 'span_labels',
        'embed_dim': 768,
        'num_layers': args.num_transformer_layers,
        'dim_feedforward': 2048,
        'nhead': args.num_transformer_heads,
        'activation': 'gelu',
        'use_span_encoder': use_span_encoder,
        'use_event_embedding': use_event_embedding,
        'label_rescaling_weights': read_label_weights(args.label_weights_path) if args.label_weights_path else None
    }

    if args.task in ['argidcls', 'argcls', 'argidcls-noisy']:
        if use_asc_model:
            model: ArgumentSpanClassifier = ArgumentSpanClassifier.from_params(**model_params)
        else:
            model: SelectorArgLinking = SelectorArgLinking.from_params(**model_params)
    elif args.task in ['emd']:
        model: SpanFinder = PredictiveSpanFinder(input_dim=model_params['embed_dim'])

    if args.test_mode:
        model.load_state_dict(torch.load(os.path.join(args.model_serialization, 'model.pickle'),
                                         map_location=device))
    elif args.pretrained_model is not None:
        model.load_state_dict(torch.load(os.path.join(args.pretrained_model, 'model.pickle'),
                                         map_location=device))

    model.to(device)

    train_or_test()

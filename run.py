import argparse
import logging
import pickle
import random
import sys

import numpy as np
import torch
import transformers
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tq
from transformers import AdamW

from const import *
from learning.dataset import TaggingDataset
from learning.evaluate import predict, calc_parse_eval, calc_tag_accuracy, report_eval_loss
from learning.learn import ModelForTetratagging, BertCRFModel, BertLSTMModel
from tagging.srtagger import SRTaggerBottomUp, SRTaggerTopDown
from tagging.tetratagger import BottomUpTetratagger

# Set random seed
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

logging.getLogger().setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "data/spmrl/"

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')
train = subparser.add_parser('train')
evaluate = subparser.add_parser('evaluate')
vocab = subparser.add_parser('vocab')

vocab.add_argument('--tagger', choices=[TETRATAGGER, TD_SR, BU_SR], required=True,
                   help="Tagging schema")
vocab.add_argument('--lang', choices=LANG, default=ENG, help="Language")
vocab.add_argument('--output-path', choices=[TETRATAGGER, TD_SR, BU_SR],
                   default="data/vocab/")

train.add_argument('--tagger', choices=[TETRATAGGER, TD_SR, BU_SR], required=True,
                   help="Tagging schema")
train.add_argument('--lang', choices=LANG, default=ENG, help="Language")
train.add_argument('--tag-vocab-path', type=str, default="data/vocab/")
train.add_argument('--model', choices= BERT + BERTCRF + BERTLSTM, required=True,
                   help="Model architecture")


train.add_argument('--model-path', type=str, default='bertlarge',
                   help="Bert model path or name, "
                        "bert-large-cased for english and xlm-roberta-large for others")
train.add_argument('--output-path', type=str, default='pat-models/',
                   help="Path to save trained models")
train.add_argument('--use-tensorboard', type=bool, default=False,
                   help="Whether to use the tensorboard for logging the results make sure to "
                        "add credentials to run.py if set to true")

train.add_argument('--max-depth', type=int, default=5,
                   help="Max stack depth used for decoding")
train.add_argument('--keep-per-depth', type=int, default=1,
                   help="Max elements to keep per depth")

train.add_argument('--lr', type=float, default=5e-5)
train.add_argument('--epochs', type=int, default=4)
train.add_argument('--batch-size', type=int, default=16)
train.add_argument('--num-warmup-steps', type=int, default=160)
train.add_argument('--weight-decay', type=float, default=0.01)

evaluate.add_argument('--model-name', type=str, required=True)
evaluate.add_argument('--lang', choices=LANG, default=ENG, help="Language")
evaluate.add_argument('--tag-vocab-path', type=str, default="data/vocab/")
evaluate.add_argument('--model-path', type=str, default='pat-models/')
evaluate.add_argument('--bert-model-path', type=str, default='mbert/')
evaluate.add_argument('--output-path', type=str, default='results/')
evaluate.add_argument('--batch-size', type=int, default=16)
evaluate.add_argument('--max-depth', type=int, default=5,
                      help="Max stack depth used for decoding")
evaluate.add_argument('--is-greedy', type=bool, default=False,
                      help="Whether or not to use greedy decoding")
evaluate.add_argument('--keep-per-depth', type=int, default=1,
                   help="Max elements to keep per depth")
evaluate.add_argument('--use-tensorboard', type=bool, default=False,
                      help="Whether to use the tensorboard for logging the results make sure "
                           "to add credentials to run.py if set to true")


def initialize_tag_system(reader, tagging_schema, lang, tag_vocab_path="",
                          add_remove_top=False):
    tag_vocab = None
    if tag_vocab_path != "":
        with open(tag_vocab_path + lang + "-" + tagging_schema + '.pkl', 'rb') as f:
            tag_vocab = pickle.load(f)
    if tagging_schema == BU_SR:
        tag_system = SRTaggerBottomUp(trees=reader.parsed_sents(lang + '.train'),
                                      tag_vocab=tag_vocab,
                                      add_remove_top=add_remove_top)
    elif tagging_schema == TD_SR:
        tag_system = SRTaggerTopDown(trees=reader.parsed_sents(lang + '.train'),
                                     tag_vocab=tag_vocab,
                                     add_remove_top=add_remove_top)
    elif tagging_schema == TETRATAGGER:
        tag_system = BottomUpTetratagger(trees=reader.parsed_sents(lang + '.train'),
                                         tag_vocab=tag_vocab, add_remove_top=add_remove_top)
    else:
        logging.error("Please specify the tagging schema")
        return
    return tag_system


def save_vocab(args):
    reader = BracketParseCorpusReader(DATA_PATH, [args.lang + '.train', args.lang + '.dev',
                                                  args.lang + '.test'])
    tag_system = initialize_tag_system(reader, args.tagger, args.lang,
                                       add_remove_top=args.lang == ENG)
    with open(args.output_path + args.lang + "-" + args.tagger + '.pkl', 'wb') as f:
        pickle.dump(tag_system.tag_vocab, f)


def prepare_training_data(reader, tag_system, tagging_schema, model_name, batch_size, lang):
    is_tetratags = True if tagging_schema == TETRATAGGER else False
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, truncation=True,
                                                           use_fast=True)
    train_dataset = TaggingDataset(lang + '.train', tokenizer, tag_system, reader, device,
                                   is_tetratags=is_tetratags)
    eval_dataset = TaggingDataset(lang + '.dev', tokenizer, tag_system, reader, device,
                                  pad_to_len=256, is_tetratags=is_tetratags)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=train_dataset.collate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate
    )
    return train_dataset, eval_dataset, train_dataloader, eval_dataloader


def prepare_test_data(reader, tag_system, tagging_schema, model_name, batch_size, lang):
    is_tetratags = True if tagging_schema == TETRATAGGER else False
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, truncation=True,
                                                           use_fast=True)
    test_dataset = TaggingDataset(lang + '.test', tokenizer, tag_system, reader, device,
                                  pad_to_len=256, is_tetratags=is_tetratags)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate
    )
    return test_dataset, test_dataloader


def generate_config(model_type, tagging_schema, tag_system, model_path, is_eng):
    if model_type in BERTCRF or model_type in BERTLSTM:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            num_labels=2 * len(tag_system.tag_vocab),
            task_specific_params={
                'model_path': model_path,
                'num_tags': len(tag_system.tag_vocab),
                'is_eng': is_eng,
            }
        )
    elif model_type in BERT and tagging_schema == TETRATAGGER:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            num_labels=len(tag_system.tag_vocab),
            id2label={i: label for i, label in enumerate(tag_system.tag_vocab)},
            label2id={label: i for i, label in enumerate(tag_system.tag_vocab)},
            task_specific_params={
                'model_path': model_path,
                'num_even_tags': tag_system.decode_moderator.leaf_tag_vocab_size,
                'num_odd_tags': tag_system.decode_moderator.internal_tag_vocab_size,
                'is_eng': is_eng
            }
        )
    elif model_type in BERT and tagging_schema != TETRATAGGER:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            num_labels=2 * len(tag_system.tag_vocab),
            task_specific_params={
                'model_path': model_path,
                'num_even_tags': len(tag_system.tag_vocab),
                'num_odd_tags': len(tag_system.tag_vocab),
                'is_eng': is_eng
            }
        )
    else:
        logging.error("Invalid combination of model type and tagging schema")
        return
    return config


def initialize_model(model_type, tagging_schema, tag_system, model_path, is_eng):
    config = generate_config(model_type, tagging_schema, tag_system, model_path, is_eng)
    if model_type in BERTCRF:
        model = BertCRFModel(config=config)
    elif model_type in BERTLSTM:
        model = BertLSTMModel(config=config)
    elif model_type in BERT:
        model = ModelForTetratagging(config=config)
    else:
        logging.error("Invalid model type")
        return
    return model


def initialize_optimizer_and_scheduler(model, dataset_size, lr=5e-5, num_epochs=4,
                                       num_warmup_steps=160, weight_decay_rate=0.0):
    num_training_steps = num_epochs * dataset_size
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_rate)
    scheduler = transformers.get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler, num_training_steps


def register_run_metrics(writer, run_name, lr, epochs, eval_loss, even_tag_accuracy,
                         odd_tag_accuracy):
    writer.add_hparams({'run_name': run_name, 'lr': lr, 'epochs': epochs},
                       {'eval_loss': eval_loss, 'odd_tag_accuracy': odd_tag_accuracy,
                        'even_tag_accuracy': even_tag_accuracy})


def train(args):
    reader = BracketParseCorpusReader(DATA_PATH, [args.lang + '.train', args.lang + '.dev',
                                                  args.lang + '.test'])
    logging.info("Initializing Tag System")
    tag_system = initialize_tag_system(reader, args.tagger, args.lang,
                                       tag_vocab_path=args.tag_vocab_path,
                                       add_remove_top=args.lang == ENG)
    logging.info("Preparing Data")
    train_dataset, eval_dataset, train_dataloader, eval_dataloader = prepare_training_data(
        reader,
        tag_system, args.tagger, args.model_path, args.batch_size, args.lang)
    logging.info("Initializing The Model")
    is_eng = True if args.lang == ENG else False
    model = initialize_model(args.model, args.tagger, tag_system, args.model_path, is_eng)
    train_set_size = len(train_dataloader)
    optimizer, scheduler, num_training_steps = initialize_optimizer_and_scheduler(model,
                                                                                  train_set_size,
                                                                                  args.lr,
                                                                                  args.epochs,
                                                                                  args.num_warmup_steps,
                                                                                  args.weight_decay)
    run_name = args.lang + "-" + args.tagger + "-" + args.model + "-" + str(
        args.lr) + "-" + str(args.epochs)
    writer = None
    if args.use_tensorboard:
        writer = SummaryWriter(comment=run_name)

    num_leaf_labels, num_tags = calc_num_tags_per_task(args.tagger, tag_system)
    model = torch.nn.DataParallel(model)
    model.to(device)
    logging.info("Starting The Training Loop")
    model.train()
    n_iter = 0

    when_to_eval = int(len(train_dataset) / (4 * args.batch_size))
    # eval_loss = 1000
    last_fscore = 0
    best_fscore = 0
    tol = 5

    for _ in tq(range(args.epochs)):
        t = 0
        for batch in tq(train_dataloader):
            optimizer.zero_grad()
            model.train()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.mean().backward()
            if args.use_tensorboard:
                writer.add_scalar('Loss/train', torch.mean(loss), n_iter)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if t % when_to_eval == 0:
                predictions, eval_labels = predict(model, eval_dataloader, len(eval_dataset),
                                                   num_tags, args.batch_size,
                                                   device)
                dev_metrics = calc_parse_eval(predictions, eval_labels, eval_dataset,
                                              tag_system, None,
                                              "", args.max_depth, args.keep_per_depth, False)
                eval_loss = report_eval_loss(model, eval_dataloader, device, n_iter, writer)

                writer.add_scalar('Fscore/dev', dev_metrics.fscore, n_iter)
                writer.add_scalar('Precision/dev', dev_metrics.precision, n_iter)
                writer.add_scalar('Recall/dev', dev_metrics.recall, n_iter)
                writer.add_scalar('loss/dev', eval_loss, n_iter)

                logging.info("current fscore {}".format(dev_metrics.fscore))
                logging.info("last fscore {}".format(last_fscore))
                logging.info("best fscore {}".format(best_fscore))
                if dev_metrics.fscore > last_fscore:  #if dev_metrics.fscore > last_fscore or dev_loss < last...
                    tol = 5
                    logging.info("tol refill")
                    if dev_metrics.fscore > best_fscore:  #if dev_metrics.fscore > best_fscore:
                        logging.info("save the best model")
                        best_fscore = dev_metrics.fscore
                        _save_best_model(model, args.output_path, run_name)
                elif dev_metrics.fscore > 0: #dev_metrics.fscore
                    tol -= 1
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 2.

                if tol < 0:
                    _finish_training(model, tag_system, eval_dataloader,
                                     eval_dataset, eval_loss, run_name, writer, args)
                    return
                if dev_metrics.fscore > 0:  # not propagating the nan
                    last_fscore = dev_metrics.fscore

            n_iter += 1
            t += 1

    _finish_training(model, tag_system, eval_dataloader, eval_dataset, eval_loss,
                     run_name, writer, args)


def _save_best_model(model, output_path, run_name):
    logging.info("Saving The Newly Found Best Model")
    torch.save(model.state_dict(), output_path + run_name)


def _finish_training(model, tag_system, eval_dataloader, eval_dataset, eval_loss,
                     run_name, writer, args):
    num_leaf_labels, num_tags = calc_num_tags_per_task(args.tagger, tag_system)
    predictions, eval_labels = predict(model, eval_dataloader, len(eval_dataset),
                                       num_tags, args.batch_size,
                                       device)
    even_acc, odd_acc = calc_tag_accuracy(predictions, eval_labels, num_leaf_labels, writer,
                                          args.use_tensorboard)
    register_run_metrics(writer, run_name, args.lr, args.epochs, eval_loss, even_acc, odd_acc)


def decode_model_name(model_name):
    name_chunks = model_name.split("-")
    name_chunks = name_chunks[1:]
    if name_chunks[0] == "td" or name_chunks[0] == "bu":
        tagging_schema = name_chunks[0] + "-" + name_chunks[1]
        model_type = name_chunks[2]
    else:
        tagging_schema = name_chunks[0]
        model_type = name_chunks[1]
    return tagging_schema, model_type


def calc_num_tags_per_task(tagging_schema, tag_system):
    if tagging_schema == TETRATAGGER:
        num_leaf_labels = tag_system.decode_moderator.leaf_tag_vocab_size
        num_tags = len(tag_system.tag_vocab)
    else:
        num_leaf_labels = len(tag_system.tag_vocab)
        num_tags = 2 * len(tag_system.tag_vocab)
    return num_leaf_labels, num_tags


def evaluate(args):
    reader = BracketParseCorpusReader(DATA_PATH, [args.lang + '.train', args.lang + '.dev',
                                                  args.lang + '.test'])
    tagging_schema, model_type = decode_model_name(args.model_name)
    writer = SummaryWriter(comment=args.model_name)
    logging.info("Initializing Tag System")
    tag_system = initialize_tag_system(reader, tagging_schema, args.lang,
                                       tag_vocab_path=args.tag_vocab_path,
                                       add_remove_top=args.lang == ENG)
    logging.info("Preparing Data")
    eval_dataset, eval_dataloader = prepare_test_data(reader,
                                                      tag_system, tagging_schema,
                                                      args.bert_model_path,
                                                      args.batch_size,
                                                      args.lang)

    is_eng = True if args.lang == ENG else False
    model = initialize_model(model_type, tagging_schema, tag_system, args.bert_model_path,
                             is_eng)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path + args.model_name))
    model.to(device)

    num_leaf_labels, num_tags = calc_num_tags_per_task(tagging_schema, tag_system)

    predictions, eval_labels = predict(model, eval_dataloader, len(eval_dataset),
                                       num_tags, args.batch_size, device)
    calc_tag_accuracy(predictions, eval_labels, num_leaf_labels, writer, args.use_tensorboard)
    parse_metrics = calc_parse_eval(predictions, eval_labels, eval_dataset, tag_system,
                                    args.output_path,
                                    args.model_name,
                                    args.max_depth,
                                    args.keep_per_depth,
                                    args.is_greedy)  # TODO: missing CRF transition matrix
    print(parse_metrics)


def main():
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'vocab':
        save_vocab(args)


if __name__ == '__main__':
    main()

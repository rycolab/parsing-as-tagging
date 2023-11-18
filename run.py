import os
import argparse
import logging
import pickle
import random
import sys
import json

import numpy as np
import torch
import transformers
from bitsandbytes.optim import AdamW
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tq

from const import *
from learning.dataset import TaggingDataset
from learning.evaluate import predict, dependency_eval, calc_parse_eval, calc_tag_accuracy, dependency_decoding
from learning.learn import ModelForTetratagging
from tagging.hexatagger import HexaTagger

# Set random seed
RANDOM_SEED = 1
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')
train = subparser.add_parser('train')
evaluate = subparser.add_parser('evaluate')
predict_parser = subparser.add_parser('predict')
vocab = subparser.add_parser('vocab')

vocab.add_argument('--tagger', choices=[HEXATAGGER, TETRATAGGER, TD_SR, BU_SR], required=True,
                   help="Tagging schema")
vocab.add_argument('--lang', choices=LANG, default=ENG, help="Language")
vocab.add_argument('--output-path', choices=[HEXATAGGER, TETRATAGGER, TD_SR, BU_SR],
                   default="data/vocab/")

train.add_argument('--tagger', choices=[HEXATAGGER, TETRATAGGER, TD_SR, BU_SR], required=True,
                   help="Tagging schema")
train.add_argument('--lang', choices=LANG, default=ENG, help="Language")
train.add_argument('--tag-vocab-path', type=str, default="data/vocab/")
train.add_argument('--model', choices=BERT, required=True, help="Model architecture")

train.add_argument('--model-path', type=str, default='bertlarge',
                   help="Bert model path or name, "
                        "xlnet-large-cased for english, hfl/chinese-xlnet-mid for chinese")
train.add_argument('--output-path', type=str, default='pat-models/',
                   help="Path to save trained models")
train.add_argument('--use-tensorboard', type=bool, default=False,
                   help="Whether to use the tensorboard for logging the results make sure to "
                        "add credentials to run.py if set to true")

train.add_argument('--max-depth', type=int, default=10,
                   help="Max stack depth used for decoding")
train.add_argument('--keep-per-depth', type=int, default=1,
                   help="Max elements to keep per depth")

train.add_argument('--lr', type=float, default=2e-5)
train.add_argument('--epochs', type=int, default=50)
train.add_argument('--batch-size', type=int, default=32)
train.add_argument('--num-warmup-steps', type=int, default=200)
train.add_argument('--weight-decay', type=float, default=0.01)

evaluate.add_argument('--model-name', type=str, required=True)
evaluate.add_argument('--lang', choices=LANG, default=ENG, help="Language")
evaluate.add_argument('--tagger', choices=[HEXATAGGER, TETRATAGGER, TD_SR, BU_SR],
                      required=True,
                      help="Tagging schema")
evaluate.add_argument('--tag-vocab-path', type=str, default="data/vocab/")
evaluate.add_argument('--model-path', type=str, default='pat-models/')
evaluate.add_argument('--bert-model-path', type=str, default='mbert/')
evaluate.add_argument('--output-path', type=str, default='results/')
evaluate.add_argument('--batch-size', type=int, default=16)
evaluate.add_argument('--max-depth', type=int, default=10,
                      help="Max stack depth used for decoding")
evaluate.add_argument('--is-greedy', type=bool, default=False,
                      help="Whether or not to use greedy decoding")
evaluate.add_argument('--keep-per-depth', type=int, default=1,
                      help="Max elements to keep per depth")
evaluate.add_argument('--use-tensorboard', type=bool, default=False,
                      help="Whether to use the tensorboard for logging the results make sure "
                           "to add credentials to run.py if set to true")


predict_parser.add_argument('--model-name', type=str, required=True)
predict_parser.add_argument('--lang', choices=LANG, default=ENG, help="Language")
predict_parser.add_argument('--tagger', choices=[HEXATAGGER, TETRATAGGER, TD_SR, BU_SR],
                      required=True,
                      help="Tagging schema")
predict_parser.add_argument('--tag-vocab-path', type=str, default="data/vocab/")
predict_parser.add_argument('--model-path', type=str, default='pat-models/')
predict_parser.add_argument('--bert-model-path', type=str, default='mbert/')
predict_parser.add_argument('--output-path', type=str, default='results/')
predict_parser.add_argument('--batch-size', type=int, default=16)
predict_parser.add_argument('--max-depth', type=int, default=10,
                      help="Max stack depth used for decoding")
predict_parser.add_argument('--is-greedy', type=bool, default=False,
                      help="Whether or not to use greedy decoding")
predict_parser.add_argument('--keep-per-depth', type=int, default=1,
                      help="Max elements to keep per depth")
predict_parser.add_argument('--use-tensorboard', type=bool, default=False,
                      help="Whether to use the tensorboard for logging the results make sure "
                           "to add credentials to run.py if set to true")


def initialize_tag_system(reader, tagging_schema, lang, tag_vocab_path="",
                          add_remove_top=False):
    tag_vocab = None
    if tag_vocab_path != "":
        with open(tag_vocab_path + lang + "-" + tagging_schema + '.pkl', 'rb') as f:
            tag_vocab = pickle.load(f)

    if tagging_schema == HEXATAGGER:
        # for BHT
        tag_system = HexaTagger(
            trees=reader.parsed_sents(lang + '.bht.train'),
            tag_vocab=tag_vocab, add_remove_top=False
        )
    else:
        logging.error("Please specify the tagging schema")
        return
    return tag_system


def get_data_path(tagger):
    if tagger == HEXATAGGER:
        return DEP_DATA_PATH
    return DATA_PATH


def save_vocab(args):
    data_path = get_data_path(args.tagger)
    if args.tagger == HEXATAGGER:
        prefix = args.lang + ".bht"
    else:
        prefix = args.lang
    reader = BracketParseCorpusReader(
        data_path, [prefix + '.train', prefix + '.dev', prefix + '.test'])
    tag_system = initialize_tag_system(
        reader, args.tagger, args.lang, add_remove_top=True)
    with open(args.output_path + args.lang + "-" + args.tagger + '.pkl', 'wb+') as f:
        pickle.dump(tag_system.tag_vocab, f)


def prepare_training_data(reader, tag_system, tagging_schema, model_name, batch_size, lang):
    is_tetratags = True if tagging_schema == TETRATAGGER or tagging_schema == HEXATAGGER else False
    prefix = lang + ".bht" if tagging_schema == HEXATAGGER else lang

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, truncation=True, use_fast=True)
    train_dataset = TaggingDataset(prefix + '.train', tokenizer, tag_system, reader, device,
                                   is_tetratags=is_tetratags, language=lang)
    eval_dataset = TaggingDataset(prefix + '.test', tokenizer, tag_system, reader, device,
                                  is_tetratags=is_tetratags, language=lang)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=train_dataset.collate,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate, pin_memory=True
    )
    return train_dataset, eval_dataset, train_dataloader, eval_dataloader


def prepare_test_data(reader, tag_system, tagging_schema, model_name, batch_size, lang):
    is_tetratags = True if tagging_schema == TETRATAGGER or tagging_schema == HEXATAGGER else False
    prefix = lang + ".bht" if tagging_schema == HEXATAGGER else lang

    print(f"Evaluating {model_name}, {tagging_schema}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, truncation=True, use_fast=True)
    test_dataset = TaggingDataset(
        prefix + '.test', tokenizer, tag_system, reader, device,
        is_tetratags=is_tetratags, language=lang
    )
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
    elif model_type in BERT and tagging_schema in [TETRATAGGER, HEXATAGGER]:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            num_labels=len(tag_system.tag_vocab),
            id2label={i: label for i, label in enumerate(tag_system.tag_vocab)},
            label2id={label: i for i, label in enumerate(tag_system.tag_vocab)},
            task_specific_params={
                'model_path': model_path,
                'num_even_tags': tag_system.decode_moderator.leaf_tag_vocab_size,
                'num_odd_tags': tag_system.decode_moderator.internal_tag_vocab_size,
                'pos_emb_dim': 256,
                'num_pos_tags': 50,
                'lstm_layers': 3,
                'dropout': 0.33,
                'is_eng': is_eng,
                'use_pos': True
            }
        )
    elif model_type in BERT and tagging_schema != TETRATAGGER and tagging_schema != HEXATAGGER:
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
    config = generate_config(
        model_type, tagging_schema, tag_system, model_path, is_eng
    )
    if model_type in BERT:
        model = ModelForTetratagging(config=config)
    else:
        logging.error("Invalid model type")
        return
    return model


def initialize_optimizer_and_scheduler(model, dataset_size, lr=5e-5, num_epochs=4,
                                       num_warmup_steps=160, weight_decay_rate=0.0):
    num_training_steps = num_epochs * dataset_size
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "bert" not in n],
            "weight_decay": 0.0,
            "lr": lr * 50, "betas": (0.9, 0.9),
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "bert" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr, "betas": (0.9, 0.999),
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "bert" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
            "lr": lr, "betas": (0.9, 0.999),
        },
    ]

    optimizer = AdamW(
        grouped_parameters, lr=lr
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
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


def train_command(args):
    if args.tagger == HEXATAGGER:
        prefix = args.lang + ".bht"
    else:
        prefix = args.lang
    data_path = get_data_path(args.tagger)
    reader = BracketParseCorpusReader(data_path, [prefix + '.train', prefix + '.dev',
                                                  prefix + '.test'])
    logging.info("Initializing Tag System")
    tag_system = initialize_tag_system(
        reader, args.tagger, args.lang,
        tag_vocab_path=args.tag_vocab_path, add_remove_top=True
    )
    logging.info("Preparing Data")
    train_dataset, eval_dataset, train_dataloader, eval_dataloader = prepare_training_data(
        reader, tag_system, args.tagger, args.model_path, args.batch_size, args.lang)
    logging.info("Initializing The Model")
    is_eng = True if args.lang == ENG else False
    model = initialize_model(
        args.model, args.tagger, tag_system, args.model_path, is_eng
    )
    model.to(device)

    train_set_size = len(train_dataloader)
    optimizer, scheduler, num_training_steps = initialize_optimizer_and_scheduler(
        model, train_set_size, args.lr, args.epochs,
        args.num_warmup_steps, args.weight_decay
    )
    optimizer.zero_grad()
    run_name = args.lang + "-" + args.tagger + "-" + args.model + "-" + str(
        args.lr) + "-" + str(args.epochs)
    writer = None
    if args.use_tensorboard:
        writer = SummaryWriter(comment=run_name)

    num_leaf_labels, num_tags = calc_num_tags_per_task(args.tagger, tag_system)

    logging.info("Starting The Training Loop")
    model.train()
    n_iter = 0

    last_fscore = 0
    best_fscore = 0
    tol = 99999

    for epo in tq(range(args.epochs)):
        logging.info(f"*******************EPOCH {epo}*******************")
        t = 1
        model.train()

        with tq(train_dataloader, disable=False) as progbar:
            for batch in progbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                if device == "cuda":
                    with torch.cuda.amp.autocast(
                            enabled=True, dtype=torch.bfloat16
                    ):
                        outputs = model(**batch)
                else:
                    with torch.cpu.amp.autocast(
                            enabled=True, dtype=torch.bfloat16
                    ):
                        outputs = model(**batch)

                loss = outputs[0]
                loss.mean().backward()
                if args.use_tensorboard:
                    writer.add_scalar('Loss/train', torch.mean(loss), n_iter)
                progbar.set_postfix(loss=torch.mean(loss).item())

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                n_iter += 1
                t += 1

        if True:  # evaluation at the end of epoch
            predictions, eval_labels = predict(
                model, eval_dataloader, len(eval_dataset),
                num_tags, args.batch_size, device
            )
            calc_tag_accuracy(
                predictions, eval_labels,
                num_leaf_labels, writer, args.use_tensorboard)

            if args.tagger == HEXATAGGER:
                dev_metrics_las, dev_metrics_uas = dependency_eval(
                    predictions, eval_labels, eval_dataset,
                    tag_system, None, "", args.max_depth,
                    args.keep_per_depth, False)
            else:
                dev_metrics = calc_parse_eval(
                    predictions, eval_labels, eval_dataset,
                    tag_system, None, "",
                    args.max_depth, args.keep_per_depth, False)

            eval_loss = 0.5
            if args.tagger == HEXATAGGER:
                writer.add_scalar('LAS_Fscore/dev',
                                  dev_metrics_las.fscore, n_iter)
                writer.add_scalar('LAS_Precision/dev',
                                  dev_metrics_las.precision, n_iter)
                writer.add_scalar('LAS_Recall/dev',
                                  dev_metrics_las.recall, n_iter)
                writer.add_scalar('loss/dev', eval_loss, n_iter)

                logging.info("current LAS {}".format(dev_metrics_las))
                logging.info("current UAS {}".format(dev_metrics_uas))
                logging.info("last LAS fscore {}".format(last_fscore))
                logging.info("best LAS fscore {}".format(best_fscore))
                # setting main metric for model selection
                dev_metrics = dev_metrics_las
            else:
                writer.add_scalar('Fscore/dev', dev_metrics.fscore, n_iter)
                writer.add_scalar('Precision/dev', dev_metrics.precision, n_iter)
                writer.add_scalar('Recall/dev', dev_metrics.recall, n_iter)
                writer.add_scalar('loss/dev', eval_loss, n_iter)

                logging.info("current fscore {}".format(dev_metrics.fscore))
                logging.info("last fscore {}".format(last_fscore))
                logging.info("best fscore {}".format(best_fscore))

                # if dev_metrics.fscore > last_fscore or dev_loss < last...
                if dev_metrics.fscore > last_fscore:
                    tol = 5
                    if dev_metrics.fscore > best_fscore:  # if dev_metrics.fscore > best_fscore:
                        logging.info("save the best model")
                        best_fscore = dev_metrics.fscore
                        _save_best_model(model, args.output_path, run_name)
                elif dev_metrics.fscore > 0:  # dev_metrics.fscore
                    tol -= 1

                if tol < 0:
                    _finish_training(model, tag_system, eval_dataloader,
                                     eval_dataset, eval_loss, run_name, writer, args)
                    return
                if dev_metrics.fscore > 0:  # not propagating the nan
                    last_eval_loss = eval_loss
                    last_fscore = dev_metrics.fscore

            # if dev_metrics.fscore > last_fscore or dev_loss < last...
            if dev_metrics.fscore > best_fscore:
                tol = 99999
                logging.info("tol refill")
                logging.info("save the best model")
                best_eval_loss = eval_loss
                best_fscore = dev_metrics.fscore
                _save_best_model(model, args.output_path, run_name)
            elif eval_loss > 0:
                tol -= 1

            if tol < 0:
                _finish_training(model, tag_system, eval_dataloader,
                                 eval_dataset, eval_loss, run_name, writer, args)
                return
            if eval_loss > 0:  # not propagating the nan
                last_eval_loss = eval_loss
            # end of epoch
            pass

    _finish_training(model, tag_system, eval_dataloader, eval_dataset, eval_loss,
                     run_name, writer, args)


def _save_best_model(model, output_path, run_name):
    logging.info("Saving The Newly Found Best Model")
    os.makedirs(output_path, exist_ok=True)
    to_save_file = os.path.join(output_path, run_name)
    torch.save(model.state_dict(), to_save_file)


def _finish_training(model, tag_system, eval_dataloader, eval_dataset, eval_loss,
                     run_name, writer, args):
    num_leaf_labels, num_tags = calc_num_tags_per_task(args.tagger, tag_system)
    predictions, eval_labels = predict(model, eval_dataloader, len(eval_dataset),
                                       num_tags, args.batch_size,
                                       device)
    even_acc, odd_acc = calc_tag_accuracy(predictions, eval_labels, num_leaf_labels, writer,
                                          args.use_tensorboard)
    register_run_metrics(writer, run_name, args.lr,
                         args.epochs, eval_loss, even_acc, odd_acc)


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
    if tagging_schema == TETRATAGGER or tagging_schema == HEXATAGGER:
        num_leaf_labels = tag_system.decode_moderator.leaf_tag_vocab_size
        num_tags = len(tag_system.tag_vocab)
    else:
        num_leaf_labels = len(tag_system.tag_vocab)
        num_tags = 2 * len(tag_system.tag_vocab)
    return num_leaf_labels, num_tags


def evaluate_command(args):
    tagging_schema, model_type = decode_model_name(args.model_name)
    data_path = get_data_path(tagging_schema)  # HexaTagger or others
    print("Evaluation Args", args)
    if args.tagger == HEXATAGGER:
        prefix = args.lang + ".bht"
    else:
        prefix = args.lang
    reader = BracketParseCorpusReader(
        data_path,
        [prefix + '.train', prefix + '.dev', prefix + '.test'])
    writer = SummaryWriter(comment=args.model_name)
    logging.info("Initializing Tag System")
    tag_system = initialize_tag_system(reader, tagging_schema, args.lang,
                                       tag_vocab_path=args.tag_vocab_path,
                                       add_remove_top=True)
    logging.info("Preparing Data")
    eval_dataset, eval_dataloader = prepare_test_data(
        reader, tag_system, tagging_schema,
        args.bert_model_path, args.batch_size,
        args.lang)

    is_eng = True if args.lang == ENG else False
    model = initialize_model(
        model_type, tagging_schema, tag_system, args.bert_model_path, is_eng
    )
    model.load_state_dict(torch.load(args.model_path + args.model_name))
    model.to(device)

    num_leaf_labels, num_tags = calc_num_tags_per_task(tagging_schema, tag_system)

    predictions, eval_labels = predict(
        model, eval_dataloader, len(eval_dataset),
        num_tags, args.batch_size, device)
    calc_tag_accuracy(predictions, eval_labels,
                      num_leaf_labels, writer, args.use_tensorboard)
    if tagging_schema == HEXATAGGER:
        dev_metrics_las, dev_metrics_uas = dependency_eval(
            predictions, eval_labels, eval_dataset,
            tag_system, args.output_path, args.model_name,
            args.max_depth, args.keep_per_depth, False)
        print(
            "LAS: ", dev_metrics_las, "\n",
            "UAS: ", dev_metrics_uas, sep=""
        )
    else:
        parse_metrics = calc_parse_eval(predictions, eval_labels, eval_dataset, tag_system,
                                        args.output_path,
                                        args.model_name,
                                        args.max_depth,
                                        args.keep_per_depth,
                                        args.is_greedy)
        print(parse_metrics)


def predict_command(args):
    tagging_schema, model_type = decode_model_name(args.model_name)
    data_path = get_data_path(tagging_schema)  # HexaTagger or others
    print("predict Args", args)

    if args.tagger == HEXATAGGER:
        prefix = args.lang + ".bht"
    else:
        prefix = args.lang
    reader = BracketParseCorpusReader(data_path, [])
    writer = SummaryWriter(comment=args.model_name)
    logging.info("Initializing Tag System")
    tag_system = initialize_tag_system(None, tagging_schema, args.lang,
                                       tag_vocab_path=args.tag_vocab_path,
                                       add_remove_top=True)
    logging.info("Preparing Data")
    eval_dataset, eval_dataloader = prepare_test_data(
        reader, tag_system, tagging_schema,
        args.bert_model_path, args.batch_size,
        "input")

    is_eng = True if args.lang == ENG else False
    model = initialize_model(
        model_type, tagging_schema, tag_system, args.bert_model_path, is_eng
    )
    model.load_state_dict(torch.load(args.model_path + args.model_name))
    model.to(device)

    num_leaf_labels, num_tags = calc_num_tags_per_task(tagging_schema, tag_system)

    predictions, eval_labels = predict(
        model, eval_dataloader, len(eval_dataset),
        num_tags, args.batch_size, device)
    
    if tagging_schema == HEXATAGGER:
        pred_output = dependency_decoding(
            predictions, eval_labels, eval_dataset,
            tag_system, args.output_path, args.model_name,
            args.max_depth, args.keep_per_depth, False)

        with open(args.output_path + args.model_name + ".pred.json", "w") as fout:
            print("Saving predictions to", args.output_path + args.model_name + ".pred.json")
            json.dump(pred_output, fout)

        """pred_output contains: {
            "predicted_dev_triples": predicted_dev_triples,
            "predicted_hexa_tags": pred_hexa_tags
        }"""




def main():
    args = parser.parse_args()
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'vocab':
        save_vocab(args)


if __name__ == '__main__':
    main()

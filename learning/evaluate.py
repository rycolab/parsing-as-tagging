import logging
import math
import os.path
import re
import subprocess
import tempfile
from copy import deepcopy
from typing import Tuple
import tempfile

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm as tq

from tagging.tree_tools import create_dummy_tree

repo_directory = os.path.abspath(__file__)

class ParseMetrics(object):
    def __init__(self, recall, precision, fscore, complete_match, tagging_accuracy=100):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore
        self.complete_match = complete_match
        self.tagging_accuracy = tagging_accuracy

    def __str__(self):
        if self.tagging_accuracy < 100:
            return "(Recall={:.4f}, Precision={:.4f}, ParseMetrics={:.4f}, CompleteMatch={:.4f}, TaggingAccuracy={:.4f})".format(
                self.recall, self.precision, self.fscore, self.complete_match,
                self.tagging_accuracy)
        else:
            return "(Recall={:.4f}, Precision={:.4f}, ParseMetrics={:.4f}, CompleteMatch={:.4f})".format(
                self.recall, self.precision, self.fscore, self.complete_match)


def report_eval_loss(model, eval_dataloader, device, n_iter, writer) -> np.ndarray:
    loss = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss.append(torch.mean(outputs[0]).cpu())

    mean_loss = np.mean(loss)
    logging.info("Eval Loss: {}".format(mean_loss))
    if writer is not None:
        writer.add_scalar('eval_loss', mean_loss, n_iter)
    return mean_loss


def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device
) -> Tuple[np.array, np.array]:
    model.eval()
    predictions = []
    eval_labels = []
    max_len = 0
    idx = 0

    for batch in tq(eval_dataloader, disable=True):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=True, dtype=torch.bfloat16
        ):
            outputs = model(**batch)

        logits = outputs[1].float().cpu().numpy()
        max_len = max(max_len, logits.shape[1])
        predictions.append(logits)
        labels = batch['labels'].int().cpu().numpy()
        eval_labels.append(labels)
        idx += 1

    predictions = np.concatenate([np.pad(logits,
                                         ((0, 0), (0, max_len - logits.shape[1]), (0, 0)),
                                         'constant', constant_values=0) for logits in
                                  predictions], axis=0)
    eval_labels = np.concatenate([np.pad(labels, ((0, 0), (0, max_len - labels.shape[1])),
                                         'constant', constant_values=0) for labels in
                                  eval_labels], axis=0)

    return predictions, eval_labels


def calc_tag_accuracy(
        predictions, eval_labels, num_leaf_labels, writer, use_tensorboard
) -> Tuple[float, float]:
    even_predictions = predictions[..., -num_leaf_labels:]
    odd_predictions = predictions[..., :-num_leaf_labels]
    even_labels = eval_labels % (num_leaf_labels + 1) - 1
    odd_labels = eval_labels // (num_leaf_labels + 1) - 1

    odd_predictions = odd_predictions[odd_labels != -1].argmax(-1)
    even_predictions = even_predictions[even_labels != -1].argmax(-1)

    odd_labels = odd_labels[odd_labels != -1]
    even_labels = even_labels[even_labels != -1]

    odd_acc = (odd_predictions == odd_labels).mean()
    even_acc = (even_predictions == even_labels).mean()

    logging.info('odd_tags_accuracy: {}'.format(odd_acc))
    logging.info('even_tags_accuracy: {}'.format(even_acc))

    if use_tensorboard:
        writer.add_pr_curve('odd_tags_pr_curve', odd_labels, odd_predictions, 0)
        writer.add_pr_curve('even_tags_pr_curve', even_labels, even_predictions, 1)
    return even_acc, odd_acc


def get_dependency_from_lexicalized_tree(lex_tree, triple_dict, offset=0):
    # this recursion assumes projectivity
    # Input:
    #     root of lex-tree
    # Output:
    #     the global index of the dependency root
    if type(lex_tree) not in {str, dict} and len(lex_tree) == 1:
        # unary rule
        # returning the global index of the head
        return offset

    head_branch_index = int(lex_tree.label().split("^^^")[1])
    head_global_index = None
    branch_to_global_dict = {}

    for branch_id_child, child in enumerate(lex_tree):
        global_id_child = get_dependency_from_lexicalized_tree(
            child, triple_dict, offset=offset
        )
        offset = offset + len(child.leaves())
        branch_to_global_dict[branch_id_child] = global_id_child
        if branch_id_child == head_branch_index:
            head_global_index = global_id_child

    for branch_id_child, child in enumerate(lex_tree):
        if branch_id_child != head_branch_index:
            triple_dict[branch_to_global_dict[branch_id_child]] = head_global_index

    return head_global_index


def is_punctuation(pos):
    punct_set = '.' '``' "''" ':' ','
    return (pos in punct_set) or (pos.lower() in ['pu', 'punct'])  # for CTB & UD


def tree_to_dep_triples(lex_tree):
    triple_dict = {}
    dep_triples = []
    sent_root = get_dependency_from_lexicalized_tree(
        lex_tree, triple_dict
    )
    # the root of the whole sentence should refer to ROOT
    assert sent_root not in triple_dict
    # the root of the sentence
    triple_dict[sent_root] = -1
    for head, tail in sorted(triple_dict.items()):
        dep_triples.append((
            head, tail,
            lex_tree.pos()[head][1].split("^^^")[1].split("+")[0],
            lex_tree.pos()[head][1].split("^^^")[1].split("+")[1]
        ))
    return dep_triples



def dependency_eval(
        predictions, eval_labels, eval_dataset, tag_system, output_path,
        model_name, max_depth, keep_per_depth, is_greedy
) -> ParseMetrics:
    ud_flag = eval_dataset.language not in {'English', 'Chinese'}

    # This can be parallelized!
    predicted_dev_triples, predicted_dev_triples_unlabeled = [], []
    gold_dev_triples, gold_dev_triples_unlabeled = [], []
    c_err = 0

    gt_triple_data, pred_triple_data = [], []

    for i in tq(range(predictions.shape[0]), disable=True):
        logits = predictions[i]
        is_word = (eval_labels[i] != 0)

        original_tree = deepcopy(eval_dataset.trees[i])
        original_tree.collapse_unary(collapsePOS=True, collapseRoot=True)

        try:  # ignore the ones that failed in unchomsky_normal_form
            tree = tag_system.logits_to_tree(
                logits, original_tree.pos(),
                mask=is_word,
                max_depth=max_depth,
                keep_per_depth=keep_per_depth,
                is_greedy=is_greedy
            )
            tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            c_err += 1
            predicted_dev_triples.append(create_dummy_tree(original_tree.pos()))
            continue
        if tree.leaves() != original_tree.leaves():
            c_err += 1
            predicted_dev_triples.append(create_dummy_tree(original_tree.pos()))
            continue

        gt_triples = tree_to_dep_triples(original_tree)
        pred_triples = tree_to_dep_triples(tree)

        gt_triple_data.append(gt_triples)
        pred_triple_data.append(pred_triples)

        assert len(gt_triples) == len(
            pred_triples), f"wrong length {len(gt_triples)} vs. {len(pred_triples)}!"

        for x, y in zip(sorted(gt_triples), sorted(pred_triples)):
            if is_punctuation(x[3]) and not ud_flag:
                # ignoring punctuations for evaluation
                continue
            assert x[0] == y[0], f"wrong tree {gt_triples} vs. {pred_triples}!"

            gold_dev_triples.append(f"{x[0]}-{x[1]}-{x[2].split(':')[0]}")
            gold_dev_triples_unlabeled.append(f"{x[0]}-{x[1]}")

            predicted_dev_triples.append(f"{y[0]}-{y[1]}-{y[2].split(':')[0]}")
            predicted_dev_triples_unlabeled.append(f"{y[0]}-{y[1]}")

    if ud_flag:
        # UD
        predicted_dev_triples, predicted_dev_triples_unlabeled = [], []
        gold_dev_triples, gold_dev_triples_unlabeled = [], []

        language, split = eval_dataset.language, eval_dataset.split.split(".")[-1]

        gold_temp_out, pred_temp_out = tempfile.mktemp(dir=os.path.dirname(repo_directory)), \
                                       tempfile.mktemp(dir=os.path.dirname(repo_directory))
        gold_temp_in, pred_temp_in = gold_temp_out + ".deproj", pred_temp_out + ".deproj"


        save_triplets(gt_triple_data, gold_temp_out)
        save_triplets(pred_triple_data, pred_temp_out)

        for filename, tgt_filename in zip([gold_temp_out, pred_temp_out], [gold_temp_in, pred_temp_in]):
            command = f"cd ./malt/maltparser-1.9.2/; java -jar maltparser-1.9.2.jar -c {language}_{split} -m deproj" \
                    f" -i {filename} -o {tgt_filename} ; cd ../../"
            os.system(command)

        loaded_gold_dev_triples = load_triplets(gold_temp_in)
        loaded_pred_dev_triples = load_triplets(pred_temp_in)

        for gt_triples, pred_triples in zip(loaded_gold_dev_triples, loaded_pred_dev_triples):
            for x, y in zip(sorted(gt_triples), sorted(pred_triples)):
                if is_punctuation(x[3]):
                    # ignoring punctuations for evaluation
                    continue
                assert x[0] == y[0], f"wrong tree {gt_triples} vs. {pred_triples}!"

                gold_dev_triples.append(f"{x[0]}-{x[1]}-{x[2].split(':')[0]}")
                gold_dev_triples_unlabeled.append(f"{x[0]}-{x[1]}")

                predicted_dev_triples.append(f"{y[0]}-{y[1]}-{y[2].split(':')[0]}")
                predicted_dev_triples_unlabeled.append(f"{y[0]}-{y[1]}")
        

    logging.warning("Number of binarization error: {}\n".format(c_err))
    las_recall, las_precision, las_fscore, _ = precision_recall_fscore_support(
        gold_dev_triples, predicted_dev_triples, average='micro'
    )
    uas_recall, uas_precision, uas_fscore, _ = precision_recall_fscore_support(
        gold_dev_triples_unlabeled, predicted_dev_triples_unlabeled, average='micro'
    )

    return (ParseMetrics(las_recall, las_precision, las_fscore, complete_match=1),
            ParseMetrics(uas_recall, uas_precision, uas_fscore, complete_match=1))


def save_triplets(triplet_data, file_path):
    # save triplets to file in conll format
    with open(file_path, 'w') as f:
        for triplets in triplet_data:
            for triplet in triplets:
                # 8	Витоша	витоша	PROPN	Npfsi	Definite=Ind|Gender=Fem|Number=Sing	6	nmod	_	_
                head, tail, label, pos = triplet
                f.write(f"{head+1}\t-\t-\t{pos}\t-\t-\t{tail+1}\t{label}\t_\t_\n")
            f.write('\n')

    return


def load_triplets(file_path):
    # load triplets from file in conll format
    triplet_data = []
    with open(file_path, 'r') as f:
        triplets = []
        for line in f.readlines():
            if line.startswith('#') or line == '\n':
                if triplets:
                    triplet_data.append(triplets)
                triplets = []
                continue
            line_list = line.strip().split('\t')
            head, tail, label, pos = line_list[0], line_list[6], line_list[7], line_list[3]
            triplets.append((head, tail, label, pos))
        if triplets:
            triplet_data.append(triplets)
    return triplet_data


def calc_parse_eval(predictions, eval_labels, eval_dataset, tag_system, output_path,
                    model_name, max_depth, keep_per_depth, is_greedy) -> ParseMetrics:
    predicted_dev_trees = []
    gold_dev_trees = []
    c_err = 0
    for i in tq(range(predictions.shape[0])):
        logits = predictions[i]
        is_word = eval_labels[i] != 0
        original_tree = eval_dataset.trees[i]
        gold_dev_trees.append(original_tree)
        try:  # ignore the ones that failed in unchomsky_normal_form
            tree = tag_system.logits_to_tree(logits, original_tree.pos(), mask=is_word,
                                             max_depth=max_depth,
                                             keep_per_depth=keep_per_depth,
                                             is_greedy=is_greedy)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            # print(message)
            c_err += 1
            predicted_dev_trees.append(create_dummy_tree(original_tree.pos()))
            continue
        if tree.leaves() != original_tree.leaves():
            c_err += 1
            predicted_dev_trees.append(create_dummy_tree(original_tree.pos()))
            continue
        predicted_dev_trees.append(tree)

    logging.warning("Number of binarization error: {}".format(c_err))

    return evalb("EVALB_SPMRL/", gold_dev_trees, predicted_dev_trees)


def save_predictions(predicted_trees, file_path):
    with open(file_path, 'w') as f:
        for tree in predicted_trees:
            f.write(' '.join(str(tree).split()) + '\n')


def evalb(evalb_dir, gold_trees, predicted_trees, ref_gold_path=None) -> ParseMetrics:
    # Code from: https://github.com/nikitakit/self-attentive-parser/blob/master/src/evaluate.py
    assert os.path.exists(evalb_dir)
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_spmrl_program_path = os.path.join(evalb_dir, "evalb_spmrl")
    assert os.path.exists(evalb_program_path) or os.path.exists(evalb_spmrl_program_path)

    if os.path.exists(evalb_program_path):
        evalb_param_path = os.path.join(evalb_dir, "nk.prm")
    else:
        evalb_program_path = evalb_spmrl_program_path
        evalb_param_path = os.path.join(evalb_dir, "spmrl.prm")

    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        gold_leaves = list(gold_tree.leaves())
        predicted_leaves = list(predicted_tree.leaves())

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")

    with open(gold_path, "w") as outfile:
        if ref_gold_path is None:
            for tree in gold_trees:
                outfile.write(' '.join(str(tree).split()) + '\n')
        else:
            # For the SPMRL dataset our data loader performs some modifications
            # (like stripping morphological features), so we compare to the
            # raw gold file to be certain that we haven't spoiled the evaluation
            # in some way.
            with open(ref_gold_path) as goldfile:
                outfile.write(goldfile.read())

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write(' '.join(str(tree).split()) + '\n')

    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        gold_path,
        predicted_path,
        output_path,
    )
    subprocess.run(command, shell=True)

    fscore = ParseMetrics(math.nan, math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
            match = re.match(r"Complete match\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.complete_match = float(match.group(1))
            match = re.match(r"Tagging accuracy\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.tagging_accuracy = float(match.group(1))
                break

    success = (
            not math.isnan(fscore.fscore) or
            fscore.recall == 0.0 or
            fscore.precision == 0.0)

    if success:
        temp_dir.cleanup()
    else:
        print("Error reading EVALB results.")
        print("Gold path: {}".format(gold_path))
        print("Predicted path: {}".format(predicted_path))
        print("Output path: {}".format(output_path))

    return fscore




def dependency_decoding(
        predictions, eval_labels, eval_dataset, tag_system, output_path,
        model_name, max_depth, keep_per_depth, is_greedy
) -> ParseMetrics:
    ud_flag = eval_dataset.language not in {'English', 'Chinese'}

    # This can be parallelized!
    predicted_dev_triples, predicted_dev_triples_unlabeled = [], []
    gold_dev_triples, gold_dev_triples_unlabeled = [], []
    pred_hexa_tags = []
    c_err = 0

    gt_triple_data, pred_triple_data = [], []

    for i in tq(range(predictions.shape[0]), disable=True):
        logits = predictions[i]
        is_word = (eval_labels[i] != 0)

        original_tree = deepcopy(eval_dataset.trees[i])
        original_tree.collapse_unary(collapsePOS=True, collapseRoot=True)

        try:  # ignore the ones that failed in unchomsky_normal_form
            tree = tag_system.logits_to_tree(
                logits, original_tree.pos(),
                mask=is_word,
                max_depth=max_depth,
                keep_per_depth=keep_per_depth,
                is_greedy=is_greedy
            )
            hexa_ids = tag_system.logits_to_ids(
                logits, is_word, max_depth, keep_per_depth, is_greedy=is_greedy
            )
            pred_hexa_tags.append(hexa_ids)

            tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            c_err += 1
            predicted_dev_triples.append(create_dummy_tree(original_tree.pos()))
            continue
        if tree.leaves() != original_tree.leaves():
            c_err += 1
            predicted_dev_triples.append(create_dummy_tree(original_tree.pos()))
            continue

        gt_triples = tree_to_dep_triples(original_tree)
        pred_triples = tree_to_dep_triples(tree)

        gt_triple_data.append(gt_triples)
        pred_triple_data.append(pred_triples)

        assert len(gt_triples) == len(
            pred_triples), f"wrong length {len(gt_triples)} vs. {len(pred_triples)}!"

        for x, y in zip(sorted(gt_triples), sorted(pred_triples)):
            if is_punctuation(x[3]) and not ud_flag:
                # ignoring punctuations for evaluation
                continue
            assert x[0] == y[0], f"wrong tree {gt_triples} vs. {pred_triples}!"

            gold_dev_triples.append(f"{x[0]}-{x[1]}-{x[2].split(':')[0]}")
            gold_dev_triples_unlabeled.append(f"{x[0]}-{x[1]}")

            predicted_dev_triples.append(f"{y[0]}-{y[1]}-{y[2].split(':')[0]}")
            predicted_dev_triples_unlabeled.append(f"{y[0]}-{y[1]}")

    if ud_flag:
        # UD
        predicted_dev_triples, predicted_dev_triples_unlabeled = [], []
        gold_dev_triples, gold_dev_triples_unlabeled = [], []

        language, split = eval_dataset.language, eval_dataset.split.split(".")[-1]

        gold_temp_out, pred_temp_out = tempfile.mktemp(dir=os.path.dirname(repo_directory)), \
                                       tempfile.mktemp(dir=os.path.dirname(repo_directory))
        gold_temp_in, pred_temp_in = gold_temp_out + ".deproj", pred_temp_out + ".deproj"


        save_triplets(gt_triple_data, gold_temp_out)
        save_triplets(pred_triple_data, pred_temp_out)

        for filename, tgt_filename in zip([gold_temp_out, pred_temp_out], [gold_temp_in, pred_temp_in]):
            command = f"cd ./malt/maltparser-1.9.2/; java -jar maltparser-1.9.2.jar -c {language}_{split} -m deproj" \
                    f" -i {filename} -o {tgt_filename} ; cd ../../"
            os.system(command)

        loaded_gold_dev_triples = load_triplets(gold_temp_in)
        loaded_pred_dev_triples = load_triplets(pred_temp_in)

        for gt_triples, pred_triples in zip(loaded_gold_dev_triples, loaded_pred_dev_triples):
            for x, y in zip(sorted(gt_triples), sorted(pred_triples)):
                if is_punctuation(x[3]):
                    # ignoring punctuations for evaluation
                    continue
                assert x[0] == y[0], f"wrong tree {gt_triples} vs. {pred_triples}!"

                gold_dev_triples.append(f"{x[0]}-{x[1]}-{x[2].split(':')[0]}")
                gold_dev_triples_unlabeled.append(f"{x[0]}-{x[1]}")

                predicted_dev_triples.append(f"{y[0]}-{y[1]}-{y[2].split(':')[0]}")
                predicted_dev_triples_unlabeled.append(f"{y[0]}-{y[1]}")
        
    return {
        "predicted_dev_triples": predicted_dev_triples,
        "predicted_hexa_tags": pred_hexa_tags
    }
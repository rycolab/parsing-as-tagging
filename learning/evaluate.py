import logging
import math
import os.path
import re
import subprocess
import tempfile

import numpy as np
import torch
from tqdm import tqdm as tq

from tagging.tree_tools import create_dummy_tree


class ParseMetrics(object):
    # Code from: https://github.com/mrdrozdov/self-attentive-parser-with-extra-features/blob/master/src/evaluate.py
    def __init__(self, recall, precision, fscore, complete_match, tagging_accuracy=100):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore
        self.complete_match = complete_match
        self.tagging_accuracy = tagging_accuracy

    def __str__(self):
        if self.tagging_accuracy < 100:
            return "(Recall={:.2f}, Precision={:.2f}, ParseMetrics={:.2f}, CompleteMatch={:.2f}, TaggingAccuracy={:.2f})".format(
                self.recall, self.precision, self.fscore, self.complete_match,
                self.tagging_accuracy)
        else:
            return "(Recall={:.2f}, Precision={:.2f}, ParseMetrics={:.2f}, CompleteMatch={:.2f})".format(
                self.recall, self.precision, self.fscore, self.complete_match)


def report_eval_loss(model, eval_dataloader, device, n_iter, writer) -> np.ndarray:
    loss = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss.append(torch.mean(outputs[0]).cpu())

    mean_loss = np.mean(loss)
    logging.info("Eval Loss: {}".format(mean_loss))
    if writer is not None:
        writer.add_scalar('eval_loss', mean_loss, n_iter)
    return mean_loss


def predict(model, eval_dataloader, dataset_size, num_tags, batch_size, device) -> ([], []):
    model.eval()
    predictions = np.zeros((dataset_size, 256, num_tags))
    eval_labels = np.zeros((dataset_size, 256), dtype=int)
    idx = 0
    for batch in tq(eval_dataloader):
        if idx * batch_size >= dataset_size:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs[1]
        predictions[idx * batch_size:(idx + 1) * batch_size, :, :] = logits.cpu().numpy()
        labels = batch['labels']
        eval_labels[idx * batch_size:(idx + 1) * batch_size, :] = labels.cpu().numpy()
        idx += 1

    return predictions, eval_labels


def calc_tag_accuracy(predictions, eval_labels, num_leaf_labels, writer, use_tensorboard) -> (
float, float):
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
    return even_acc, odd_acc


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
                                             max_depth=max_depth, keep_per_depth=keep_per_depth, is_greedy=is_greedy)
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
    # save_predictions(predicted_dev_trees, output_path + model_name + "_predictions.txt")
    # save_predictions(gold_dev_trees, output_path + model_name + "_gold.txt")
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

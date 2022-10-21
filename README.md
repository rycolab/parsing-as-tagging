# An Analysis of Parsing as Tagging
<p align="center">
  <img src="https://github.com/rycolab/parsing-tools/blob/main/header.jpg" width=400>
</p>
This repository contains code for training and evaluation of parsing as tagging methods for constituency parsing.

## Setting Up The Environment
Set up a virtual environment and install the dependencies:
```bash
pip install -r requirements.txt
```

## Getting The Data
Follow the instructions in this [repo](https://github.com/nikitakit/self-attentive-parser/tree/master/data) to do the initial preprocessing on English WSJ and SPMRL datasets. The default data path is `data/spmrl` folder, where each file titled in `[LANGUAGE].[train/dev/test]` format.


## Building The Tagging Vocab
In order to use taggers, we need to build the vocabulary of tags for in-order, pre-order and post-order linearizations. You can cache these vocabularies using:
```bash
python run.py vocab --lang [LANGUAGE] --tagger [TAGGER]
```
Tagger can be `td-sr` for top-down (pre-order) shift--reduce linearization, `bu-sr` for bottom-up (post-order) shift--reduce linearization,
or `tetra` for in-order linearization.

## Training
Train the model and store the best checkpoint.
```bash
python run.py train --batch-size [BATCH_SIZE]  --tagger [TAGGER] --lang [LANGUAGE] --model [MODEL] --epochs [EPOCHS] --lr [LR] --model-path [MODEL_PATH] --output-path [PATH] --max-depth [DEPTH] --keep-per-depth [KPD] [--use-tensorboard]
```
- batch size: use 32 to reproduce the results
- tagger: `td-sr` or `bu-sr` or `tetra`
- lang: language, one of the nine languages reported in the paper
- model: `bert`, `bert+crf`, `bert+lstm`
- model path: path that pretrained model is saved
- output path: path to save the best trained model
- max depth: maximum depth to keep in the decoding lattice
- keep per depth: number of elements to keep track of in the decoding step
- use-tensorboard: whether to store the logs in tensorboard or not (true or false)

## Evaluation
Calculate evaluation metrics: fscore, precision, recall, loss.
```bash
python run.py evaluate --lang [LANGUAGE] --model-name [MODEL]  --model-path [MODEL_PATH] --bert-model-path [BERT_PATH] --max-depth [DEPTH] --keep-per-depth [KPD]  [--is-greedy]
```
- lang: language, one of the nine languages reported in the paper
- model name: name of the checkpoint
- model path: path of the checkpoint
- bert model path: path to the pretrained model
- max depth: maximum depth to keep in the decoding lattice
- keep per depth: number of elements to keep track of in the decoding step
- is greedy: whether or not use the greedy decoding, default is false


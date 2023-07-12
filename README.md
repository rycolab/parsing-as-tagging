# An Analysis of Parsing as Tagging
<p align="center">
  <img src="https://github.com/rycolab/parsing-as-tagging/blob/main/header.jpg" width=400>
  <img src="https://github.com/rycolab/parsing-as-tagging/blob/main/header.jpg" width=400>
</p>
This repository contains code for training and evaluation of two papers:

- On Parsing as Tagging 
- Hexatagging: Projective Dependency Parsing as Tagging

## Setting Up The Environment
Set up a virtual environment and install the dependencies:
```bash
pip install -r requirements.txt
```

## Getting The Data
### Constituency Parsing
Follow the instructions in this [repo](https://github.com/nikitakit/self-attentive-parser/tree/master/data) to do the initial preprocessing on English WSJ and SPMRL datasets. The default data path is `data/spmrl` folder, where each file titled in `[LANGUAGE].[train/dev/test]` format.
### Dependency Parsing with Hexatagger
1. Convert CoNLL to Binary Headed Trees:
```bash
python data/dep2bht.py
```
This will generate the phrase-structured BHT trees in the `data/bht` directory. 
We placed the processed files already under the `data/bht` directory.

## Building The Tagging Vocab
In order to use taggers, we need to build the vocabulary of tags for in-order, pre-order and post-order linearizations. You can cache these vocabularies using:
```bash
python run.py vocab --lang [LANGUAGE] --tagger [TAGGER]
```
Tagger can be `td-sr` for top-down (pre-order) shift--reduce linearization, `bu-sr` for bottom-up (post-order) shift--reduce linearization,`tetra` for in-order, and `hexa` for hexatagging linearization.

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

# Exact Commands for Hexatagging
The above commands can be used together with different taggers, models, and on different languages. To reproduce our Hexatagging results, here we put the exact commands used for training and evaluation of Hexatagger. 
## Train
### PTB (English)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path xlnet-large-cased --output-path ./checkpoints/ --use-tensorboard True
```
### CTB (Chinese)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang Chinese --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path hfl/chinese-xlnet-mid --output-path ./checkpoints/ --use-tensorboard True
```

### UD
```bash
python run.py evaluate --lang bg --max-depth 10 --tagger hexa --bert-model-path bert-base-multilingual-cased --model-name bg-hexa-bert-1e-05-50 --batch-size 32 --model-path ./checkpoints/                                    
```
## Evaluate
### PTB
```bash
python run.py evaluate --lang English --max-depth 10 --tagger hexa --bert-model-path xlnet-large-cased --model-name English-hexa-bert-3e-05-50 --batch-size 64 --model-path ./
```

### CTB
```bash
python run.py evaluate --lang Chinese --max-depth 10 --tagger hexa --bert-model-path bert-base-chinese --model-name Chinese-hexa-bert-3e-05-50 --batch-size 64 --model-path ./checkpoints/
```
### UD
```bash
python run.py evaluate --lang bg --max-depth 10 --tagger hexa --bert-model-path bert-base-multilingual-cased --model-name bg-hexa-bert-1e-05-50 --batch-size 64 --model-path ./checkpoints/
```

# Citation
If you find this repository useful, please cite our papers:
```bibtex
@inproceedings{amini-cotterell-2022-parsing,
    title = "On Parsing as Tagging",
    author = "Amini, Afra  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.607",
    pages = "8884--8900",
}
```

```bibtext
@inproceedings{amini-etal-2023-hexatagging,
    title = "Hexatagging: Projective Dependency Parsing as Tagging",
    author = "Amini, Afra  and
      Liu, Tianyu  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.124",
    pages = "1453--1464",
}
```


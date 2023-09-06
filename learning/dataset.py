import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}


def ptb_unescape(sent):
    cleaned_words = []
    for word in sent:
        word = BERT_TOKEN_MAPPING.get(word, word)
        word = word.replace('\\/', '/').replace('\\*', '*')
        # Mid-token punctuation occurs in biomedical text
        word = word.replace('-LSB-', '[').replace('-RSB-', ']')
        word = word.replace('-LRB-', '(').replace('-RRB-', ')')
        if word == "n't" and cleaned_words:
            cleaned_words[-1] = cleaned_words[-1] + "n"
            word = "'t"
        cleaned_words.append(word)
    return cleaned_words


class TaggingDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, tag_system, reader, device, is_tetratags=False,
                 language="english", pad_to_len=None,
                 max_train_len=350):
        self.reader = reader
        self.split = split
        self.trees = self.reader.parsed_sents(split)
        self.tokenizer = tokenizer
        self.language = language
        self.tag_system = tag_system
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_to_len = pad_to_len
        self.device = device
        self.is_tetratags = is_tetratags

        if "train" in split and max_train_len is not None:
            # To speed up training, we only train on short sentences.
            print(len(self.trees), "trees before filtering")
            self.trees = [
                tree for tree in self.trees if
                (len(tree.leaves()) <= max_train_len and len(tree.leaves()) >= 2)]
            print(len(self.trees), "trees after filtering")
        else:
            # speed up!
            self.trees = [
                tree for tree in self.trees if len(tree.leaves()) <= max_train_len]

        if not os.path.exists(
                f"./data/pos.{language.lower()}.json") and "train" in split:
            self.pos_dict = self.get_pos_dict()
            with open(f"./data/pos.{language.lower()}.json", 'w') as fp:
                json.dump(self.pos_dict, fp)
        else:
            with open(f"./data/pos.{language.lower()}.json", 'r') as fp:
                self.pos_dict = json.load(fp)

    def get_pos_dict(self):
        pos_dict = {}
        for t in self.trees:
            for _, x in t.pos():
                pos_dict[x] = pos_dict.get(x, 1 + len(pos_dict))
        return pos_dict

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree = self.trees[index]
        words = ptb_unescape(tree.leaves())
        pos_tags = [self.pos_dict.get(x[1], 0) for x in tree.pos()]

        if 'albert' in str(type(self.tokenizer)):
            # albert is case insensitive!
            words = [w.lower() for w in words]

        encoded = self.tokenizer.encode_plus(' '.join(words))
        word_end_positions = [
            encoded.char_to_token(i)
            for i in np.cumsum([len(word) + 1 for word in words]) - 2]
        word_start_positions = [
            encoded.char_to_token(i)
            for i in np.cumsum([0] + [len(word) + 1 for word in words])[:-1]]

        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        pair_ids = torch.zeros_like(input_ids)
        end_of_word = torch.zeros_like(input_ids)
        pos_ids = torch.zeros_like(input_ids)

        tag_ids = self.tag_system.tree_to_ids_pipeline(tree)

        # Pack both leaf and internal tag ids into a single "label" field.
        # (The huggingface API isn't flexible enough to use multiple label fields)
        tag_ids = [tag_id + 1 for tag_id in tag_ids] + [0]
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)
        labels = torch.zeros_like(input_ids)

        odd_labels = tag_ids[1::2]
        if self.is_tetratags:
            even_labels = tag_ids[
                          ::2] - self.tag_system.decode_moderator.internal_tag_vocab_size
            labels[word_end_positions] = (
                    odd_labels * (
                        self.tag_system.decode_moderator.leaf_tag_vocab_size + 1) + even_labels)
            pos_ids[word_end_positions] = torch.tensor(pos_tags, dtype=torch.long)
            pair_ids[word_end_positions] = torch.as_tensor(word_start_positions)

            end_of_word[word_end_positions] = 1
            end_of_word[word_end_positions[-1]] = 2  # last word
        else:
            even_labels = tag_ids[::2]
            labels[word_end_positions] = (
                    odd_labels * (len(self.tag_system.tag_vocab) + 1) + even_labels)

        if self.pad_to_len is not None:
            pad_amount = self.pad_to_len - input_ids.shape[0]
            if pad_amount >= 0:
                input_ids = F.pad(input_ids, [0, pad_amount], value=self.pad_token_id)
                pos_ids = F.pad(pos_ids, [0, pad_amount], value=0)
                labels = F.pad(labels, [0, pad_amount], value=0)

        return {
            'input_ids': input_ids,
            'pos_ids': pos_ids,
            'pair_ids': pair_ids,
            'end_of_word': end_of_word,
            'labels': labels
        }

    def collate(self, batch):
        # for GPT-2, self.pad_token_id is None
        pad_token_id = self.pad_token_id if self.pad_token_id is not None else -100
        input_ids = pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True, padding_value=pad_token_id)

        attention_mask = (input_ids != pad_token_id).float()
        # for GPT-2, change -100 back into 0
        input_ids = torch.where(
            input_ids == -100,
            0,
            input_ids
        )

        pair_ids = pad_sequence(
            [item['pair_ids'] for item in batch],
            batch_first=True, padding_value=pad_token_id)
        end_of_word = pad_sequence(
            [item['end_of_word'] for item in batch],
            batch_first=True, padding_value=0)
        pos_ids = pad_sequence(
            [item['pos_ids'] for item in batch],
            batch_first=True, padding_value=0)

        labels = pad_sequence(
            [item['labels'] for item in batch],
            batch_first=True, padding_value=0)

        return {
            'input_ids': input_ids,
            'pos_ids': pos_ids,
            'pair_ids': pair_ids,
            'end_of_word': end_of_word,
            'attention_mask': attention_mask,
            'labels': labels,
        }

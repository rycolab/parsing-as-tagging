import logging
from abc import ABC

from nltk import ParentedTree as PTree
from tqdm import tqdm as tq

from learning.decode import BeamSearch, GreedySearch
from tagging.tagger import Tagger, TagDecodeModerator
from tagging.transform import LeftCornerTransformer

import numpy as np

from tagging.tree_tools import find_node_type, NodeType


class SRTagDecodeModerator(TagDecodeModerator, ABC):
    def __init__(self, tag_vocab):
        super().__init__(tag_vocab)
        self.reduce_tag_size = len([tag for tag in tag_vocab if tag.startswith("r")])
        self.shift_tag_size = len([tag for tag in tag_vocab if tag.startswith("s")])

        self.rr_tag_size = len([tag for tag in tag_vocab if tag.startswith("rr")])
        self.sr_tag_size = len([tag for tag in tag_vocab if tag.startswith("sr")])

        self.rl_tag_size = self.reduce_tag_size - self.rr_tag_size  # left reduce tag size
        self.sl_tag_size = self.shift_tag_size - self.sr_tag_size  # left shift tag size

        self.mask_binarize = True

    def mask_scores_for_binarization(self, labels, scores) -> []:
        raise NotImplementedError


class BUSRTagDecodeModerator(SRTagDecodeModerator):
    def __init__(self, tag_vocab):
        super().__init__(tag_vocab)
        stack_depth_change_by_id = [None] * len(tag_vocab)
        for i, tag in enumerate(tag_vocab):
            if tag.startswith("s"):
                stack_depth_change_by_id[i] = +1
            elif tag.startswith("r"):
                stack_depth_change_by_id[i] = -1
        assert None not in stack_depth_change_by_id
        self.stack_depth_change_by_id = np.array(
            stack_depth_change_by_id, dtype=int)

        self.reduce_only_mask = np.full((len(tag_vocab),), -np.inf)
        self.shift_only_mask = np.full((len(tag_vocab),), -np.inf)
        self.reduce_only_mask[:self.reduce_tag_size] = 0.0
        self.shift_only_mask[-self.shift_tag_size:] = 0.0

    def mask_scores_for_binarization(self, labels, scores) -> []:
        # after rr(right) -> only reduce, after rl(left) -> only shift
        # after sr(right) -> only reduce, after sl(left) -> only shift
        mask1 = np.where(
            (labels[:, None] >= self.rl_tag_size) & (labels[:, None] < self.reduce_tag_size),
            self.reduce_only_mask, 0.0)
        mask2 = np.where(labels[:, None] < self.rl_tag_size, self.shift_only_mask, 0.0)
        mask3 = np.where(
            labels[:, None] >= (self.sl_tag_size + self.reduce_tag_size),
            self.reduce_only_mask, 0.0)
        mask4 = np.where((labels[:, None] >= self.reduce_tag_size) & (
                labels[:, None] < (self.sl_tag_size + self.reduce_tag_size)),
                         self.shift_only_mask, 0.0)
        all_new_scores = scores + mask1 + mask2 + mask3 + mask4
        return all_new_scores


class TDSRTagDecodeModerator(SRTagDecodeModerator):
    def __init__(self, tag_vocab):
        super().__init__(tag_vocab)
        is_shift_mask = np.concatenate(
            [
                np.zeros(self.reduce_tag_size),
                np.ones(self.shift_tag_size),
            ]
        )
        self.reduce_tags_only = np.asarray(-1e9 * is_shift_mask, dtype=float)

        stack_depth_change_by_id = [None] * len(tag_vocab)
        stack_depth_change_by_id_l2 = [None] * len(tag_vocab)
        for i, tag in enumerate(tag_vocab):
            if tag.startswith("s"):
                stack_depth_change_by_id_l2[i] = 0
                stack_depth_change_by_id[i] = -1
            elif tag.startswith("r"):
                stack_depth_change_by_id_l2[i] = -1
                stack_depth_change_by_id[i] = +2
        assert None not in stack_depth_change_by_id
        assert None not in stack_depth_change_by_id_l2
        self.stack_depth_change_by_id = np.array(
            stack_depth_change_by_id, dtype=int)
        self.stack_depth_change_by_id_l2 = np.array(
            stack_depth_change_by_id_l2, dtype=int)
        self._initialize_binarize_mask(tag_vocab)

    def _initialize_binarize_mask(self, tag_vocab) -> None:
        self.right_only_mask = np.full((len(tag_vocab),), -np.inf)
        self.left_only_mask = np.full((len(tag_vocab),), -np.inf)

        self.right_only_mask[self.rl_tag_size:self.reduce_tag_size] = 0.0
        self.right_only_mask[-self.sr_tag_size:] = 0.0

        self.left_only_mask[:self.rl_tag_size] = 0.0
        self.left_only_mask[
        self.reduce_tag_size:self.reduce_tag_size + self.sl_tag_size] = 0.0

    def mask_scores_for_binarization(self, labels, scores) -> []:
        # if shift -> rr and sr, if reduce -> rl and sl
        mask1 = np.where(labels[:, None] >= self.reduce_tag_size, self.right_only_mask, 0.0)
        mask2 = np.where(labels[:, None] < self.reduce_tag_size, self.left_only_mask, 0.0)
        all_new_scores = scores + mask1 + mask2
        return all_new_scores


class SRTagger(Tagger, ABC):
    def __init__(self, trees=None, tag_vocab=None, add_remove_top=False):
        super().__init__(trees, tag_vocab, add_remove_top)

    def add_trees_to_vocab(self, trees: []) -> None:
        self.label_vocab = set()
        for tree in tq(trees):
            for tag in self.tree_to_tags_pipeline(tree)[0]:
                self.tag_vocab.add(tag)
                idx = tag.find("/")
                if idx != -1:
                    self.label_vocab.add(tag[idx + 1:])
                else:
                    self.label_vocab.add("")
        self.tag_vocab = sorted(self.tag_vocab)
        self.label_vocab = sorted(self.label_vocab)

    @staticmethod
    def create_shift_tag(label: str, is_right_child=False) -> str:
        suffix = "r" if is_right_child else ""
        if label.find("+") != -1:
            tag = "s" + suffix + "/" + "/".join(label.split("+")[:-1])
        else:
            tag = "s" + suffix
        return tag

    @staticmethod
    def create_shift_label(tag: str) -> str:
        idx = tag.find("/")
        if idx != -1:
            return tag[idx + 1:].replace("/", "+") + "+"
        else:
            return ""

    @staticmethod
    def create_reduce_tag(label: str, is_right_child=False) -> str:
        if label.find("|") != -1:  # drop extra node labels created after binarization
            tag = "r"
        else:
            tag = "r" + "/" + label.replace("+", "/")
        return "r" + tag if is_right_child else tag

    @staticmethod
    def _create_reduce_label(tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            label = "|"  # to mark the second part as an extra node created via binarizaiton
        else:
            label = tag[idx + 1:].replace("/", "+")
        return label


class SRTaggerBottomUp(SRTagger):
    def __init__(self, trees=None, tag_vocab=None, add_remove_top=False):
        super().__init__(trees, tag_vocab, add_remove_top)
        self.decode_moderator = BUSRTagDecodeModerator(self.tag_vocab)

    def tree_to_tags(self, root: PTree) -> ([str], int):
        tags = []
        lc = LeftCornerTransformer.extract_left_corner_no_eps(root)
        if len(root) == 1:  # edge case
            tags.append(self.create_shift_tag(lc.label(), False))
            return tags, 1

        is_right_child = lc.left_sibling() is not None
        tags.append(self.create_shift_tag(lc.label(), is_right_child))

        logging.debug("SHIFT {}".format(lc.label()))
        stack = [lc]
        max_stack_len = 1

        while len(stack) > 0:
            node = stack[-1]
            max_stack_len = max(max_stack_len, len(stack))

            if node.left_sibling() is None and node.right_sibling() is not None:
                lc = LeftCornerTransformer.extract_left_corner_no_eps(node.right_sibling())
                stack.append(lc)
                logging.debug("SHIFT {}".format(lc.label()))
                is_right_child = lc.left_sibling() is not None
                tags.append(self.create_shift_tag(lc.label(), is_right_child))

            elif len(stack) >= 2 and (
                    node.right_sibling() == stack[-2] or node.left_sibling() == stack[-2]):
                prev_node = stack[-2]
                logging.debug("REDUCE[ {0} {1} --> {2} ]".format(
                    *(prev_node.label(), node.label(), node.parent().label())))

                parent_is_right = node.parent().left_sibling() is not None
                tags.append(self.create_reduce_tag(node.parent().label(), parent_is_right))
                stack.pop()
                stack.pop()
                stack.append(node.parent())

            elif stack[0].parent() is None and len(stack) == 1:
                stack.pop()
                continue
        return tags, max_stack_len

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        created_node_stack = []
        node = None

        if len(tags) == 1:  # base case
            assert tags[0].startswith('s')
            prefix = self.create_shift_label(tags[0])
            return PTree(prefix + input_seq[0][1], [input_seq[0][0]])
        for tag in tags:
            if tag.startswith('s'):
                prefix = self.create_shift_label(tag)
                created_node_stack.append(PTree(prefix + input_seq[0][1], [input_seq[0][0]]))
                input_seq.pop(0)
            else:
                last_node = created_node_stack.pop()
                last_2_node = created_node_stack.pop()
                node = PTree(self._create_reduce_label(tag), [last_2_node, last_node])
                created_node_stack.append(node)

        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return node

    def logits_to_ids(self, logits: [], mask, max_depth, keep_per_depth, crf_transitions=None,
                      is_greedy=False) -> [int]:
        if is_greedy:
            searcher = GreedySearch(
                self.decode_moderator,
                initial_stack_depth=0,
                crf_transitions=crf_transitions,
                max_depth=max_depth,
                keep_per_depth=keep_per_depth,
            )
        else:

            searcher = BeamSearch(
                self.decode_moderator,
                initial_stack_depth=0,
                crf_transitions=crf_transitions,
                max_depth=max_depth,
                keep_per_depth=keep_per_depth,
            )

        last_t = None
        seq_len = sum(mask)
        idx = 1
        for t in range(logits.shape[0]):
            if mask is not None and not mask[t]:
                continue
            if last_t is not None:
                searcher.advance(
                    logits[last_t, :-len(self.tag_vocab)]
                )
            if idx == seq_len:
                searcher.advance(logits[t, -len(self.tag_vocab):], is_last=True)
            else:
                searcher.advance(logits[t, -len(self.tag_vocab):])
            last_t = t

        score, best_tag_ids = searcher.get_path()
        return best_tag_ids


class SRTaggerTopDown(SRTagger):
    def __init__(self, trees=None, tag_vocab=None, add_remove_top=False):
        super().__init__(trees, tag_vocab, add_remove_top)
        self.decode_moderator = TDSRTagDecodeModerator(self.tag_vocab)

    def tree_to_tags(self, root: PTree) -> ([str], int):
        stack: [PTree] = [root]
        max_stack_len = 1
        tags = []

        while len(stack) > 0:
            node = stack[-1]
            max_stack_len = max(max_stack_len, len(stack))

            if find_node_type(node) == NodeType.NT:
                stack.pop()
                logging.debug("REDUCE[ {0} --> {1} {2}]".format(
                    *(node.label(), node[0].label(), node[1].label())))
                is_right_node = node.left_sibling() is not None
                tags.append(self.create_reduce_tag(node.label(), is_right_node))
                stack.append(node[1])
                stack.append(node[0])

            else:
                logging.debug("-->\tSHIFT[ {0} ]".format(node.label()))
                is_right_node = node.left_sibling() is not None
                tags.append(self.create_shift_tag(node.label(), is_right_node))
                stack.pop()

        return tags, max_stack_len

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        if len(tags) == 1:  # base case
            assert tags[0].startswith('s')
            prefix = self.create_shift_label(tags[0])
            return PTree(prefix + input_seq[0][1], [input_seq[0][0]])

        assert tags[0].startswith('r')
        node = PTree(self._create_reduce_label(tags[0]), [])
        created_node_stack: [PTree] = [node]

        for tag in tags[1:]:
            parent: PTree = created_node_stack[-1]
            if tag.startswith('s'):
                prefix = self.create_shift_label(tag)
                new_node = PTree(prefix + input_seq[0][1], [input_seq[0][0]])
                input_seq.pop(0)
            else:
                label = self._create_reduce_label(tag)
                new_node = PTree(label, [])

            if len(parent) == 0:
                parent.insert(0, new_node)
            elif len(parent) == 1:
                parent.insert(1, new_node)
                created_node_stack.pop()

            if tag.startswith('r'):
                created_node_stack.append(new_node)

        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return node

    def logits_to_ids(self, logits: [], mask, max_depth, keep_per_depth, crf_transitions=None,
                      is_greedy=False) -> [int]:
        if is_greedy:
            searcher = GreedySearch(
                self.decode_moderator,
                initial_stack_depth=1,
                crf_transitions=crf_transitions,
                max_depth=max_depth,
                min_depth=0,
                keep_per_depth=keep_per_depth,
            )
        else:
            searcher = BeamSearch(
                self.decode_moderator,
                initial_stack_depth=1,
                crf_transitions=crf_transitions,
                max_depth=max_depth,
                min_depth=0,
                keep_per_depth=keep_per_depth,
            )

        last_t = None
        seq_len = sum(mask)
        idx = 1
        is_last = False
        for t in range(logits.shape[0]):
            if mask is not None and not mask[t]:
                continue
            if last_t is not None:
                searcher.advance(
                    logits[last_t, :-len(self.tag_vocab)]
                )
            if idx == seq_len:
                is_last = True
            if last_t is None:
                searcher.advance(
                    logits[t, -len(self.tag_vocab):] + self.decode_moderator.reduce_tags_only,
                    is_last=is_last)
            else:
                searcher.advance(logits[t, -len(self.tag_vocab):], is_last=is_last)
            last_t = t

        score, best_tag_ids = searcher.get_path(required_stack_depth=0)
        return best_tag_ids

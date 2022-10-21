from abc import ABC

from nltk import ParentedTree as PTree
from nltk import Tree
from tqdm import tqdm as tq

from tagging.tree_tools import add_plus_to_tree, remove_plus_from_tree


class TagDecodeModerator(ABC):
    def __init__(self, tag_vocab):
        self.vocab_len = len(tag_vocab)
        self.stack_depth_change_by_id = None
        self.stack_depth_change_by_id_l2 = None


class Tagger(ABC):
    def __init__(self, trees=None, tag_vocab=None, add_remove_top=False):
        self.tag_vocab = set()
        self.add_remove_top = add_remove_top

        if tag_vocab is not None:
            self.tag_vocab = tag_vocab
        elif trees is not None:
            self.add_trees_to_vocab(trees)

        self.decode_moderator = None

    def add_trees_to_vocab(self, trees: []) -> None:
        for tree in tq(trees):
            tags = self.tree_to_tags_pipeline(tree)[0]
            for tag in tags:
                self.tag_vocab.add(tag)
        self.tag_vocab = sorted(self.tag_vocab)

    def tree_to_tags(self, root: PTree) -> [str]:
        raise NotImplementedError("tree to tags is not implemented")

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        raise NotImplementedError("tags to tree is not implemented")

    def tree_to_tags_pipeline(self, tree: Tree) -> ([str], int):
        ptree = self.preprocess(tree)
        return self.tree_to_tags(ptree)

    def tree_to_ids_pipeline(self, tree: Tree) -> [int]:
        tags = self.tree_to_tags_pipeline(tree)[0]
        res = []
        for tag in tags:
            if tag in self.tag_vocab:
                res.append(self.tag_vocab.index(tag))
            elif tag[0:tag.find("/")] in self.tag_vocab:
                res.append(self.tag_vocab.index(tag[0:tag.find("/")]))
            else:
                res.append(0)

        return res

    def tags_to_tree_pipeline(self, tags: [str], input_seq: []) -> Tree:
        filtered_input_seq = []
        for token, pos in input_seq:
            filtered_input_seq.append((token, pos.replace("+", "@")))
        ptree = self.tags_to_tree(tags, filtered_input_seq)
        return self.postprocess(ptree)

    def ids_to_tree_pipeline(self, ids: [int], input_seq: []) -> Tree:
        tags = [self.tag_vocab[idx] for idx in ids]
        return self.tags_to_tree_pipeline(tags, input_seq)

    def logits_to_ids(self, logits: [], mask, max_depth, keep_per_depth, is_greedy=False) -> [int]:
        raise NotImplementedError("logits to ids is not implemented")

    def logits_to_tree(self, logits: [], leave_nodes: [], mask=None, max_depth=5, keep_per_depth=1, is_greedy=False) -> Tree:
        ids = self.logits_to_ids(logits, mask, max_depth, keep_per_depth, is_greedy=is_greedy)
        return self.ids_to_tree_pipeline(ids, leave_nodes)

    def preprocess(self, original_tree: Tree) -> PTree:
        tree = original_tree.copy(deep=True)
        if self.add_remove_top:
            cut_off_tree = tree[0]
        else:
            cut_off_tree = tree

        remove_plus_from_tree(cut_off_tree)
        cut_off_tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        cut_off_tree.chomsky_normal_form()
        ptree = PTree.convert(cut_off_tree)
        return ptree

    def postprocess(self, tree: PTree) -> Tree:
        tree = Tree.convert(tree)
        tree.un_chomsky_normal_form()
        add_plus_to_tree(tree)
        if self.add_remove_top:
            return Tree("TOP", [tree])
        else:
            return tree

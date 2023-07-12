import random
import string
from enum import Enum

import numpy as np
from nltk import ParentedTree
from nltk import Tree

from const import DUMMY_LABEL


class NodeType(Enum):
    NT = 0
    NT_NT = 1
    PT = 2


def find_node_type(node: ParentedTree) -> NodeType:
    if len(node) == 1:
        return NodeType.PT
    elif node.label().find("\\") != -1:
        return NodeType.NT_NT
    else:
        return NodeType.NT


def is_node_epsilon(node: ParentedTree) -> bool:
    node_leaves = node.leaves()
    if len(node_leaves) == 1 and node_leaves[0] == "EPS":
        return True
    return False


def is_topo_equal(first: ParentedTree, second: ParentedTree) -> bool:
    if len(first) == 1 and len(second) != 1:
        return False
    if len(first) != 1 and len(second) == str:
        return False
    if len(first) == 1 and len(second) == 1:
        return True
    return is_topo_equal(first[0], second[0]) and is_topo_equal(first[1], second[1])


def random_tree(node: ParentedTree, depth=0, p=.75, cutoff=7) -> None:
    """ sample a random tree
    @param input_str: list of sampled terminals
    """
    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the left child tree
        left_label = "X/" + str(depth)
        left = ParentedTree(left_label, [])
        node.insert(0, left)
        random_tree(left, depth=depth + 1, p=p, cutoff=cutoff)
    else:
        label = "X/" + str(depth)
        left = ParentedTree(label, [random.choice(string.ascii_letters)])
        node.insert(0, left)

    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the right child tree
        right_label = "X/" + str(depth)
        right = ParentedTree(right_label, [])
        node.insert(1, right)
        random_tree(right, depth=depth + 1, p=p, cutoff=cutoff)
    else:
        label = "X/" + str(depth)
        right = ParentedTree(label, [random.choice(string.ascii_letters)])
        node.insert(1, right)


def create_dummy_tree(leaves):
    dummy_tree = Tree("S", [])
    idx = 0
    for token, pos in leaves:
        dummy_tree.insert(idx, Tree(pos, [token]))
        idx += 1

    return dummy_tree


def remove_plus_from_tree(tree):
    if type(tree) == str:
        return
    label = tree.label()
    new_label = label.replace("+", "@")
    tree.set_label(new_label)
    for child in tree:
        remove_plus_from_tree(child)


def add_plus_to_tree(tree):
    if type(tree) == str:
        return
    label = tree.label()
    new_label = label.replace("@", "+")
    tree.set_label(new_label)
    for child in tree:
        add_plus_to_tree(child)


def process_label(label, node_type):
    # label format is NT^^^HEAD_IDX
    suffix = label[label.rindex("^^^"):]
    head_idx = int(label[label.rindex("^^^") + 3:])
    return node_type + suffix, head_idx


def extract_head_idx(node):
    label = node.label()
    return int(label[label.rindex("^^^") + 3:])


def attach_to_tree(ref_node, parent_node, stack):
    if len(ref_node) == 1:
        new_node = Tree(ref_node.label(), [ref_node[0]])
        parent_node.insert(len(parent_node), new_node)
        return stack

    new_node = Tree(ref_node.label(), [])
    stack.append((ref_node, new_node))
    parent_node.insert(len(parent_node), new_node)
    return stack


def relabel_tree(root):
    stack = [root]
    while len(stack) != 0:
        cur_node = stack.pop()
        if type(cur_node) == str:
            continue
        cur_node.set_label(f"X^^^{extract_head_idx(cur_node)}")
        for child in cur_node:
            stack.append(child)


def debinarize_lex_tree(root, new_root):
    stack = [(root, new_root)]  # (node in binarized tree, node in debinarized tree)
    while len(stack) != 0:
        ref_node, new_node = stack.pop()
        head_idx = extract_head_idx(ref_node)

        # attach the left child
        stack = attach_to_tree(ref_node[0], new_node, stack)

        cur_node = ref_node[1]
        while cur_node.label().startswith(DUMMY_LABEL):
            right_idx = extract_head_idx(cur_node)
            head_idx += right_idx
            # attach the left child
            stack = attach_to_tree(cur_node[0], new_node, stack)
            cur_node = cur_node[1]

        # attach the right child
        stack = attach_to_tree(cur_node, new_node, stack)
        # update the label
        new_node.set_label(f"X^^^{head_idx}")


def binarize_lex_tree(children, node, node_type):
    # node type is X if normal node, Y|X (DUMMY_LABEL) if a dummy node created because of binarization

    if len(children) == 1:
        node.insert(0, children[0])
        return node
    new_label, head_idx = process_label(node.label(), node_type)
    node.set_label(new_label)
    if len(children) == 2:
        left_node = Tree(children[0].label(), [])
        left_node = binarize_lex_tree(children[0], left_node, "X")
        right_node = Tree(children[1].label(), [])
        right_node = binarize_lex_tree(children[1], right_node, "X")
        node.insert(0, left_node)
        node.insert(1, right_node)
        return node
    elif len(children) > 2:
        if head_idx > 1:
            node.set_label(f"{node_type}^^^1")
        if head_idx >= 1:
            head_idx -= 1
        left_node = Tree(children[0].label(), [])
        left_node = binarize_lex_tree(children[0], left_node, "X")
        right_node = Tree(f"{DUMMY_LABEL}^^^{head_idx}", [])
        right_node = binarize_lex_tree(children[1:], right_node, DUMMY_LABEL)
        node.insert(0, left_node)
        node.insert(1, right_node)
        return node
    else:
        raise ValueError("node has zero children!")


def expand_unary(tree):
    if len(tree) == 1:
        label = tree.label().split("+")
        pos_label = label[1]
        tree.set_label(label[0])
        tree[0] = Tree(pos_label, [tree[0]])
        return
    for child in tree:
        expand_unary(child)

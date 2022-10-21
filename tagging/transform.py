from nltk import ParentedTree as PTree
from tagging.tree_tools import find_node_type, is_node_epsilon, NodeType


class Transformer:
    @classmethod
    def expand_nt(cls, node: PTree, ref_node: PTree) -> (PTree, PTree, PTree, PTree):
        raise NotImplementedError("expand non-terminal is not implemented")

    @classmethod
    def expand_nt_nt(cls, node: PTree, ref_node1: PTree, ref_node2: PTree) -> (
            PTree, PTree, PTree, PTree):
        raise NotImplementedError("expand paired non-terimnal is not implemented")

    @classmethod
    def extract_right_corner(cls, node: PTree) -> PTree:
        while type(node[0]) != str:
            if len(node) > 1:
                node = node[1]
            else:  # unary rules
                node = node[0]
        return node

    @classmethod
    def extract_left_corner(cls, node: PTree) -> PTree:
        while len(node) > 1:
            node = node[0]
        return node

    @classmethod
    def transform(cls, node: PTree, ref_node1: PTree, ref_node2: PTree) -> None:
        if node is None:
            return
        type = find_node_type(node)
        if type == NodeType.NT:
            left_ref1, left_ref2, right_ref1, right_ref2, is_base_case = cls.expand_nt(node,
                                                                                       ref_node1)
        elif type == NodeType.NT_NT:
            is_base_case = False
            left_ref1, left_ref2, right_ref1, right_ref2 = cls.expand_nt_nt(
                node, ref_node1,
                ref_node2)
        else:
            return
        if is_base_case:
            return
        cls.transform(node[0], left_ref1, left_ref2)
        cls.transform(node[1], right_ref1, right_ref2)


class LeftCornerTransformer(Transformer):

    @classmethod
    def extract_left_corner_no_eps(cls, node: PTree) -> PTree:
        while len(node) > 1:
            if not is_node_epsilon(node[0]):
                node = node[0]
            else:
                node = node[1]
        return node

    @classmethod
    def expand_nt(cls, node: PTree, ref_node: PTree) -> (PTree, PTree, PTree, PTree, bool):
        leftcorner_node = cls.extract_left_corner(ref_node)
        if leftcorner_node == ref_node:  # this only happens if the tree only consists of one terminal rule
            node.insert(0, ref_node.leaves()[0])
            return None, None, None, None, True
        new_right_node = PTree(node.label() + "\\" + leftcorner_node.label(), [])
        new_left_node = PTree(leftcorner_node.label(), leftcorner_node.leaves())

        node.insert(0, new_left_node)
        node.insert(1, new_right_node)
        return leftcorner_node, leftcorner_node, ref_node, leftcorner_node, False

    @classmethod
    def expand_nt_nt(cls, node: PTree, ref_node1: PTree, ref_node2: PTree) -> (
            PTree, PTree, PTree, PTree):
        parent_node = ref_node2.parent()
        if ref_node1 == parent_node:
            new_right_node = PTree(node.label().split("\\")[0] + "\\" + parent_node.label(),
                                  ["EPS"])
        else:
            new_right_node = PTree(node.label().split("\\")[0] + "\\" + parent_node.label(),
                                  [])

        sibling_node = ref_node2.right_sibling()
        if len(sibling_node) == 1:
            new_left_node = PTree(sibling_node.label(), sibling_node.leaves())
        else:
            new_left_node = PTree(sibling_node.label(), [])

        node.insert(0, new_left_node)
        node.insert(1, new_right_node)
        return sibling_node, sibling_node, ref_node1, parent_node

    @classmethod
    def rev_transform(cls, node: PTree, ref_node: PTree, pick_up_labels=True) -> PTree:
        if find_node_type(ref_node) == NodeType.NT_NT and pick_up_labels:
            node.set_label(ref_node[1].label().split("\\")[1])
        if len(ref_node) == 1 and find_node_type(ref_node) == NodeType.PT:  # base case
            return ref_node
        if find_node_type(ref_node[0]) == NodeType.PT and not is_node_epsilon(ref_node[1]):
            # X -> word X
            pt_node = PTree(ref_node[0].label(), ref_node[0].leaves())
            if node[0] == "":
                node[0] = pt_node
            else:
                node[1] = pt_node
                par_node = PTree("X", [node, ""])
                node = par_node
            return cls.rev_transform(node, ref_node[1], pick_up_labels)
        elif find_node_type(ref_node[0]) != NodeType.PT and is_node_epsilon(ref_node[1]):
            # X -> X X-X
            if node[0] == "":
                raise ValueError(
                    "When reaching the root the left branch should already exist")
            node[1] = cls.rev_transform(PTree("X", ["", ""]), ref_node[0], pick_up_labels)
            return node
        elif find_node_type(ref_node[0]) == NodeType.PT and is_node_epsilon(ref_node[1]):
            # X -> word X-X
            if node[0] == "":
                raise ValueError(
                    "When reaching the end of the chain the left branch should already exist")
            node[1] = PTree(ref_node[0].label(), ref_node[0].leaves())
            return node
        elif find_node_type(ref_node[0]) != NodeType.PT and find_node_type(
                ref_node[1]) != NodeType.PT:
            # X -> X X
            node[1] = cls.rev_transform(PTree("X", ["", ""]), ref_node[0], pick_up_labels)
            par_node = PTree("X", [node, ""])
            return cls.rev_transform(par_node, ref_node[1], pick_up_labels)


class RightCornerTransformer(Transformer):
    @classmethod
    def expand_nt(cls, node: PTree, ref_node: PTree) -> (PTree, PTree, PTree, PTree, bool):
        rightcorner_node = cls.extract_right_corner(ref_node)
        if rightcorner_node == ref_node:
            node.insert(0, ref_node.leaves()[0])
            return None, None, None, None, True
        new_left_node = PTree(node.label() + "\\" + rightcorner_node.label(), [])
        new_right_node = PTree(rightcorner_node.label(), rightcorner_node.leaves())

        node.insert(0, new_left_node)
        node.insert(1, new_right_node)
        return ref_node, rightcorner_node, rightcorner_node, rightcorner_node, False

    @classmethod
    def expand_nt_nt(cls, node: PTree, ref_node1: PTree, ref_node2: PTree) -> (
            PTree, PTree, PTree, PTree):
        parent_node = ref_node2.parent()
        if ref_node1 == parent_node:
            new_left_node = PTree(node.label().split("\\")[0] + "\\" + parent_node.label(),
                                 ["EPS"])
        else:
            new_left_node = PTree(node.label().split("\\")[0] + "\\" + parent_node.label(), [])

        sibling_node = ref_node2.left_sibling()
        if len(sibling_node) == 1:
            new_right_node = PTree(sibling_node.label(), sibling_node.leaves())
        else:
            new_right_node = PTree(sibling_node.label(), [])

        node.insert(0, new_left_node)
        node.insert(1, new_right_node)
        return ref_node1, parent_node, sibling_node, sibling_node

    @classmethod
    def rev_transform(cls, node: PTree, ref_node: PTree, pick_up_labels=True) -> PTree:
        if find_node_type(ref_node) == NodeType.NT_NT and pick_up_labels:
            node.set_label(ref_node[0].label().split("\\")[1])
        if len(ref_node) == 1 and find_node_type(ref_node) == NodeType.PT:  # base case
            return ref_node
        if find_node_type(ref_node[1]) == NodeType.PT and not is_node_epsilon(ref_node[0]):
            # X -> X word
            pt_node = PTree(ref_node[1].label(), ref_node[1].leaves())
            if node[1] == "":
                node[1] = pt_node
            else:
                node[0] = pt_node
                par_node = PTree("X", ["", node])
                node = par_node
            return cls.rev_transform(node, ref_node[0], pick_up_labels)
        elif find_node_type(ref_node[1]) != NodeType.PT and is_node_epsilon(ref_node[0]):
            # X -> X-X X
            if node[1] == "":
                raise ValueError(
                    "When reaching the root the right branch should already exist")
            node[0] = cls.rev_transform(PTree("X", ["", ""]), ref_node[1], pick_up_labels)
            return node
        elif find_node_type(ref_node[1]) == NodeType.PT and is_node_epsilon(ref_node[0]):
            # X -> X-X word
            if node[1] == "":
                raise ValueError(
                    "When reaching the end of the chain the right branch should already exist")
            node[0] = PTree(ref_node[1].label(), ref_node[1].leaves())
            return node
        elif find_node_type(ref_node[1]) != NodeType.PT and find_node_type(
                ref_node[0]) != NodeType.PT:
            # X -> X X
            node[0] = cls.rev_transform(PTree("X", ["", ""]), ref_node[1], pick_up_labels)
            par_node = PTree("X", ["", node])
            return cls.rev_transform(par_node, ref_node[0], pick_up_labels)

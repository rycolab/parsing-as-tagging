import os
import tempfile
from nltk.corpus.reader import DependencyCorpusReader
from nltk.corpus.reader.util import *
from nltk.tree import Tree
from tqdm import tqdm as tq
import sys

LANG_TO_DIR = {
    "bg": "/UD_Bulgarian-BTB/bg_btb-ud-{split}.conllu",
    "ca": "/UD_Catalan-AnCora/ca_ancora-ud-{split}.conllu",
    "cs": "/UD_Czech-PDT/cs_pdt-ud-{split}.conllu",
    "de": "/UD_German-GSD/de_gsd-ud-{split}.conllu",
    "en": "/UD_English-EWT/en_ewt-ud-{split}.conllu",
    "es": "/UD_Spanish-AnCora/es_ancora-ud-{split}.conllu",
    "fr": "/UD_French-GSD/fr_gsd-ud-{split}.conllu",
    "it": "/UD_Italian-ISDT/it_isdt-ud-{split}.conllu",
    "nl": "/UD_Dutch-Alpino/nl_alpino-ud-{split}.conllu",
    "no": "/UD_Norwegian-Bokmaal/no_bokmaal-ud-{split}.conllu",
    "ro": "/UD_Romanian-RRT/ro_rrt-ud-{split}.conllu",
    "ru": "/UD_Russian-SynTagRus/ru_syntagrus-ud-{split}.conllu",
}


# using ^^^ as delimiter
# since ^ never appears

def augment_constituent_tree(const_tree, dep_tree):
    # augment constituent tree leaves into dicts

    assert len(const_tree.leaves()) == len(dep_tree.nodes) - 1

    leaf_nodes = list(const_tree.treepositions('leaves'))
    for i, pos in enumerate(leaf_nodes):
        x = dep_tree.nodes[1 + i]
        y = const_tree[pos].replace("\\", "")
        assert (x['word'] == y), (const_tree, dep_tree)

        # expanding leaves with dependency info
        const_tree[pos] = {
            "word": dep_tree.nodes[1 + i]["word"],
            "head": dep_tree.nodes[1 + i]["head"],
            "rel": dep_tree.nodes[1 + i]["rel"]
        }

    return const_tree


def get_bht(root, offset=0):
    # Return:
    #     The index of the head in this tree
    # and:
    #     The dependant that this tree points to
    if type(root) is dict:
        # leaf node already, return its head
        return 0, root['head']

    # word offset in the current tree
    words_seen = 0
    root_projection = (offset, offset + len(root.leaves()))
    # init the return values to be None
    head, root_points_to = None, None

    # traverse the consituent tree
    for idx, child in enumerate(root):
        head_of_child, child_points_to = get_bht(child, offset + words_seen)
        if type(child) == type(root):
            words_seen += len(child.leaves())
        else:
            # leaf node visited
            words_seen += 1

        if child_points_to < root_projection[0] or child_points_to >= root_projection[1]:
            # pointing to outside of the current tree
            if head is not None:
                print("error! Non-projectivity detected.", root_projection, idx)
                continue  # choose the first child as head
            head = idx
            root_points_to = child_points_to

    if root_points_to is None:
        # self contained sub-sentence
        print("multiple roots detected", root)
        root_points_to = 0

    original_label = root.label()
    root.set_label(f"{original_label}^^^{head}")

    return head, root_points_to


def dep2lex(dep_tree, language="English"):
    # left-first attachment
    def dfs(node_idx):
        dependencies = []
        for rel in dep_tree.nodes[node_idx]["deps"]:
            for dependent in dep_tree.nodes[node_idx]["deps"][rel]:
                dependencies.append((dependent, rel))

        dependencies.append((node_idx, "SELF"))
        if len(dependencies) == 1:
            # no dependent at all, leaf node
            return Tree(
                f"X^^^{dep_tree.nodes[node_idx]['rel']}",
                [
                    Tree(
                        f"{dep_tree.nodes[node_idx]['ctag']}",
                        [
                            f"{dep_tree.nodes[node_idx]['word']}"
                        ]
                    )
                ]
            )
        # Now, len(dependencies) >= 2, sort dependents
        dependencies = sorted(dependencies)

        lex_tree_root = Tree(f"X^^^{0}", [])
        empty_slot = lex_tree_root  # place to fill in the next subtree
        for idx, dependency in enumerate(dependencies):
            if dependency[0] < node_idx:
                # waiting for a head in the right child
                lex_tree_root.set_label(f"X^^^{1}")
                if len(lex_tree_root) == 0:
                    # the first non-head child
                    lex_tree_root.insert(0, dfs(dependency[0]))
                else:
                    # not the first non-head child
                    # insert a sub tree: \
                    #                  X^^^1
                    #                  /   \
                    #                word  [empty_slot]
                    empty_slot.insert(1, Tree(f"X^^^{1}", [dfs(dependency[0])]))
                    empty_slot = empty_slot[1]
            elif dependency[0] == node_idx:
                tree_piece = Tree(
                    f"X^^^{dep_tree.nodes[dependency[0]]['rel']}",
                    [
                        Tree(
                            f"{dep_tree.nodes[dependency[0]]['ctag']}",
                            [
                                f"{dep_tree.nodes[dependency[0]]['word']}"
                            ]
                        )
                    ]
                )
                if len(empty_slot) == 1:
                    # This is the head
                    empty_slot.insert(1, tree_piece)
                else:  # len(empty_slot) == 0
                    lex_tree_root = tree_piece
                pass
            else:
                # moving on to the right of the head
                lex_tree_root = Tree(f"X^^^{0}", [lex_tree_root, dfs(dependency[0])])
        return lex_tree_root

    return dfs(
        dep_tree.nodes[0]["deps"]["root"][0] if "root" in dep_tree.nodes[0]["deps"] else
        dep_tree.nodes[0]["deps"]["ROOT"][0]
    )


def dep2lex_right_first(dep_tree, language="English"):
    # right-first attachment

    def dfs(node_idx):
        dependencies = []
        for rel in dep_tree.nodes[node_idx]["deps"]:
            for dependent in dep_tree.nodes[node_idx]["deps"][rel]:
                dependencies.append((dependent, rel))

        dependencies.append((node_idx, "SELF"))
        if len(dependencies) == 1:
            # no dependent at all, leaf node
            return Tree(
                f"X^^^{dep_tree.nodes[node_idx]['rel']}",
                [
                    Tree(
                        f"{dep_tree.nodes[node_idx]['ctag']}",
                        [
                            f"{dep_tree.nodes[node_idx]['word']}"
                        ]
                    )
                ]
            )
        # Now, len(dependencies) >= 2, sort dependents
        dependencies = sorted(dependencies)

        lex_tree_root = Tree(f"X^^^{0}", [])
        empty_slot = lex_tree_root  # place to fill in the next subtree
        for idx, dependency in enumerate(dependencies):
            if dependency[0] < node_idx:
                # waiting for a head in the right child
                lex_tree_root.set_label(f"X^^^{1}")
                if len(lex_tree_root) == 0:
                    # the first non-head child
                    lex_tree_root.insert(
                        0, dfs(dependency[0])
                    )
                else:
                    # not the first non-head child
                    # insert a sub tree: \
                    #                  X^^^1
                    #                  /   \
                    #                word  [empty_slot]
                    empty_slot.insert(
                        1,
                        Tree(f"X^^^{1}", [
                            dfs(dependency[0])
                        ])
                    )
                    empty_slot = empty_slot[1]
            elif dependency[0] == node_idx:
                # This is the head
                right_branch_root = Tree(
                    f"X^^^{dep_tree.nodes[dependency[0]]['rel']}",
                    [
                        Tree(
                            f"{dep_tree.nodes[dependency[0]]['ctag']}",
                            [
                                f"{dep_tree.nodes[dependency[0]]['word']}"
                            ]
                        )
                    ]
                )
            else:
                # moving on to the right of the head
                right_branch_root = Tree(
                    f"X^^^{0}",
                    [
                        right_branch_root,
                        dfs(dependency[0])
                    ]
                )

        if len(empty_slot) == 0:
            lex_tree_root = right_branch_root
        else:
            empty_slot.insert(1, right_branch_root)

        return lex_tree_root

    return dfs(
        dep_tree.nodes[0]["deps"]["root"][0] if "root" in dep_tree.nodes[0]["deps"] else
        dep_tree.nodes[0]["deps"]["ROOT"][0]
    )


if __name__ == "__main__":
    repo_directory = os.path.abspath(__file__)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        reader = DependencyCorpusReader(
            os.path.dirname(repo_directory),
            [path],
        )
        dep_trees = reader.parsed_sents(path)

        bhts = []
        for dep_tree in tq(dep_trees):
            lex_tree = dep2lex(dep_tree, language="input")
            bhts.append(lex_tree)
            lex_tree_leaves = tuple(lex_tree.leaves())
            dep_tree_leaves = tuple(
                [str(node["word"]) for _, node in sorted(dep_tree.nodes.items())])

            dep_tree_leaves = dep_tree_leaves[1:]

        print(f"Writing BHTs to {os.path.dirname(repo_directory)}/input.bht.test")
        with open(os.path.dirname(repo_directory) + f"/bht/input.bht.test",
                    "w") as fout:
            for lex_tree in bhts:
                fout.write(lex_tree._pformat_flat("", "()", False) + "\n")

        exit(0)

    for language in [
        "English",  # PTB
        "Chinese",  # CTB
        "bg","ca","cs","de","en","es","fr","it","nl","no","ro","ru" # UD2.2
    ]:
        print(f"Processing {language}...")
        if language == "English":
            path = os.path.dirname(repo_directory) + "/ptb/ptb_{split}_3.3.0.sd.clean"
            paths = [path.format(split=split) for split in ["train", "dev", "test"]]
        elif language == "Chinese":
            path = os.path.dirname(repo_directory) + "/ctb/{split}.ctb.conll"
            paths = [path.format(split=split) for split in ["train", "dev", "test"]]
        elif language in ["bg", "ca","cs","de","en","es","fr","it","nl","no","ro","ru"]:
            path = os.path.dirname(repo_directory)+f"/ctb_ptb_ud22/ud2.2/{LANG_TO_DIR[language]}"
            paths = []
            groups = re.match(r'(\w+)_\w+-ud-(\w+)\.conllu', os.path.split(path)[-1])

            for split in ["train", "dev", "test"]:
                conll_path = path.format(split=split)
                command = f"cd ../malt/maltparser-1.9.2/; java -jar maltparser-1.9.2.jar -c {language}_{split} -m proj" \
                        f" -i {conll_path} -o {conll_path}.proj -pp head"
                os.system(command)
                paths.append(conll_path + ".proj")

        reader = DependencyCorpusReader(
            os.path.dirname(repo_directory),
            paths,
        )

        for path, split in zip(paths, ["train", "dev", "test"]):
            print(f"Converting {path} to lexicalized tree")

            with tempfile.NamedTemporaryFile(mode="w") as tmp_file:
                with open(path, "r", encoding='utf-8-sig') as fin:
                    lines = [line for line in fin.readlines() if not line.startswith("#")]
                for idl, line in enumerate(lines):
                    if line.strip() == "":
                        continue
                    assert len(line.split("\t")) == 10, line
                    if '\xad' in line:
                        lines[idl] = lines[idl].replace('\xad', '')
                    if '\ufeff' in line:
                        lines[idl] = lines[idl].replace('\ufeff', '')
                    if " " in line:
                        lines[idl] = lines[idl].replace(" ", ",")
                    if ")" in line:
                        lines[idl] = lines[idl].replace(")", "-RRB-")
                    if "(" in line:
                        lines[idl] = lines[idl].replace("(", "-LRB-")

                tmp_file.writelines(lines)
                tmp_file.flush()
                tmp_file.seek(0)

                dep_trees = reader.parsed_sents(tmp_file.name)

            bhts = []
            for dep_tree in tq(dep_trees):
                lex_tree = dep2lex(dep_tree, language=language)
                bhts.append(lex_tree)
                lex_tree_leaves = tuple(lex_tree.leaves())
                dep_tree_leaves = tuple(
                    [str(node["word"]) for _, node in sorted(dep_tree.nodes.items())])

                dep_tree_leaves = dep_tree_leaves[1:]

            print(f"Writing BHTs to {os.path.dirname(repo_directory)}/{language}.bht.{split}")
            with open(os.path.dirname(repo_directory) + f"/bht/{language}.bht.{split}",
                      "w") as fout:
                for lex_tree in bhts:
                    fout.write(lex_tree._pformat_flat("", "()", False) + "\n")

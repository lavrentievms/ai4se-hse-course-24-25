from datasets import Dataset, load_dataset

##^

DATASET = load_dataset("code-search-net/code_search_net", split="test")
DATASET_PY = DATASET.filter(
        lambda item: item["language"] == "python",
        cache_file_name="dataset_py"
)
DATASET_GO = DATASET.filter(
        lambda item: item["language"] == "go",
        cache_file_name="dataset_go"
)


##^

import tree_sitter as ts
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
PY_PARSER = Parser(PY_LANGUAGE)


FUNCNAME_QUERY = PY_LANGUAGE.query("""
    (function_definition
         name: (identifier) @name
    )
""")

FUNCBODY_QUERY = PY_LANGUAGE.query("""
    (function_definition
         body: (block) @body
    )
""")

COMMENT_QUERY = PY_LANGUAGE.query("""
    (block . (expression_statement (string) @comment))
    (comment) @comment
""")


def assert_not_none[T](value: T | None) -> T:
    if value is None:
        raise RuntimeError("Unexpected None")
    return value


def funcname(whole_func: ts.Tree) -> str:
    node, _ = FUNCNAME_QUERY.captures(whole_func.root_node)[0]
    return assert_not_none(node.text).decode('utf-8')


def funcbody_node(whole_func: ts.Tree) -> ts.Node:
    return FUNCBODY_QUERY.captures(whole_func.root_node)[0][0]


def funcbody(body: ts.Node) -> str:
    return assert_not_none(body.text).decode('utf-8')


def strip_comments(body: ts.Node) -> str:
    source = assert_not_none(body.text)
    body_start = body.start_byte
    comments = COMMENT_QUERY.captures(body)

    stripped = source
    for node, _ in reversed(comments):  # Reverse to maintain indices
        start, end = node.start_byte, node.end_byte
        stripped = stripped[:start - body_start] + stripped[end - body_start:]

    return stripped.decode('utf-8')


##^

from typing import Any


def update_example(example: dict[str, Any]) -> dict[str, Any]:
    wfs = example['whole_func_string']
    tree = PY_PARSER.parse(wfs.encode("utf-8"))
    name = funcname(tree)
    body_node = funcbody_node(tree)
    body = funcbody(body_node)
    body_stripped = strip_comments(body_node)
    return {
            'my_func_name': name,
            'body': body,
            'body_stripped': body_stripped
    }


DATASET_PY_PREPARED = DATASET_PY.map(
        update_example,
        cache_file_name='dataset_py_prepared1'
)


##^

from random import randint


def show_random_example(dataset: Dataset, dataset_prepared: Dataset):
    idx = randint(0, len(dataset))
    example, example_prep = dataset[idx], dataset_prepared[idx]
    wfs = example["whole_func_string"]
    func_name = example["func_name"]
    my_func_name = example_prep["my_func_name"]
    body = example_prep["body"]
    body_stripped = example_prep["body_stripped"]

    print(
            f"\n--- Example #{idx} ---\n"
            f"whole_func_string:\n{wfs}\n"
            f"{func_name=}\n"
            f"{my_func_name=}\n"
            f"body:\n{body}\n"
            f"body_stripped:\n{body_stripped}\n"
    )

##^

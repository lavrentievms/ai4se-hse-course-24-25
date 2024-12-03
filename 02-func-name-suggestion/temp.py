from typing import Any
from datasets import load_dataset

DATASET = load_dataset("code-search-net/code_search_net", split="test")
DATASET_PY = DATASET.filter(lambda item: item["language"] == "python")
DATASET_GO = DATASET.filter(lambda item: item["language"] == "go")


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


def update_example(example: dict[str, Any]) -> dict[str, Any]:
    wfs = example['whole_func_string']
    tree = PY_PARSER.parse(wfs.encode("utf-8"))
    name = funcname(tree)
    body_node = funcbody_node(tree)
    body = funcbody(body_node)
    body_stripped = funcbody(body_node)
    return {
            'my_func_name': name,
            'body': body,
            'body_stripped': body_stripped
    }


DATASET_PY_PREPARED = DATASET_PY.map(update_example)

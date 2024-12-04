from datasets import Dataset, load_dataset

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

from dataclasses import dataclass
from typing import Any

import tree_sitter as ts
from tree_sitter import Language, Parser


def assert_not_none[T](value: T | None) -> T:
    if value is None:
        raise RuntimeError("Unexpected None")
    return value


@dataclass
class Analyzer:
    parser: ts.Parser
    funcname_query: ts.Query
    comment_query: ts.Query
    add_docstr: bool = False

    def extract_funcname(self, whole_func: ts.Tree) -> tuple[str, bytes]:
        node, _ = self.funcname_query.captures(whole_func.root_node)[0]
        name = assert_not_none(node.text).decode("utf-8")
        body = assert_not_none(whole_func.root_node.text)
        wo_name = body[:node.start_byte] + b"<extra_id_0>" + body[node.end_byte:]
        return name, wo_name


    def strip_comments(self, body: ts.Tree) -> str:
        source = assert_not_none(body.root_node.text)
        comments = self.comment_query.captures(body.root_node)

        stripped = source
        for node, _ in reversed(comments):  # Reverse to maintain indices
            start, end = node.start_byte, node.end_byte
            stripped = stripped[:start] + stripped[end:]

        return stripped.decode("utf-8")


    def update_example(self, example: dict[str, Any]) -> dict[str, Any]:
        wfs = example["whole_func_string"]
        tree = self.parser.parse(wfs.encode("utf-8"))
        name, wo_name = self.extract_funcname(tree)
        tree = self.parser.parse(wo_name)
        body = assert_not_none(tree.root_node.text).decode("utf-8")
        if self.add_docstr:
            docstr = example["func_documentation_string"] + "\n"
            body = docstr + body
        body_stripped = self.strip_comments(tree)
        return {
                "my_func_name": name,
                "body": body,
                "body_stripped": body_stripped
        }


##^

import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
PY_PARSER = Parser(PY_LANGUAGE)


PY_FUNCNAME_QUERY = PY_LANGUAGE.query("""
    (function_definition
         name: (identifier) @name
    )
""")

PY_COMMENT_QUERY = PY_LANGUAGE.query("""
    (block . (expression_statement (string) @comment))
    (comment) @comment
""")


PY_ANALYZER = Analyzer(
    parser=PY_PARSER,
    funcname_query=PY_FUNCNAME_QUERY,
    comment_query=PY_COMMENT_QUERY,
)

##^

DATASET_PY_PREPARED = DATASET_PY.map(
        PY_ANALYZER.update_example,
        cache_file_name="dataset_py_prepared"
)

##^

import tree_sitter_go as ts_go

GO_LANGUAGE = Language(ts_go.language())
GO_PARSER = Parser(GO_LANGUAGE)


GO_FUNCNAME_QUERY = GO_LANGUAGE.query("""
    (function_declaration
         name: (identifier) @name
    )
""")

GO_COMMENT_QUERY = GO_LANGUAGE.query("""
    (comment) @comment
""")


GO_ANALYZER = Analyzer(
    parser=GO_PARSER,
    funcname_query=GO_FUNCNAME_QUERY,
    comment_query=GO_COMMENT_QUERY,
    add_docstr=True,
)


def is_go_named_func(whole_func: str) -> bool:
    tree = GO_PARSER.parse(whole_func.encode("utf-8"))
    caps = GO_FUNCNAME_QUERY.captures(tree.root_node)
    return len(caps) > 0

##^

DATASET_GO_PREPARED = (
        DATASET_GO
        .filter(lambda ex: is_go_named_func(ex["whole_func_string"]))
        .map(
            GO_ANALYZER.update_example,
            cache_file_name="dataset_go_prepared")
)


##^

from random import randint


def show_random_prepared_example(dataset: Dataset):
    idx = randint(0, len(dataset) - 1)
    example = dataset[idx]
    wfs = example["whole_func_string"]
    func_name = example["func_name"]
    my_func_name = example["my_func_name"]
    body = example["body"]
    body_stripped = example["body_stripped"]

    print(
            f"\n--- Example #{idx} ---\n"
            f"{func_name=}\n"
            f"{my_func_name=}\n"
            f"\nwhole_func_string:\n\n{wfs}\n"
            f"\nbody:\n\n{body}\n"
            f"\nbody_stripped:\n\n{body_stripped}\n"
    )

##^

from transformers import AutoTokenizer, T5ForConditionalGeneration

CHECKPOINT = "Salesforce/codet5p-220m"
DEVICE = "cpu" # for GPU usage or "cpu" for CPU usage

TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
MODEL = T5ForConditionalGeneration.from_pretrained(CHECKPOINT).to(DEVICE)


def guess_func_name(prepared_func: str) -> str:
    inputs = TOKENIZER.encode(prepared_func, return_tensors="pt").to(DEVICE)
    outputs = MODEL.generate(inputs, max_new_tokens=8)
    raw = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    words = raw.split()
    if words:
        return max(words, key=len)
    else:
        print("can't guess name")
        return "cant_guess_name"

##^


def add_guesses(example: dict[str, Any], idx: int) -> dict[str, Any]:
    body, body_stripped = example["body"], example["body_stripped"]
    guess = guess_func_name(body)
    guess_wo_comments = guess_func_name(body_stripped)
    print(f"Done #{idx}:", example["my_func_name"])
    return {
            'guess': guess,
            'guess_wo_comments': guess_wo_comments
    }

##^


DATASET_PY_WITH_GUESSES = DATASET_PY_PREPARED.take(1000).map(
        add_guesses,
        cache_file_name='dataset_py_with_guesses',
        with_indices=True,
)


##^


DATASET_GO_WITH_GUESSES = DATASET_GO_PREPARED.take(1000).map(
        add_guesses,
        cache_file_name='dataset_go_with_guesses',
        with_indices=True,
)


##^

import evaluate

EXACT_MATCH = evaluate.load("exact_match")
ROUGE = evaluate.load("rouge")

def eval_results(dataset: Dataset, column: str):
    em = EXACT_MATCH.compute(
            references=dataset["my_func_name"],
            predictions=dataset[column])
    rouge = ROUGE.compute(
            references=dataset["my_func_name"],
            predictions=dataset[column])
    print(f"Score (column = {column})")
    print(f"{em=}")
    print(f"{rouge=}")

# Python
# em_w_comments={'exact_match': np.float64(0.212)}
# rouge_w_comments={'rouge1': np.float64(0.5085007936507936), 'rouge2': np.float64(0.2992714285714286), 'rougeL': np.float64(0.5080626984126987), 'rougeLsum': np.float64(0.5075238095238095)}
# em_wo_comments={'exact_match': np.float64(0.145)}
# rouge_wo_comments={'rouge1': np.float64(0.3887785714285713), 'rouge2': np.float64(0.20346547619047617), 'rougeL': np.float64(0.38668531746031753), 'rougeLsum': np.float64(0.3866595238095237)}

# Go
# em_w_comments={'exact_match': np.float64(0.746)}
# rouge_w_comments={'rouge1': np.float64(0.752), 'rouge2': np.float64(0.0), 'rougeL': np.float64(0.7526666666666667), 'rougeLsum': np.float64(0.7516666666666667)}
# em_wo_comments={'exact_match': np.float64(0.1)}
# rouge_wo_comments={'rouge1': np.float64(0.164), 'rouge2': np.float64(0.0), 'rougeL': np.float64(0.16366666666666668), 'rougeLsum': np.float64(0.164)}


##^

from random import randint


def show_random_guesses(dataset: Dataset):
    idx = randint(0, len(dataset) - 1)
    example = dataset[idx]
    real_name = example["my_func_name"]
    guess_with_comments = example["guess"]
    guess_wo_comments = example["guess_wo_comments"]

    rouge_w_comments = assert_not_none(ROUGE.compute(
        references=[real_name],
        predictions=[guess_with_comments]
    ))["rouge1"]
    rouge_wo_comments = assert_not_none(ROUGE.compute(
        references=[real_name],
        predictions=[guess_wo_comments]
    ))["rouge1"]

    full_text = example["body"]

    print(
            f"\n--- Example #{idx} ---\n"
            f"{real_name=}\n"
            f"{guess_with_comments=}\n"
            f"{guess_wo_comments=}\n\n"
            f"{rouge_w_comments=}\n"
            f"{rouge_wo_comments=}\n\n"
            f"{full_text}\n"
    )

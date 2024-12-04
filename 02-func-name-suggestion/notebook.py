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


def extract_funcname(whole_func: ts.Tree) -> tuple[str, bytes]:
    node, _ = FUNCNAME_QUERY.captures(whole_func.root_node)[0]
    name = assert_not_none(node.text).decode("utf-8")
    body = assert_not_none(whole_func.root_node.text)
    wo_name = body[:node.start_byte] + b"<extra_id_0>" + body[node.end_byte:]
    return name, wo_name


def strip_comments(body: ts.Tree) -> str:
    source = assert_not_none(body.root_node.text)
    comments = COMMENT_QUERY.captures(body.root_node)

    stripped = source
    for node, _ in reversed(comments):  # Reverse to maintain indices
        start, end = node.start_byte, node.end_byte
        stripped = stripped[:start] + stripped[end:]

    return stripped.decode("utf-8")


##^

from typing import Any


def update_example(example: dict[str, Any]) -> dict[str, Any]:
    wfs = example["whole_func_string"]
    tree = PY_PARSER.parse(wfs.encode("utf-8"))
    name, wo_name = extract_funcname(tree)
    tree = PY_PARSER.parse(wo_name)
    body = assert_not_none(tree.root_node.text).decode("utf-8")
    body_stripped = strip_comments(tree)
    return {
            "my_func_name": name,
            "body": body,
            "body_stripped": body_stripped
    }


DATASET_PY_PREPARED = DATASET_PY.map(
        update_example,
        cache_file_name="dataset_py_prepared"
)


##^

from random import randint


def show_random_prepared_example(dataset: Dataset):
    idx = randint(0, len(dataset))
    example = dataset[idx]
    wfs = example["whole_func_string"]
    func_name = example["func_name"]
    my_func_name = example["my_func_name"]
    body = example["body"]
    body_stripped = example["body_stripped"]

    print(
            f"\n--- Example #{idx} ---\n"
            f"whole_func_string:\n{wfs}\n"
            f"{func_name=}\n"
            f"{my_func_name=}\n"
            f"body:\n{body}\n"
            f"body_stripped:\n{body_stripped}\n"
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


DATASET_PY_WITH_GUESSES = DATASET_PY_PREPARED.take(1000).map(
        add_guesses,
        cache_file_name='dataset_py_with_guesses',
        with_indices=True,
)


##^

from random import randint


def show_random_guesses(dataset: Dataset):
    idx = randint(0, len(dataset))
    example = dataset[idx]
    real_name = example["my_func_name"]
    guess_with_comments = example["guess"]
    guess_wo_comments = example["guess_wo_comments"]

    print(
            f"\n--- Example #{idx} ---\n"
            f"{real_name=}\n"
            f"{guess_with_comments=}\n"
            f"{guess_wo_comments=}\n"
    )


##^

import evaluate

EXACT_MATCH = evaluate.load("exact_match")
ROUGE = evaluate.load("rouge")

def eval_results(dataset: Dataset):
    em_w_comments = EXACT_MATCH.compute(
            references=dataset["my_func_name"],
            predictions=dataset["guess"])
    rouge_w_comments = ROUGE.compute(
            references=dataset["my_func_name"],
            predictions=dataset["guess"])
    print(f"{em_w_comments=}")
    print(f"{rouge_w_comments=}")

    em_wo_comments = EXACT_MATCH.compute(
            references=dataset["my_func_name"],
            predictions=dataset["guess_wo_comments"])
    rouge_wo_comments = ROUGE.compute(
            references=dataset["my_func_name"],
            predictions=dataset["guess_wo_comments"])
    print(f"{em_wo_comments=}")
    print(f"{rouge_wo_comments=}")

# em_w_comments={'exact_match': np.float64(0.212)}
# rouge_w_comments={'rouge1': np.float64(0.5085007936507936), 'rouge2': np.float64(0.2992714285714286), 'rougeL': np.float64(0.5080626984126987), 'rougeLsum': np.float64(0.5075238095238095)}
# em_wo_comments={'exact_match': np.float64(0.145)}
# rouge_wo_comments={'rouge1': np.float64(0.3887785714285713), 'rouge2': np.float64(0.20346547619047617), 'rougeL': np.float64(0.38668531746031753), 'rougeLsum': np.float64(0.3866595238095237)}

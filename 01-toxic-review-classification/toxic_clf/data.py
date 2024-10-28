from pathlib import Path

import datasets
import pandas as pd


def prepare(raw_data: Path) -> datasets.Dataset:
    data = pd.read_excel(raw_data)
    return datasets.Dataset.from_pandas(data.dropna())


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))

"""Script to split a csv file into training, validation and test csv files."""

import logging
from pathlib import Path

import fire
import numpy as np
import pandas as pd


def split(
    csv_path: str,
    training_ratio: float = 1 / 2,
    validation_ratio: float = 1 / 3,
    seed: int = 0,
) -> None:
    """Split a csv file rows into three csv files. Write these files next to
    the original one by appending '_train.csv', '_val.csv' and '_test.csv' to
    the original filename.

    Args:
        csv_path (str): path to the csv file to split.
        training ratio (float): ratio of rows to put in the training file.
        validation_ratio (float): ratio of rows to put in the validation file.
        seed (int): random seed to reproduce the split.
    """

    np.random.seed(seed)

    dataframe = pd.read_csv(csv_path)
    dataset_size = dataframe.shape[0]
    permutation = np.random.permutation(dataset_size)

    train_df = dataframe.iloc[permutation[: int(training_ratio * dataset_size)]]
    validation_df = dataframe.iloc[
        permutation[
            int(training_ratio * dataset_size) : int(
                (training_ratio + validation_ratio) * dataset_size
            )
        ]
    ]
    test_df = dataframe.iloc[
        permutation[int((training_ratio + validation_ratio) * dataset_size) :]
    ]

    train_df.to_csv(str(Path(csv_path).with_suffix("")) + "_train.csv", index=False)
    validation_df.to_csv(str(Path(csv_path).with_suffix("")) + "_val.csv", index=False)
    test_df.to_csv(str(Path(csv_path).with_suffix("")) + "_test.csv", index=False)
    logging.getLogger(__name__).info(
        f"Split csv files saved next to the file {csv_path}."
    )


if __name__ == "__main__":
    fire.Fire(split)

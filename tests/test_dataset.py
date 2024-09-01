import pathlib

import pytest
import torch
from torch import Tensor

from PROJECT_NAME.dataset import ImageClassificationDataset

toy_dataset_path = pathlib.Path(__file__).parent / "data/toy_dataset"


@pytest.fixture
def dataset() -> ImageClassificationDataset:
    return ImageClassificationDataset(
        annotations_file=toy_dataset_path / "labels/labels.csv",
        img_dir=toy_dataset_path / "images",
        labels=["BELGIUM", "FRANCE"],
    )


@pytest.fixture
def dataset_zero_transform() -> ImageClassificationDataset:
    def zero_transform(image: Tensor) -> Tensor:
        return torch.zeros_like(image)

    return ImageClassificationDataset(
        annotations_file=toy_dataset_path / "labels/labels.csv",
        img_dir=toy_dataset_path / "images",
        labels=["BELGIUM", "FRANCE"],
        transform=zero_transform,
    )


def test_image_classification_dataset_len(dataset) -> None:
    """Test the len method of the ImageClassificationDataset class."""

    assert len(dataset) == 8


def test_image_classification_dataset_getitem(dataset) -> None:
    """Test the getitem method of the ImageClassificationDataset class."""

    for data in dataset:
        assert data["img"].shape == (3, 32, 32)
        assert isinstance(data["img_relative_path"], str)
        assert data["label"].shape == (1,)


def test_image_classification_dataset_transform(dataset_zero_transform) -> None:
    """Test transform initialization parameter of the
    ImageClassificationDataset class."""

    for data in dataset_zero_transform:
        assert torch.equal(data["img"], torch.zeros((3, 32, 32)))


def test_image_classification_dataset_get_image_relative_path(dataset) -> None:
    """Test get_image_relative_path method of the ImageClassificationDataset
    class."""

    assert dataset[0]["img_relative_path"] == "0.png"
    assert dataset[1]["img_relative_path"] == "1.png"

import pathlib

import pytest
import torch

from PROJECT_NAME.models import BaselineClassicationModel

toy_dataset_path = pathlib.Path(__file__).parent / "data/toy_dataset"


@pytest.fixture
def baseline_classification_model() -> BaselineClassicationModel:
    return BaselineClassicationModel(
        input_size=32 * 32,
        output_dimension=5,
    )


def test_baseline_classification_model_init(baseline_classification_model) -> None:
    """Test the init method of the BaselineClassicationModel class."""
    # Note: nothing to do, we just test the fixture instantiation.
    pass


def test_baseline_classification_model_forward(baseline_classification_model) -> None:
    """Test the forward method of the BaselineClassicationModel class."""

    x = torch.rand(2, 32, 32)
    y = baseline_classification_model(x)
    assert y.shape == (2, 5)

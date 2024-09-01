import tempfile
from pathlib import Path

import pytest
import torch

from PROJECT_NAME.models import BaselineClassicationModel
from PROJECT_NAME.onnx_conversion import convert_checkpoint_to_onnx


@pytest.fixture
def baseline_classification_model() -> BaselineClassicationModel:
    return BaselineClassicationModel(
        input_size=32 * 32,
        output_dimension=5,
    )


def test_convert_checkpoint_to_onnx(baseline_classification_model):
    """Test the test_convert_checkpoint_to_onnx function on a baseline
    classification model."""
    sample_input = torch.rand(2, 32, 32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        convert_checkpoint_to_onnx(
            model=baseline_classification_model,
            sample_input=sample_input,
            onnx_file_path=Path(tmp_dir) / "model.onnx",
        )

import pathlib

import pytest
from torch.nn import BCEWithLogitsLoss
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from PROJECT_NAME.lightning_modules import ClassificationLightningModule
from PROJECT_NAME.models import BaselineClassicationModel

toy_dataset_path = pathlib.Path(__file__).parent / "data/toy_dataset"


@pytest.fixture
def lightning_module_baseline_classification_model() -> ClassificationLightningModule:
    return ClassificationLightningModule(
        model=BaselineClassicationModel(input_size=3 * 32 * 32, output_dimension=1),
        loss=BCEWithLogitsLoss(),
        metrics={
            "accuracy": BinaryAccuracy(threshold=0.0),
            "f1_score": BinaryF1Score(threshold=0.0),
            "precision": BinaryPrecision(threshold=0.0),
            "recall": BinaryRecall(threshold=0.0),
        },
        learning_rate=1e-4,
    )


def test_lightning_module_init(lightning_module_baseline_classification_model) -> None:
    """Test the init method of the ClassificationLightningModule class."""
    # Note: nothing to do, we just test the fixture instantiation.
    pass

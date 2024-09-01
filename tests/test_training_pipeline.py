import tempfile
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ColorJitter, Compose, RandomPerspective

from PROJECT_NAME.dataset import ImageClassificationDataset
from PROJECT_NAME.lightning_modules import ClassificationLightningModule
from PROJECT_NAME.models import BaselineClassicationModel
from PROJECT_NAME.training_pipeline import training_pipeline

toy_dataset_path = Path(__file__).parent / "data/toy_dataset"


def test_training_pipeline() -> None:
    """Test the training pipeline."""

    torch.manual_seed(0)
    np.random.seed(0)

    lightning_module = ClassificationLightningModule(
        model=BaselineClassicationModel(input_size=3 * 32 * 32, output_dimension=1),
        loss=torch.nn.BCEWithLogitsLoss(),
        metrics={
            "accuracy": torchmetrics.classification.BinaryAccuracy(threshold=0.0),
            "f1_score": torchmetrics.classification.BinaryF1Score(threshold=0.0),
            "precision": torchmetrics.classification.BinaryPrecision(threshold=0.0),
            "recall": torchmetrics.classification.BinaryRecall(threshold=0.0),
        },
        learning_rate=1e-2,
        cyclic_lr=True,
    )

    train_dataloader = DataLoader(
        dataset=ImageClassificationDataset(
            annotations_file="../tests/data/toy_dataset/labels/labels_train.csv",
            img_dir="../tests/data/toy_dataset/images",
            labels=["BELGIUM", "FRANCE"],
            transform=Compose(
                [
                    ColorJitter(
                        brightness=0.5,
                        saturation=0.5,
                        contrast=0.5,
                        hue=0.1,
                    ),
                    RandomPerspective(p=0.5, distortion_scale=0.25),
                ]
            ),
        ),
        batch_size=2,
        num_workers=1,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        dataset=ImageClassificationDataset(
            annotations_file="../tests/data/toy_dataset/labels/labels_val.csv",
            img_dir="../tests/data/toy_dataset/images",
            labels=["BELGIUM", "FRANCE"],
        ),
        batch_size=2,
        num_workers=1,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        dataset=ImageClassificationDataset(
            annotations_file="../tests/data/toy_dataset/labels/labels_test.csv",
            img_dir="../tests/data/toy_dataset/images",
            labels=["BELGIUM", "FRANCE"],
        ),
        batch_size=2,
        num_workers=1,
        shuffle=False,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            default_root_dir=tmp_dir,
            max_epochs=4,
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(Path(tmp_dir) / "checkpoints"),
                    monitor="val_f1_score",
                    mode="max",
                    save_last=True,
                    filename="best-{epoch}-{val_f1_score:.4f}",
                    every_n_epochs=1,
                    save_top_k=1,
                )
            ],
            logger=TensorBoardLogger(
                save_dir=str(Path(tmp_dir) / "tb_logs"),
                name="Classification",
            ),
            log_every_n_steps=1,
            accelerator="cpu",
            devices=1,
            deterministic=True,
        )

        training_pipeline(
            output_directory=tmp_dir,
            lightning_module=lightning_module,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            trainer=trainer,
        )

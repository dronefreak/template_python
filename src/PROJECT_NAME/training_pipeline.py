"""Module implementing a model training pipeline."""

import glob
from pathlib import Path

import yaml
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from PROJECT_NAME.lightning_modules import ClassificationLightningModule
from PROJECT_NAME.onnx_conversion import convert_checkpoint_to_onnx


def training_pipeline(
    *,
    output_directory: str,
    lightning_module: LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    trainer: Trainer,
) -> None:
    """Training pipeline. Train the model using Pytorch lightning, evaluate the
    best one (selected the validation set) on the test set and convert it to
    onnx.

    Args:
        output_directory (str): output directory to store the logs, the checkpoints
            and the onnx models.
        lightning_module (LightningModule): lightning module encapsulating
            the model, the train, val and test steps, the loss computation...
        train_dataloader (DataLoader): train dataloader.
        val_dataloader (DataLoader): validation dataloader
        test_dataloader (DataLoader): test dataloader.
        trainer (Trainer): Pytorch lightning trainer to train the model.
    """
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Evaluate the best model on the test set
    test_results = trainer.test(model=lightning_module, dataloaders=test_dataloader)
    with open(Path(output_directory) / "test_results.yml", "w") as file:
        yaml.dump(test_results, file, default_flow_style=False)

    # Convert best model to onnx
    best_model_checkpoints = glob.glob(
        str(Path(output_directory) / "checkpoints/best*.ckpt")
    )
    assert len(best_model_checkpoints) == 1
    best_model_checkpoint = best_model_checkpoints[0]
    lightning_module = ClassificationLightningModule.load_from_checkpoint(
        best_model_checkpoint,
        model=lightning_module.model,
    )
    model = lightning_module.model
    sample_input = next(iter(val_dataloader))["img"]
    convert_checkpoint_to_onnx(
        model=model,
        onnx_file_path=str(Path(output_directory) / "model.onnx"),
        sample_input=sample_input,
    )

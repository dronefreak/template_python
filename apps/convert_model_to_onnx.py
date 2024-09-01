"""Application used to convert a Pytorch Lightning checkpoint into an onnx
model."""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf.dictconfig import DictConfig

from PROJECT_NAME.lightning_modules import ClassificationLightningModule
from PROJECT_NAME.onnx_conversion import convert_checkpoint_to_onnx


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="convert_model_to_onnx",
)
def main(config: DictConfig) -> None:
    """Application main function. Convert a Pytorch lightning checkpoint into
    an onnx model.

    Args:
        config (dict): Hydra config file.
    """
    output_directory = HydraConfig.get().runtime.output_dir

    dataloader = hydra.utils.instantiate(config.dataloader)
    sample_input = next(iter(dataloader))["img"]

    checkpoint = config.checkpoint
    lightning_module = ClassificationLightningModule.load_from_checkpoint(
        checkpoint,
        model=hydra.utils.instantiate(config.lightning_module.model),
    )
    model = lightning_module.model

    logging.getLogger(__name__).info("Convert Pytorch model to ONNX format.")
    onnx_file_path = Path(output_directory) / "model.onnx"
    convert_checkpoint_to_onnx(
        model=model,
        onnx_file_path=str(onnx_file_path),
        sample_input=sample_input,
    )
    logging.getLogger(__name__).info(f"ONNX model saved at {str(onnx_file_path)}")


if __name__ == "__main__":
    main()

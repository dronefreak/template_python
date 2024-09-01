"""Application used to peform an inference with a trained model using Pytorch
Lightning."""

import logging
from pathlib import Path

import cv2
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf.dictconfig import DictConfig

from PROJECT_NAME.lightning_modules import ClassificationLightningModule


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="inference",
)
def main(config: DictConfig) -> None:
    """Application main function. Perform an inference using a model trained
    with Pytorch Lightning.

    Args:
        config (dict): Hydra config file.
    """
    output_directory = HydraConfig.get().runtime.output_dir

    device = config.device
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

    dataset = hydra.utils.instantiate(config.dataset)

    checkpoint = config.checkpoint
    lightning_module = ClassificationLightningModule.load_from_checkpoint(
        checkpoint,
        model=hydra.utils.instantiate(config.lightning_module.model),
    )
    model = lightning_module.model
    model.eval()
    model.to(device)

    logging.getLogger(__name__).info("Run the model on the dataset images.")
    for data in dataset:
        image = data["img"]
        image_batch = image.unsqueeze(0).to(device)
        image_array = image.permute((1, 2, 0)).numpy()

        with torch.no_grad():
            pred = model(image_batch)
            pred = pred.to("cpu")

            logit = pred.squeeze().numpy()
            predicted_class = 1 if logit >= 0 else 0

            cv2.imwrite(
                str(
                    Path(output_directory)
                    / (str(predicted_class) + "_" + data["img_relative_path"])
                ),
                cv2.cvtColor(image_array * 255, cv2.COLOR_RGB2BGR),
            )

    logging.getLogger(__name__).info(f"Inference results saved in {output_directory}")


if __name__ == "__main__":
    main()

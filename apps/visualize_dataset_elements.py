"""Application used to visualize a few random elements drawn from a dataset.

It is mainly used to visualize the impact of the data augmentation on
the training set.
"""

import logging
import random
from pathlib import Path

import cv2
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf.dictconfig import DictConfig


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="visualize_dataset_elements",
)
def main(config: DictConfig) -> None:
    """Application main function. Instantiate a dataset from an Hydra config
    file and save a few randomly sampled images in the Hydra experiment folder.

    Args:
        config (dict): Hydra config file.
            It should includes a 'dataset' field
            and a 'number_sampled_elements' field.
    """

    output_directory = HydraConfig.get().runtime.output_dir
    dataset = hydra.utils.instantiate(config.dataset)
    number_sampled_elements = config.number_sampled_elements

    sampled_ids = random.choices(
        range(len(dataset)),
        k=number_sampled_elements,
    )
    logging.getLogger(__name__).info(
        "Extract sampled dataset images for visualization."
    )
    for id in sampled_ids:
        image = dataset[id]["img"]
        image = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(Path(output_directory) / dataset[id]["img_relative_path"]),
            image * 255,
        )
    logging.getLogger(__name__).info(
        f"Sampled dataset images saved in {output_directory}"
    )


if __name__ == "__main__":
    main()

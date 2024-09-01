"""Application used to train a model using Pytorch Lightning."""

import logging

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf.dictconfig import DictConfig

from PROJECT_NAME.training_pipeline import training_pipeline


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="train",
)
def main(config: DictConfig) -> None:
    """Application main function. Train a model using Pytorch Lightning.

    Args:
        config (dict): Hydra config file.
            It should define a lightning module, a train, validation and test
            dataloaders, and a trainer.
    """
    random_seed = config.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    output_directory = HydraConfig.get().runtime.output_dir

    lightning_module = hydra.utils.instantiate(config.lightning_module)
    train_dataloader = hydra.utils.instantiate(config.train_dataloader)
    val_dataloader = hydra.utils.instantiate(config.val_dataloader)
    test_dataloader = hydra.utils.instantiate(config.test_dataloader)
    trainer = hydra.utils.instantiate(config.trainer)

    training_pipeline(
        output_directory=output_directory,
        lightning_module=lightning_module,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        trainer=trainer,
    )

    logging.getLogger(__name__).info(f"Training files saved in {output_directory}")


if __name__ == "__main__":
    main()

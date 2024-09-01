"""Application used to peform an inference with an onnx model."""

import logging
from pathlib import Path

import cv2
import hydra
import onnx
import onnxruntime
from hydra.core.hydra_config import HydraConfig
from omegaconf.dictconfig import DictConfig


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="inference_onnx",
)
def main(config: DictConfig) -> None:
    """Application main function. Perform an inference with an onnx model.

    Args:
        config (dict): Hydra config file.
    """
    output_directory = HydraConfig.get().runtime.output_dir

    dataset = hydra.utils.instantiate(config.dataset)

    onnx_file_path = config.onnx_file_path
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(
        onnx_file_path, providers=["CPUExecutionProvider"]
    )

    logging.getLogger(__name__).info("Run the model on the dataset images.")
    for data in dataset:
        image = data["img"]
        image_array = image.permute((1, 2, 0)).numpy()

        ort_outputs = ort_session.run(None, {"input_1": image.unsqueeze(0).numpy()})
        logit = ort_outputs[0].squeeze()
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

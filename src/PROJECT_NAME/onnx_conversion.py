"""Module to convert a Pytorch model into an onnx model."""

import numpy as np
import onnx
import onnxruntime
import torch


def convert_checkpoint_to_onnx(
    *, model: torch.nn.Module, onnx_file_path: str, sample_input: torch.Tensor
) -> None:
    """Convert a Pytorch model into an onnx one.

    Args:
        model (torch.nn.Module): Pytorch model to convert.
        onnx_file_path (str): path of the resulting onnx model.
        sample_input(torch.Tensor): sample input used to trace the model operations.
    """

    model.eval()
    model = model.to("cpu")

    torch.onnx.export(
        model,
        sample_input,
        onnx_file_path,
        input_names=["input_1"],
        output_names=["output_1"],
        dynamic_axes={"input_1": {0: "batch_size"}, "output_1": {0: "batch_size"}},
    )

    # Test the onnx model gives similar results
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(
        onnx_file_path, providers=["CPUExecutionProvider"]
    )
    onnx_output = ort_session.run(None, {"input_1": sample_input.numpy()})[0]
    with torch.no_grad():
        torch_output = model(sample_input)
    np.testing.assert_almost_equal(torch_output, onnx_output, decimal=3)

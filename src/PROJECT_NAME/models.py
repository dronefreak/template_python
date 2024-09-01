"""Module providing Pytorch models."""

from torch import Tensor, nn


class BaselineClassicationModel(nn.Module):
    """Baseline classification model, used mainly for test purposes."""

    def __init__(self, *, input_size: int, output_dimension: int) -> None:
        """Initialization method for the class BaselineClassicationModel.

        Args:
            input_size (int): Number of pixels of the input images.
            output_dimension (int): Number of output logits.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dimension),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the model on a given input.

        Args:
            x (Tensor): input on which the model is applied.

        Returns:
            Tensor: logits predicted by the model.
        """

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

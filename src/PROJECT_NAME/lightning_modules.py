"""Module providing a LightningModule for trainings using Pytorch Lightning."""

import torch
from lightning import LightningModule
from torch import Tensor


class ClassificationLightningModule(LightningModule):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        loss: torch.nn.Module = None,
        metrics: dict = None,
        learning_rate: float = None,
        cyclic_lr: bool = False,
    ) -> None:
        """Initialization method for the class ClassificationLightningModule.

        Args:
            model (torch.nn.Module): Internal model used in the LightningModule.
            loss (torch.nn.Module): Loss used during the training.
            metrics (dict): Dictionary of metrics used during the training.
            learning_rate (float): learning rate to be used by the optimizer.
            cyclic_lr (bool): if True, use a clyclic learning rate scheduler.
        """

        super().__init__()
        self.model = model
        self._loss = loss
        if metrics:
            self._metrics = torch.nn.ModuleDict(metrics)
        self._learning_rate = learning_rate
        self._cyclic_lr = cyclic_lr

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Training step used during the training.

        Args:
            batch (dict): Dictionary with keys "img" and "label".
            batch_idx (int): id of the current batch.

        Returns:
            Tensor: Loss value for the current batch.
        """
        x = batch["img"]
        labels = batch["label"]
        batch_size = labels.shape[0]
        pred = self.model(x)
        loss = self._loss(pred, labels.float())
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        for metric_name, metric in self._metrics.items():
            value = metric(torch.where(pred > 0, 1, 0), labels)
            self.log(
                f"train_{metric_name}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Validation step used during the training.

        Args:
            batch (dict): Dictionary with keys "img" and "label".
            batch_idx (int): id of the current batch.
        """
        x = batch["img"]
        labels = batch["label"]
        batch_size = labels.shape[0]
        pred = self.model(x)
        loss = self._loss(pred, labels.float())
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        for metric_name, metric in self._metrics.items():
            value = metric(torch.where(pred > 0, 1, 0), labels)
            self.log(
                f"val_{metric_name}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step used after the training.

        Args:
            batch (dict): Dictionary with keys "img" and "label".
            batch_idx (int): id of the current batch.
        """
        x = batch["img"]
        labels = batch["label"]
        batch_size = labels.shape[0]
        pred = self.model(x)
        loss = self._loss(pred, labels.float())
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        for metric_name, metric in self._metrics.items():
            value = metric(torch.where(pred > 0, 1, 0), labels)
            self.log(
                f"test_{metric_name}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self) -> object | dict:
        """Define the optimizer for the training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        if not self._cyclic_lr:
            return optimizer
        else:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=1e-2 * self._learning_rate,
                max_lr=self._learning_rate,
                step_size_up=256,
                cycle_momentum=False,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

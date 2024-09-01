"""Module implementing a dataset for image classification."""

from collections.abc import Callable
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class ImageClassificationDataset(Dataset):
    """Dataset used to load images and classification labels."""

    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        labels: List[str],
        transform: Callable[[Tensor], Tensor] = None,
    ) -> None:
        """Initialization method for the class ImageClassificationDataset.

        Args:
            annotations_file (str): CSV annotation file with two rows;
                image path (relative to img_dir) and image label.
            img_dir (str): directory which includes the dataset images.
            labels (List[str]): list of possible image labels.
            transform (Callable[[Tensor], Tensor]): optional function
                to be applied on the images.
        """

        self._img_labels = pd.read_csv(annotations_file)
        self._img_dir = img_dir
        self._labels_dict = {}
        for i, label in enumerate(labels):
            self._labels_dict[label] = i
        self.transform = transform

    def __len__(self) -> int:
        """Return the numer of elements in the dataset.

        Returns:
            int: Number of elements included in the dataset.
        """

        return len(self._img_labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return the dataset element with id idx.

        Args:
            idx: id of the element to retrieve.

        Returns:
            Tuple[dict, Tensor]: Dictionary, including the image and its relative
                path, and the associated label (as a single value tensor).
                The returned label is the index of the label in the label list used
                to initialize this dataset.
        """
        img_path = Path(self._img_dir) / self._img_labels.iloc[idx, 0]
        image = read_image(str(img_path), mode=ImageReadMode.RGB)
        image = image / 255.0
        if self.transform:
            image = self.transform(image)

        label = self._labels_dict[self._img_labels.iloc[idx, 1]]

        return {
            "img": image,
            "img_relative_path": img_path.name,
            "label": torch.tensor([label]),
        }

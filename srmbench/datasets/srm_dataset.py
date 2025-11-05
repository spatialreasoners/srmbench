from abc import ABC, abstractmethod
from typing import Literal

from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class SRMDataset(ABC, Dataset):
    def __init__(self, stage: Literal["train", "test"] = "train") -> None:
        self.stage = stage

    @property
    def is_deterministic(self) -> bool:
        return self.stage != "train"

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> tuple[Image.Image, Image.Image] | Image.Image: # Either a tuple of image and mask or just the image
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

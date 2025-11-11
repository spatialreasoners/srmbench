from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool
from PIL import Image

from .srm_dataset import SRMDataset


class EvenPixelsDataset(SRMDataset):
    image_shape = (32, 32)
    mask_shape = (32, 32)

    def __init__(
        self,
        stage: Literal["train", "test"] = "train",
        saturation: float = 1.0,
        value: float = 0.7,
        dataset_size: int | None = None,
        transform=None,
    ) -> None:
        super().__init__(stage)

        if saturation < 0 or saturation > 1:
            raise ValueError("saturation must be between 0 and 1")
        if value < 0 or value > 1:
            raise ValueError("value must be between 0 and 1")

        if dataset_size is not None:
            if dataset_size <= 0:
                raise ValueError("dataset_size must be greater than 0")
            self.dataset_size = dataset_size
        else:
            self.dataset_size = 1_000_000 if stage == "train" else 10000

        self.saturation = saturation
        self.value = value
        self.transform = transform

    @staticmethod
    def _get_even_binary_mask(
        w: int, h: int, rng: np.random.Generator | None
    ) -> Bool[np.ndarray, "h w"]:
        num_ones = int(w * h / 2)
        flat_mask = np.zeros(w * h)
        flat_mask[:num_ones] = 1

        if rng is not None:
            rng.shuffle(flat_mask)
        else:
            np.random.shuffle(flat_mask)

        return flat_mask.astype(bool).reshape(w, h)

    def _get_image(self, rng: np.random.Generator | None) -> Image.Image:
        w, h = self.image_shape

        if rng is not None:
            hue_offset = rng.uniform(0, 0.5)

        else:
            hue_offset = np.random.uniform(0, 0.5)

        data = np.zeros((w, h, 3))
        data[:, :, 0] = (self._get_even_binary_mask(w, h, rng) * 0.5) + hue_offset
        data[:, :, 1] = self.saturation
        data[:, :, 2] = self.value

        return Image.fromarray(np.uint8(data * 255), "HSV").convert("RGB")

    def __getitem__(self, idx: int) -> Image.Image:
        rng = np.random.default_rng(idx) if self.is_deterministic else None
        image = self._get_image(rng)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image

    def __len__(self) -> int:
        return 1000_000

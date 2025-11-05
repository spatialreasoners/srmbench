"""
FFHQ dataset implementation using HuggingFace dataset.
Uses nuwandaa/ffhq128 which provides 128x128 thumbnail images.
"""

from typing import Literal

from datasets import load_dataset
from PIL import Image

from ..srm_dataset import SRMDataset


class FFHQDataset(SRMDataset):
    """
    FFHQ (Flickr-Faces-HQ) dataset implementation.
    
    Uses HuggingFace dataset nuwandaa/ffhq128 which provides 128x128 thumbnail images.
    Images are automatically downloaded and cached on first use.
    """

    training_set_ratio: float = 0.95  # Match counting_polygons split ratio

    def __init__(
        self,
        stage: Literal["train", "test"] = "train",
        root: str | None = None,  # Kept for compatibility, not used
        image_resolution: tuple[int, int] = (256, 256),
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(stage)
        self.image_resolution = image_resolution

        # Load FFHQ 128x128 thumbnails from HuggingFace
        split = "train"
        self._dataset = load_dataset(
            "nuwandaa/ffhq128",
            split=split,
            cache_dir=cache_dir,
        )

        # Split train/test (95/5 split like counting_polygons)
        total = len(self._dataset)
        train_size = int(self.training_set_ratio * total)

        if stage == "test":
            self._dataset = self._dataset.select(range(train_size, total))
        else:
            self._dataset = self._dataset.select(range(train_size))

    def __getitem__(self, idx: int) -> Image.Image:
        """
        Get an image from the dataset.
        
        Returns a PIL Image. If the requested resolution differs from 128x128,
        the image will be resized using bicubic interpolation.
        """
        item = self._dataset[idx]
        image = item["image"]  # PIL Image (128x128)

        # Resize to desired resolution if needed
        if image.size != self.image_resolution:
            image = image.resize(self.image_resolution, Image.BICUBIC)

        return image

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def _num_available(self) -> int:
        """Number of available images in the dataset."""
        return len(self)

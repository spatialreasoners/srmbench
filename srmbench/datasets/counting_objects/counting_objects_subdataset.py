from abc import ABC, abstractmethod
from colorsys import hsv_to_rgb
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import CenterCrop, Compose, Lambda

from ..srm_dataset import SRMDataset
from .counting_objects import CountingObjectsBase


class CountingObjectsSubdataset(CountingObjectsBase, ABC):
    """
    Subdataset class that merges base images with the objects from Counting Objects dataset.
    """

    color_histogram_blur_sigma: float = 26.0
    color_histogram_blur_kernel_size: int = 255  # 256 possible values

    def __init__(
        self,
        subdataset: SRMDataset,
        stage: str = "train",
        root: str | None = None,
        image_resolution: tuple[int, int] = (128, 128),
        labeler_name: str | None = None,
        are_nums_on_images: bool = True,
        supersampling_image_size: tuple[int, int] = (512, 512),
        min_vertices: int = 3,
        max_vertices: int = 7,
        mismatched_numbers: bool = False,
        allow_nonuniform_vertices: bool = False,
        object_variant: Literal["stars", "polygons"] = "stars",
        star_radius_ratio: float = 0.1,
        hsv_saturation: float = 1.0,
        hsv_value: float = 0.9,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        # Set root if not provided
        if root is None:
            root = ""  # Will be set by subdataset if needed

        super().__init__(
            stage=stage,
            root=root or "",
            image_resolution=image_resolution,
            labeler_name=labeler_name,
            are_nums_on_images=are_nums_on_images,
            supersampling_image_size=supersampling_image_size,
            min_vertices=min_vertices,
            max_vertices=max_vertices,
            mismatched_numbers=mismatched_numbers,
            allow_nonuniform_vertices=allow_nonuniform_vertices,
            object_variant=object_variant,
            star_radius_ratio=star_radius_ratio,
            cache_dir=cache_dir,
            **kwargs,
        )

        self.subdataset = subdataset
        self.hsv_saturation = hsv_saturation
        self.hsv_value = hsv_value

        self.subdataset_image_resize = Compose(
            [
                Lambda(
                    lambda pil_image: self.relative_resize(
                        pil_image, self.image_resolution
                    )
                ),
                CenterCrop(self.image_resolution),
            ]
        )

    @staticmethod
    def relative_resize(
        image: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        width, height = image.size
        target_width, target_height = target_size

        # Calculate scaling factor
        scale = max(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), resample=Image.BICUBIC)

    @property
    @lru_cache(maxsize=None)
    def blur_kernel(self) -> torch.Tensor:
        kernel_size = 255  # 256 possible values
        sigma = self.color_histogram_blur_sigma
        blur_kernel = torch.exp(
            -((torch.arange(kernel_size) - kernel_size // 2) ** 2) / (2 * sigma**2)
        )
        blur_kernel /= blur_kernel.sum()
        return blur_kernel

    def _get_color(
        self, rng: np.random.Generator | None, base_image: Image.Image
    ) -> str | tuple[int, int, int]:
        h, _, _ = base_image.convert("HSV").split()
        hue_histogram = torch.tensor(h.histogram(), dtype=torch.float32)
        padding = self.color_histogram_blur_kernel_size // 2
        padded_histogram = F.pad(
            hue_histogram[None, None, :], (padding, padding), mode="circular"
        )
        histogram = F.conv1d(padded_histogram, self.blur_kernel[None, None, :])
        hue = torch.argmin(histogram[0, 0]).item() / 255

        value = self.hsv_value
        saturation = self.hsv_saturation
        rgb = hsv_to_rgb(hue, saturation, value)
        rgb = (
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255),
        )  # Scale to 0-255
        return rgb

    def _split_idx(self, idx: int) -> tuple[int, int, int]:
        """Loop over both datasets, and loops the smaller one"""
        subdataset_idx = idx % self.subdataset._num_available
        num_circles_idx, circles_image_idx = self.split_circles_idx(
            idx % self._num_overlay_images
        )
        return num_circles_idx, circles_image_idx, subdataset_idx

    def _get_base_image(self, subdataset_idx: int) -> Image.Image:
        image_data = self.subdataset[subdataset_idx]
        if isinstance(image_data, tuple):
            image = image_data[0]  # Get image from tuple
        else:
            image = image_data
        return self.subdataset_image_resize(image).convert("RGBA")

    @property
    def _num_available(self) -> int:
        return max(self._num_overlay_images, self.subdataset._num_available)

"""
Counting Polygons FFHQ Dataset.
Combines FFHQ background images with counting polygons overlay.
"""

from typing import Literal

from .counting_objects_subdataset import CountingObjectsSubdataset
from .ffhq import FFHQDataset


class CountingObjectsFFHQ(CountingObjectsSubdataset):
    """
    Counting Polygons dataset with FFHQ background images.
    """

    def __init__(
        self,
        stage: str = "train",
        root: str | None = None,
        ffhq_root: str | None = None,
        image_resolution: tuple[int, int] = (256, 256),
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
        # Create FFHQ subdataset
        ffhq_dataset = FFHQDataset(
            stage=stage,
            root=ffhq_root or root,
            image_resolution=image_resolution,
            cache_dir=cache_dir,
        )

        # Initialize subdataset with FFHQ
        super().__init__(
            subdataset=ffhq_dataset,
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
            hsv_saturation=hsv_saturation,
            hsv_value=hsv_value,
            **kwargs,
        )

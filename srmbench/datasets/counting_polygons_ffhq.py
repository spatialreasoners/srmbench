"""
Counting Polygons FFHQ Dataset.
Combines FFHQ background images with counting polygons overlay.
"""

from .counting_polygons_subdataset import CountingPolygonsSubdataset
from .ffhq import FFHQDataset


class CountingPolygonsFFHQ(CountingPolygonsSubdataset):
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
        are_nums_on_images: bool = False,
        supersampling_image_size: tuple[int, int] = (512, 512),
        min_vertices: int = 3,
        max_vertices: int = 7,
        font_name: str = "Roboto-Regular.ttf",
        mismatched_numbers: bool = False,
        allow_nonuniform_vertices: bool = False,
        use_stars: bool = False,
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
            font_name=font_name,
            mismatched_numbers=mismatched_numbers,
            allow_nonuniform_vertices=allow_nonuniform_vertices,
            use_stars=use_stars,
            star_radius_ratio=star_radius_ratio,
            hsv_saturation=hsv_saturation,
            hsv_value=hsv_value,
            **kwargs,
        )

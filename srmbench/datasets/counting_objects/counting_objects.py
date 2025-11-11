import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int
from PIL import Image, ImageDraw

from ..srm_dataset import SRMDataset
from .font_cache import FontCache
from .labelers import get_labeler


class CountingObjectsBase(SRMDataset, ABC):
    """
    Base class for counting polygons datasets.
    """

    training_set_ratio: float = 0.95
    circle_images_per_num_circles: int = 100_000
    circle_num_variants: int = 9
    circle_positions_file_name: str = "circle_position_radius.npy"
    artifacts_repository: str = "BartekPog/counting-objects-artifacts"
    font_file_name: str = "Roboto-Regular.ttf"

    def __init__(
        self,
        stage: Literal["train", "test"] = "train",
        root: str | Path = "",
        image_resolution: Sequence[int] = (128, 128),
        labeler_name: Literal["explicit", "ambiguous"] | None = None,
        are_nums_on_images: bool = True,
        supersampling_image_size: Sequence[int] = (512, 512),
        min_vertices: int = 3,
        max_vertices: int = 7,
        mismatched_numbers: bool = False,
        allow_nonuniform_vertices: bool = False,
        object_variant: Literal["stars", "polygons"] = "stars",
        star_radius_ratio: float = 0.1,
        cache_dir: str | None = None,
        transform=None,
    ) -> None:
        super().__init__(stage)

        if allow_nonuniform_vertices:
            assert mismatched_numbers, "Mismatched numbers must be enabled to allow nonuniform vertices"

        self.root_path = Path(root) if root else None
        self.image_resolution = tuple(image_resolution)
        self.labeler_name = labeler_name
        self.are_nums_on_images = are_nums_on_images
        self.supersampling_image_size = tuple(supersampling_image_size)
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.mismatched_numbers = mismatched_numbers
        self.allow_nonuniform_vertices = allow_nonuniform_vertices
        self.object_variant = object_variant
        self.star_radius_ratio = star_radius_ratio
        self._cache_dir = cache_dir
        self.transform = transform

        self.min_circle_num = 3 if self.are_nums_on_images else 1
        self.circle_positions = self._load_circle_positions()

        # Download font from HuggingFace
        font_path = None
        try:
            font_path = self._download_artifact(
                self.font_file_name, cache_dir=self._cache_dir
            )
        except Exception as e:
            print(f"Warning: Could not load font from HuggingFace: {e}")
            print("Font rendering will use default font.")

        if font_path and font_path.exists():
            self.font_cache = FontCache(font_path)
        else:
            # Fallback: no font cache
            self.font_cache = None

        self.labeler = (
            get_labeler(
                labeler_name=self.labeler_name,
                min_vertices=self.min_vertices,
                max_vertices=self.max_vertices,
            )
            if self.labeler_name
            else None
        )

        assert self.supersampling_image_size[0] >= self.image_resolution[0]
        assert self.supersampling_image_size[1] >= self.image_resolution[1]

        self.circle_xyr_scaling_factor = np.array(
            [
                self.supersampling_image_size[0],  # x
                self.supersampling_image_size[1],  # y
                min(self.supersampling_image_size),  # radius
            ]
        )[np.newaxis, ...]  # shape: (1, 3)

        self._draw_object = self._draw_stars if self.object_variant == "stars" else self._draw_polygons

    @abstractmethod
    def _get_base_image(self, base_image_idx: int) -> Image.Image:
        pass

    @property
    def _num_available(self) -> int:
        """Number of available samples. Override in subclasses if needed."""
        return self._num_overlay_images

    @abstractmethod
    def _split_idx(self, idx: int) -> tuple[int, int, int | None]:
        pass

    # Cache for downloaded artifacts (module-level cache)
    _artifact_cache: dict[tuple[str, str | None], Path] = {}

    @staticmethod
    def _download_artifact(filename: str, cache_dir: str | None = None) -> Path:
        """
        Download an artifact file from HuggingFace dataset.
        
        Args:
            filename: Name of the file to download
            cache_dir: Optional cache directory for HuggingFace downloads
            
        Returns:
            Path to the downloaded file
        """
        # Check cache first
        cache_key = (filename, cache_dir)
        if cache_key in CountingObjectsBase._artifact_cache:
            return CountingObjectsBase._artifact_cache[cache_key]
        
        # Download from HuggingFace (HuggingFace also has its own caching)
        file_path = hf_hub_download(
            repo_id=CountingObjectsBase.artifacts_repository,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        result = Path(file_path)
        
        # Cache the result
        CountingObjectsBase._artifact_cache[cache_key] = result
        return result

    @property
    def _training_positions_per_circles_num(self) -> int:
        return int(self.training_set_ratio * self.circle_images_per_num_circles)

    def _load_circle_positions(
        self,
    ) -> dict[int, Float[np.ndarray, "num_images num_circles_per_image 3"]]:
        # The data is in the form of a numpy array with shape (num_circles, num_circles, 3)
        # where the last dimension is the x, y, and radius of the circle
        circle_pos_file = self._download_artifact(
            self.circle_positions_file_name, cache_dir=self._cache_dir
        )
        circle_pos_radius = np.load(circle_pos_file, allow_pickle=True).item()

        possible_num_circle_nums = np.arange(
            self.min_circle_num, self.min_circle_num + self.circle_num_variants
        )  # 1-9 or 3-11

        return {
            num_circles: (
                data[: self._training_positions_per_circles_num]
                if self.stage == "train"
                else data[self._training_positions_per_circles_num :]
            )
            for num_circles, data in circle_pos_radius.items()
            if num_circles in possible_num_circle_nums
        }

    def split_circles_idx(self, idx: int) -> tuple[int, int]:
        num_circles_idx = (
            idx // self._num_positions_per_num_circles + self.min_circle_num
        )
        circles_image_idx = idx % self._num_positions_per_num_circles
        assert circles_image_idx < self._num_overlay_images
        return num_circles_idx, circles_image_idx

    def _get_circle_xyr(
        self, num_circles_idx: int, circles_image_idx: int
    ) -> Float[np.ndarray, "num_circles_per_image 3"]:
        image_circles = self.circle_positions[num_circles_idx][
            circles_image_idx
        ]  # shape: (num_circles, 3)
        return image_circles * self.circle_xyr_scaling_factor

    @abstractmethod
    def _get_color(
        self, rng: np.random.Generator | None, base_image: Image.Image
    ) -> str | tuple[int, int, int]:
        pass

    @staticmethod
    def _get_unit_polygon_vertices(
        points_on_circle: int | np.int64, angle_offset: Float[np.ndarray, "num_polygons"]
    ) -> Float[np.ndarray, "num_polygons points_on_circle 2"]:
        base = np.arange(points_on_circle) / points_on_circle * 2 * np.pi
        angles = base[np.newaxis, ...] + np.expand_dims(angle_offset, 1)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack([x, y], axis=-1)

    @staticmethod
    def random_choice(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.choice(*args, **kwargs)
        return np.random.choice(*args, **kwargs)

    @staticmethod
    def random_integers(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.integers(*args, **kwargs).item()
        return np.random.randint(*args, **kwargs)

    @staticmethod
    def random_uniform(rng: np.random.Generator | None, *args, **kwargs):
        if rng is not None:
            return rng.uniform(*args, **kwargs)
        return np.random.uniform(*args, **kwargs)

    def _draw_polygons(
        self,
        draw: ImageDraw.ImageDraw,
        vertices_per_object: Int[np.ndarray, "num_polygons"],
        polygons_xyr: Float[np.ndarray, "num_polygons 3"],
        angle_offset: Float[np.ndarray, "num_polygons"],
        color: str | tuple[int, int, int],
    ) -> None:
        drawn_elements = 0
        for vertices_num in np.unique(vertices_per_object):
            vertices_mask = vertices_per_object == vertices_num
            group_polygons_xyr = polygons_xyr[vertices_mask]
            group_angle_offset = angle_offset[vertices_mask]
            batch_polygons_vertices = self._get_unit_polygon_vertices(
                vertices_num, group_angle_offset
            )
            radii = group_polygons_xyr[:, 2]
            centers = group_polygons_xyr[:, :2]
            batch_polygons_vertices *= radii[:, np.newaxis, np.newaxis]  # scale by corresponding radius
            batch_polygons_vertices += np.expand_dims(centers, 1)  # translate to corresponding center

            for polygon_vertices in batch_polygons_vertices:
                polygon_vertices = [(x, y) for x, y in polygon_vertices]  # convert to list of tuples
                draw.polygon(polygon_vertices, fill=color)
                drawn_elements += 1

        assert drawn_elements == len(polygons_xyr), "Not all polygons were drawn"

    def _draw_stars(
        self,
        draw: ImageDraw.ImageDraw,
        vertices_per_object: Int[np.ndarray, "num_polygons"],
        polygons_xyr: Float[np.ndarray, "num_polygons 3"],
        angle_offset: Float[np.ndarray, "num_polygons"],
        color: str | tuple[int, int, int],
    ) -> None:
        for vertex_num, xyr, ang_offset in zip(vertices_per_object, polygons_xyr, angle_offset):
            center = xyr[:2]
            outer_radius = xyr[2]
            inner_radius = self.star_radius_ratio * outer_radius
            self.draw_star(
                draw=draw,
                center=center,
                outer_radius=outer_radius.item(),
                inner_radius=inner_radius.item(),
                angle_offset=ang_offset.item(),
                points=vertex_num.item(),
                color=color,
            )

    @staticmethod
    def draw_star(
        draw: ImageDraw.ImageDraw,
        center: Float[np.ndarray, "2"],
        outer_radius: float,
        inner_radius: float,
        angle_offset: float,
        points: int,
        color: str | tuple[int, int, int] = "black",
    ):
        """
        Draws a star using PIL.ImageDraw.

        :param draw: The ImageDraw object
        :param center: Tuple (x, y) for the center of the star
        :param outer_radius: Radius of the outer points of the star
        :param inner_radius: Radius of the inner points of the star
        :param angle_offset: Rotation offset in radians
        :param points: Number of star points
        :param color: Fill color of the star
        """
        star_points = []
        angle = math.pi / points  # Angle between outer and inner points
        for i in range(2 * points):  # Loop through outer and inner points
            r = outer_radius if i % 2 == 0 else inner_radius
            theta = i * angle + angle_offset  # Current angle
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            star_points.append((x, y))
        draw.polygon(star_points, fill=color)

    def _draw_numbers(
        self,
        draw: ImageDraw.ImageDraw,
        numbers_xyr: Float[np.ndarray, "2 3"],
        numbers: Int[np.ndarray, "2"],
        color: str | tuple[int, int, int],
    ) -> None:
        if self.font_cache is None:
            # Fallback: use default font if font_cache not available
            for (x, y, radius), number in zip(numbers_xyr, numbers):
                draw.text((x, y), str(number), fill=color)
            return

        for (x, y, radius), number in zip(numbers_xyr, numbers):
            font = self.font_cache.get_font(int(radius))
            draw.text((x, y), str(number), font=font, fill=color)

    def _get_image_with_objects(
        self,
        vertices_per_polygon: Int[np.ndarray, "num_polygons"],
        polygons_xyr: Float[np.ndarray, "num_polygons 3"],
        angle_offset: Float[np.ndarray, "num_polygons"],
        color: str | tuple[int, int, int],
        numbers_xyr: Float[np.ndarray, "2 3"] | None,
        numbers: Int[np.ndarray, "2"] | None,
    ) -> Image.Image:
        image = Image.new("RGBA", self.supersampling_image_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        self._draw_object(draw, vertices_per_polygon, polygons_xyr, angle_offset, color)
        if numbers_xyr is not None:
            self._draw_numbers(draw, numbers_xyr, numbers, color)

        resized_image = image.resize(self.image_resolution, resample=Image.BICUBIC)
        return resized_image

    def _get_overlay_image_w_label(
        self,
        num_circles_idx: int,
        circles_image_idx: int,
        full_idx: int,
        base_image: Image.Image,
    ) -> tuple[Image.Image, int | dict | None]:
        circle_xyr = self._get_circle_xyr(num_circles_idx, circles_image_idx)
        num_circles = circle_xyr.shape[0]

        # Randomize the order of the circles
        rng = None
        if self.is_deterministic:
            rng = np.random.default_rng(full_idx)
        else:
            circle_xyr = circle_xyr[np.random.permutation(num_circles)]

        if self.are_nums_on_images:
            numbers_xyr = circle_xyr[-2:]
            polygons_xyr = circle_xyr[:-2]
            num_polygons = num_circles - 2
        else:
            numbers_xyr = None
            polygons_xyr = circle_xyr
            num_polygons = num_circles

        if all(
            [
                self.allow_nonuniform_vertices,
                (num_polygons > 1),
                self.random_choice(rng, [True, False]),
            ]
        ):
            vertices_per_polygon = self.random_choice(
                rng,
                np.arange(self.min_vertices, self.max_vertices + 1),
                num_polygons,
                replace=True,
            )
        else:
            # set the number of vertices to num_vertices
            num_vertices = self.random_integers(
                rng, self.min_vertices, self.max_vertices + 1
            )
            vertices_per_polygon = np.full(num_polygons, num_vertices)

        are_uniform_vertices = (vertices_per_polygon[0] == vertices_per_polygon).all()

        if self.mismatched_numbers:
            # Randomize the numbers that will be drawn on the image (and in the label)
            conditioning_numbers = {
                "num_polygons": self.random_integers(rng, 1, 10),
                "num_vertices": self.random_integers(
                    rng, self.min_vertices, self.max_vertices + 1
                ),
            }
        else:
            conditioning_numbers = {
                "num_polygons": num_polygons,
                "num_vertices": vertices_per_polygon[0].item(),
            }

        numbers_label = None  # Default value
        if self.labeler is not None:
            numbers_label = self.labeler.get_label(**conditioning_numbers)

        if self.mismatched_numbers:
            label = {
                "num_polygons": num_polygons - 1,  # map 1-9 to 0-8 as class labels
                "num_vertices": int(  # map 3-7 to 0-4 as class labels
                    vertices_per_polygon[0] - self.min_vertices
                ),
                "is_uniform": int(are_uniform_vertices),
            }
            if numbers_label is not None:
                label["numbers_label"] = numbers_label
        else:
            label = numbers_label

        angle_offset = self.random_uniform(rng, 0, 2 * np.pi, size=num_polygons)
        color = self._get_color(rng, base_image)
        resized_image = self._get_image_with_objects(
            vertices_per_polygon=vertices_per_polygon,
            polygons_xyr=polygons_xyr,
            angle_offset=angle_offset,
            color=color,
            numbers_xyr=numbers_xyr,
            numbers=np.fromiter(conditioning_numbers.values(), dtype=int),
        )

        return resized_image, label

    @property
    def num_classes(self) -> int | dict[str, int]:
        if not self.mismatched_numbers:
            return self.labeler.num_classes if self.labeler else 0
        class_counts = {
            "num_polygons": self.circle_num_variants,
            "num_vertices": self.max_vertices - self.min_vertices + 1,
        }
        if self.labeler:
            class_counts["numbers_label"] = self.labeler.num_classes
        if self.allow_nonuniform_vertices:
            class_counts["is_uniform"] = 2
        return class_counts

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image] | Image.Image:
        split_result = self._split_idx(idx)
        num_circles_idx, circles_image_idx, base_image_idx = split_result
        if base_image_idx is None:
            # For non-subdataset classes, base_image_idx might be None
            # Create a white base image
            base_image = Image.new("RGBA", self.supersampling_image_size, (255, 255, 255, 255))
        else:
            base_image = self._get_base_image(base_image_idx)
        overlay_image, label = self._get_overlay_image_w_label(
            num_circles_idx, circles_image_idx, full_idx=idx, base_image=base_image
        )
        image = Image.alpha_composite(base_image, overlay_image)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Return image only (SRMDataset allows tuple or Image)
        return image

    @property
    def _num_positions_per_num_circles(self) -> int:
        return (
            self._training_positions_per_circles_num
            if self.stage == "train"
            else self.circle_images_per_num_circles - self._training_positions_per_circles_num
        )

    @property
    def _num_overlay_images(self) -> int:
        """Calculate the number of overlay images based on the number of circle xyrs"""
        return self._num_positions_per_num_circles * self.circle_num_variants

    def __len__(self) -> int:
        return self._num_available

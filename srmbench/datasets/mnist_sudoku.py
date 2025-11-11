from math import prod
from typing import Literal

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Integer
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor

from .srm_dataset import SRMDataset


class MnistSudokuDataset(SRMDataset):
    dataset_repository = "BartekPog/mnist-sudoku"
    dataset_file = "sudoku-mnist.safetensors"

    grid_size = (9, 9)
    cell_size = (28, 28)

    def __init__(
        self,
        stage: Literal["train", "test"] = "train",
        min_given_cells: int = 0,  # 0 means no given cells
        max_given_cells: int = 80,  # 80 means all but one cell is given
        transform=None,
        mask_transform=None,
    ) -> None:
        super().__init__(stage)

        if min_given_cells > max_given_cells:
            raise ValueError(
                "min_given_cells must be less than or equal to max_given_cells"
            )
        if min_given_cells < 0:
            raise ValueError("min_given_cells must be greater than or equal to 0")
        if max_given_cells > prod(self.grid_size):
            raise ValueError(
                "max_given_cells must be less than or equal to the number of cells in the grid"
            )

        self.min_given_cells = min_given_cells
        self.max_given_cells = max_given_cells
        self.transform = transform
        self.mask_transform = mask_transform

        self.mnist_images, self.sudoku_grids = self._load_sudoku_data()

        self.image_shape = (
            self.cell_size[0] * self.grid_size[0],
            self.cell_size[1] * self.grid_size[1],
        )

    def _get_dataset_files_path(self) -> str:
        return hf_hub_download(
            repo_id=self.dataset_repository,
            filename=self.dataset_file,
            repo_type="dataset",
        )

    def _load_sudoku_data(
        self,
    ) -> tuple[
        Integer[Tensor, "10 num_digits 28 28"], Integer[Tensor, "num_sudokus 9 9"]
    ]:
        safetensor = load_file(self._get_dataset_files_path())
        mnist_images = safetensor["mnist_images"]
        sudoku_grids = safetensor["sudokus"]

        return mnist_images, sudoku_grids

    def _load_full_image(self, rng: np.random.Generator | None = None) -> Image.Image:
        randint_func = rng.integers if rng is not None else np.random.randint

        grid = self.sudoku_grids[randint_func(0, len(self.sudoku_grids))]

        full_image = torch.empty((252, 252), dtype=torch.uint8)

        for j in range(9):
            for k in range(9):
                # Get the corresponding MNIST number
                candidates = self.mnist_images[int(grid[j, k])]

                # Randomly select one of the MNIST numbers
                if rng is not None:
                    index = rng.integers(0, candidates.size(0))
                else:
                    index = np.random.randint(0, candidates.size(0))

                mnist_image = candidates[index]

                # Add the MNIST tensor to the grid of MNIST numbers
                full_image[j * 28 : (j + 1) * 28, k * 28 : (k + 1) * 28] = mnist_image

        return Image.fromarray(full_image.numpy())

    def _get_random_masks(
        self, rng: np.random.Generator | None = None
    ) -> Float[np.ndarray, "height width"]:
        num_cells = prod(self.grid_size)
        num_given_cells = np.random.randint(
            self.min_given_cells, self.max_given_cells + 1
        )
        mask = np.ones(self.grid_size, dtype=bool)

        grid_idx = (np.random if rng is None else rng).choice(
            num_cells, num_given_cells, replace=False
        )
        row, col = grid_idx // self.grid_size[1], grid_idx % self.grid_size[1]
        mask[row, col] = False
        mask = np.kron(mask, np.ones(self.cell_size, dtype=bool))
        return mask.astype(float)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        rng = np.random.default_rng(idx) if self.is_deterministic else None

        image = self._load_full_image(rng)
        mask = self._get_random_masks(rng)
        mask_image = Image.fromarray(mask.astype(np.uint8))

        if self.transform is not None:
            image = self.transform(image)
        
        if self.mask_transform is not None:
            mask_image = self.mask_transform(mask_image)

        return image, mask_image

    def __len__(self) -> int:
        return len(self.sudoku_grids)

"""
Tests for dataset modules.
"""

import pytest
import torch
from PIL import Image

from srmbench.datasets.mnist_sudoku import MnistSudokuDataset
from srmbench.datasets.counting_objects import CountingObjectsBase, CountingObjectsFFHQ


class TestMnistSudokuDataset:
    """Test cases for MnistSudokuDataset."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = MnistSudokuDataset()
        assert len(dataset) > 0

    def test_dataset_getitem(self):
        """Test dataset item access."""
        dataset = MnistSudokuDataset()
        sample = dataset[0]

        assert isinstance(sample[0], Image.Image)
        assert isinstance(sample[1], Image.Image)
        
        assert sample[0].size == (252, 252)
        assert sample[1].size == (252, 252)
        
        assert sample[0].mode == "L"
        assert sample[1].mode == "L"

    def test_dataset_deterministic(self):
        """Test dataset deterministic property."""
        train_dataset = MnistSudokuDataset(stage="train")
        test_dataset = MnistSudokuDataset(stage="test")

        assert not train_dataset.is_deterministic
        assert test_dataset.is_deterministic


class TestCountingObjectsDataset:
    """Test cases for CountingObjectsBase datasets (polygons and stars)."""

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_dataset_creation(self, object_variant):
        """Test dataset creation for both polygons and stars."""
        # Note: CountingObjectsBase is abstract, so we test via CountingObjectsFFHQ
        # which is a concrete implementation
        dataset = CountingObjectsFFHQ(
            stage="test",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
        )
        assert len(dataset) > 0

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_dataset_getitem(self, object_variant):
        """Test dataset item access for both variants."""
        dataset = CountingObjectsFFHQ(
            stage="test",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
        )
        sample = dataset[0]

        assert isinstance(sample, Image.Image)
        assert sample.size == (128, 128)
        assert sample.mode == "RGB"

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_dataset_deterministic(self, object_variant):
        """Test dataset deterministic property."""
        train_dataset = CountingObjectsFFHQ(
            stage="train",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
        )
        test_dataset = CountingObjectsFFHQ(
            stage="test",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
        )

        assert not train_dataset.is_deterministic
        assert test_dataset.is_deterministic

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_dataset_with_numbers_on_images(self, object_variant):
        """Test that dataset can generate images with numbers."""
        dataset = CountingObjectsFFHQ(
            stage="test",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
        )
        sample = dataset[0]
        assert isinstance(sample, Image.Image)
        assert sample.size == (128, 128)

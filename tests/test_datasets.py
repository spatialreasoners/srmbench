"""
Tests for dataset modules.
"""

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms

from srmbench.datasets.mnist_sudoku import MnistSudokuDataset
from srmbench.datasets.counting_objects import CountingObjectsBase, CountingObjectsFFHQ
from srmbench.datasets.even_pixels import EvenPixelsDataset


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


class TestDatasetTransforms:
    """Test that datasets work correctly with transforms."""

    def test_even_pixels_with_transform(self):
        """Test EvenPixelsDataset with transform."""
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        dataset = EvenPixelsDataset(stage="test", transform=transform)
        sample = dataset[0]
        
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32
        assert sample.shape == (3, 32, 32)

    def test_mnist_sudoku_with_transform(self):
        """Test MnistSudokuDataset with transform."""
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        dataset = MnistSudokuDataset(stage="test", transform=transform)
        image, mask = dataset[0]
        
        # Image should be transformed
        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.float32
        # Mask should remain a PIL Image
        assert isinstance(mask, Image.Image)
    
    def test_mnist_sudoku_with_mask_transform(self):
        """Test MnistSudokuDataset with both image and mask transforms."""
        image_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        mask_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        dataset = MnistSudokuDataset(
            stage="test",
            transform=image_transform,
            mask_transform=mask_transform
        )
        image, mask = dataset[0]
        
        # Both should be transformed to tensors
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.dtype == torch.float32
        assert mask.dtype == torch.float32
        # Check shapes (252x252 grayscale images)
        assert image.shape == (1, 252, 252)
        assert mask.shape == (1, 252, 252)

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_counting_objects_with_transform(self, object_variant):
        """Test CountingObjectsFFHQ with transform."""
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        dataset = CountingObjectsFFHQ(
            stage="test",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
            transform=transform,
        )
        sample = dataset[0]
        
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32
        assert sample.shape == (3, 128, 128)

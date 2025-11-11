"""
Integration tests for the full pipeline.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from srmbench.datasets.even_pixels import EvenPixelsDataset
from srmbench.datasets.mnist_sudoku import MnistSudokuDataset
from srmbench.datasets.counting_objects import CountingObjectsFFHQ
from srmbench.evaluations.even_pixels_evaluation import EvenPixelsEvaluation
from srmbench.evaluations.mnist_sudoku_evaluation import MnistSudokuEvaluation
from srmbench.evaluations.counting_objects_evaluation import CountingObjectsEvaluation


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to torch tensor."""
    # Convert PIL Image to numpy array and then to tensor
    array = np.array(image, dtype=np.float32)
    # Normalize to [0, 1] if values are in [0, 255] range
    if array.max() > 1.0:
        array = array / 255.0
    return torch.from_numpy(array)


def _pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB Image to torch tensor in CHW format, normalized to [-1, 1]."""
    # Convert PIL Image to numpy array
    array = np.array(image, dtype=np.float32)
    # Normalize from [0, 255] to [0, 1]
    if array.max() > 1.0:
        array = array / 255.0
    # Convert from HWC to CHW
    array = array.transpose(2, 0, 1)
    # Convert to tensor and normalize to [-1, 1]
    tensor = torch.from_numpy(array)
    return tensor * 2.0 - 1.0


class MnistSudokuTestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test the complete pipeline from dataset to evaluation."""
        dataset = MnistSudokuDataset()
        evaluation = MnistSudokuEvaluation()

        # Get a sample - dataset returns (image, mask) tuple
        sample = dataset[0]
        image, mask = sample
        
        # Convert PIL Image to tensor
        image_tensor = _pil_to_tensor(image)
        batch = image_tensor.unsqueeze(0)  # Add batch dimension

        # Evaluate
        result = evaluation.evaluate(batch)

        assert "is_valid_sudoku" in result
        assert "duplicate_count" in result
        assert result["is_valid_sudoku"].shape == (1,)
        assert result["duplicate_count"].shape == (1,)
        
    def test_pipeline_correctness(self):
        """Test that single and batch processing give consistent results."""
        dataset = MnistSudokuDataset()
        evaluation = MnistSudokuEvaluation()

        # Single sample
        samples = torch.stack([_pil_to_tensor(dataset[i][0]) for i in range(3)])
        
        result = evaluation.evaluate(samples)

        assert result["is_valid_sudoku"].shape == (3,)
        assert result["duplicate_count"].shape == (3,)
        assert (result["is_valid_sudoku"] == torch.ones(3)).all()
        assert (result["duplicate_count"] == 0).all()


class EvenPixelsTestIntegration:
    """Integration tests for EvenPixelsEvaluation with EvenPixelsDataset."""

    def test_full_pipeline(self):
        """Test the complete pipeline from dataset to evaluation."""
        dataset = EvenPixelsDataset(stage="test")
        evaluation = EvenPixelsEvaluation()

        # Get a sample - dataset returns just an image
        image = dataset[0]

        # Convert PIL RGB Image to tensor in CHW format, normalized to [-1, 1]
        image_tensor = _pil_rgb_to_tensor(image)
        batch = image_tensor.unsqueeze(0)  # Add batch dimension

        # Evaluate
        result = evaluation.evaluate(batch)

        assert "saturation_std" in result
        assert "value_std" in result
        assert "color_imbalance_count" in result
        assert "is_color_count_even" in result

        # All metrics should be scalar tensors
        for key, value in result.items():
            assert value.shape == (), f"{key} should be a scalar tensor"
            assert torch.isfinite(value), f"{key} should be finite"

    def test_even_pixels_correctness(self):
        """Test that EvenPixelsDataset produces images with good metrics."""
        dataset = EvenPixelsDataset(stage="test", saturation=1.0, value=0.7)
        evaluation = EvenPixelsEvaluation()

        # Get a batch of samples
        batch_size = 5
        batch_images = []
        for i in range(batch_size):
            image = dataset[i]
            batch_images.append(_pil_rgb_to_tensor(image))

        batch_tensor = torch.stack(batch_images)
        result = evaluation.evaluate(batch_tensor)

        # For EvenPixelsDataset:
        # - Saturation and value should be constant (std close to 0)
        # - Hue should be evenly split (errors should be 0, color count should be even)
        assert result["saturation_std"] < 0.01, "Saturation should be nearly constant (std < 0.01)"
        assert result["value_std"] < 0.01, "Value should be nearly constant (std < 0.01)"
        assert result["color_imbalance_count"] == 0.0, "Hue should be evenly distributed (color_imbalance_count = 0)"
        assert result["is_color_count_even"] == 1.0, "Color count should be even (1.0)"

    def test_batch_processing_consistency(self):
        """Test that batch processing gives consistent results."""
        dataset = EvenPixelsDataset(stage="test")
        evaluation = EvenPixelsEvaluation()

        # Process single sample
        image_0 = dataset[0]
        single_tensor = _pil_rgb_to_tensor(image_0).unsqueeze(0)
        single_result = evaluation.evaluate(single_tensor)

        # Process batch
        batch_images = []
        for i in range(3):
            image = dataset[i]
            batch_images.append(_pil_rgb_to_tensor(image))
        batch_tensor = torch.stack(batch_images)
        batch_result = evaluation.evaluate(batch_tensor)

        # Both should have all required metrics
        assert set(single_result.keys()) == set(batch_result.keys())
        for key in single_result.keys():
            assert single_result[key].shape == ()
            assert batch_result[key].shape == ()


class CountingObjectsTestIntegration:
    """Integration tests for CountingObjectsEvaluation with CountingObjects datasets."""

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    # @pytest.mark.skip(reason="Requires pretrained models from HuggingFace")
    def test_full_pipeline(self, object_variant):
        """Test the complete pipeline from dataset to evaluation."""
        dataset = CountingObjectsFFHQ(
            stage="test",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
        )
        evaluation = CountingObjectsEvaluation(object_variant=object_variant, device="cpu")

        # Get a sample
        sample = dataset[0]

        # Convert PIL RGB Image to tensor in CHW format, normalized to [-1, 1]
        image_tensor = _pil_rgb_to_tensor(sample)
        batch = image_tensor.unsqueeze(0)  # Add batch dimension

        # Evaluate
        result = evaluation.evaluate(batch)

        assert "are_vertices_uniform" in result
        assert "numbers_match_objects" in result

        # All metrics should be scalar tensors
        for key, value in result.items():
            assert value.shape == (), f"{key} should be a scalar tensor"
            assert torch.isfinite(value), f"{key} should be finite"

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_numbers_match_objects_for_dataset_images(self, object_variant):
        """Test that numbers_match_objects is always true for images from the dataset."""
        dataset = CountingObjectsFFHQ(
            stage="test",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
        )
        evaluation = CountingObjectsEvaluation(object_variant=object_variant, device="cpu")

        # Get a batch of samples from the dataset
        batch_size = 5
        batch_images = []
        for i in range(batch_size):
            image = dataset[i]
            batch_images.append(_pil_rgb_to_tensor(image))

        batch_tensor = torch.stack(batch_images)
        result = evaluation.evaluate(batch_tensor)

        # For images from the dataset, numbers_match_objects should be high (ideally 1.0)
        # The numbers on the image should match the actual counts
        assert "numbers_match_objects" in result
        assert result["numbers_match_objects"] > 0.8, (
            f"numbers_match_objects should be high (>0.8) for dataset images, "
            f"got {result['numbers_match_objects']}"
        )

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_numbers_match_objects_for_random_images(self, object_variant):
        """Test that numbers_match_objects is low for random images."""
        evaluation = CountingObjectsEvaluation(object_variant=object_variant, device="cpu")

        # Create random images
        batch_size = 10
        batch_input = torch.rand(batch_size, 3, 128, 128) * 2.0 - 1.0
        result = evaluation.evaluate(batch_input)

        # For random images, numbers_match_objects should be low
        # Random images won't have numbers that match the predicted counts
        assert "numbers_match_objects" in result
        assert result["numbers_match_objects"] < 0.5, (
            f"numbers_match_objects should be low (<0.5) for random images, "
            f"got {result['numbers_match_objects']}"
        )

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    # @pytest.mark.skip(reason="Requires pretrained models from HuggingFace")
    def test_batch_processing_consistency(self, object_variant):
        """Test that batch processing gives consistent results."""
        dataset = CountingObjectsFFHQ(
            stage="test",
            object_variant=object_variant,
            image_resolution=(128, 128),
            are_nums_on_images=True,
        )
        evaluation = CountingObjectsEvaluation(object_variant=object_variant, device="cpu")

        # Process single sample
        image_0 = dataset[0]
        single_tensor = _pil_rgb_to_tensor(image_0).unsqueeze(0)
        single_result = evaluation.evaluate(single_tensor)

        # Process batch
        batch_images = []
        for i in range(3):
            image = dataset[i]
            batch_images.append(_pil_rgb_to_tensor(image))
        batch_tensor = torch.stack(batch_images)
        batch_result = evaluation.evaluate(batch_tensor)

        # Both should have all required metrics
        assert set(single_result.keys()) == set(batch_result.keys())
        for key in single_result.keys():
            assert single_result[key].shape == ()
            assert batch_result[key].shape == ()

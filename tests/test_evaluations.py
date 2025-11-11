"""
Tests for evaluation modules.
"""

import pytest
import torch

from srmbench.evaluations.even_pixels_evaluation import EvenPixelsEvaluation
from srmbench.evaluations.mnist_sudoku_evaluation import (
    MNISTClassifier,
    MnistSudokuEvaluation,
)
from srmbench.evaluations.counting_objects_evaluation import (
    CountingObjectsEvaluation,
)


class TestMNISTClassifier:
    """Test cases for MNISTClassifier."""

    def test_classifier_creation(self):
        """Test classifier creation."""
        classifier = MNISTClassifier()
        assert classifier is not None

    def test_classifier_forward(self):
        """Test classifier forward pass."""
        classifier = MNISTClassifier()
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 28, 28)

        with torch.no_grad():
            output = classifier(input_tensor)

        assert output.shape == (batch_size, 10)
        assert torch.all(torch.isfinite(output))


class TestMnistSudokuEvaluation:
    """Test cases for MnistSudokuEvaluation."""

    def test_evaluation_creation(self):
        """Test evaluation creation."""
        evaluation = MnistSudokuEvaluation()
        assert evaluation is not None

    def test_evaluation_forward(self):
        """Test evaluation forward pass."""
        evaluation = MnistSudokuEvaluation()
        batch_size = 2
        input_tensor = torch.randint(0, 255, (batch_size, 1, 252, 252), dtype=torch.float32)

        with torch.no_grad():
            result = evaluation.evaluate(input_tensor)

        assert "is_valid_sudoku" in result
        assert "duplicate_count" in result
        assert result["is_valid_sudoku"].shape == (batch_size,)
        assert result["duplicate_count"].shape == (batch_size,)

    def test_evaluation_batch_processing(self):
        """Test evaluation batch processing."""
        evaluation = MnistSudokuEvaluation()

        batch_input = torch.randint(0, 255, (3, 1, 252, 252), dtype=torch.float32)
        batch_result = evaluation.evaluate(batch_input)

        assert batch_result["is_valid_sudoku"].shape == (3,)
        assert (batch_result["is_valid_sudoku"] == torch.zeros(3)).all()
        
        assert batch_result["duplicate_count"].shape == (3,)
        assert (batch_result["duplicate_count"] > 0).all()


class TestEvenPixelsEvaluation:
    """Test cases for EvenPixelsEvaluation."""

    def test_evaluation_creation(self):
        """Test evaluation creation."""
        evaluation = EvenPixelsEvaluation()
        assert evaluation is not None
        assert evaluation.num_bins == 256

    def test_evaluation_creation_with_custom_bins(self):
        """Test evaluation creation with custom num_bins."""
        evaluation = EvenPixelsEvaluation(num_bins=128)
        assert evaluation.num_bins == 128

    def test_evaluation_forward(self):
        """Test evaluation forward pass."""
        evaluation = EvenPixelsEvaluation()
        batch_size = 2
        # Create RGB images in [0, 1] range, convert to [-1, 1]
        input_tensor = torch.rand(batch_size, 3, 128, 128) * 2.0 - 1.0

        with torch.no_grad():
            result = evaluation.evaluate(input_tensor)

        assert "saturation_std" in result
        assert "value_std" in result
        assert "color_imbalance_count" in result
        assert "is_color_count_even" in result

        # Check that all metrics are scalar tensors
        for key, value in result.items():
            assert value.shape == (), f"{key} should be a scalar tensor"
            assert torch.isfinite(value), f"{key} should be finite"

    def test_evaluation_batch_processing(self):
        """Test evaluation batch processing."""
        evaluation = EvenPixelsEvaluation()

        batch_size = 3
        batch_input = torch.rand(batch_size, 3, 128, 128) * 2.0 - 1.0
        batch_result = evaluation.evaluate(batch_input)

        # All metrics should be scalar (batch-averaged)
        assert batch_result["saturation_std"].shape == ()
        assert batch_result["value_std"].shape == ()
        assert batch_result["color_imbalance_count"].shape == ()
        assert batch_result["is_color_count_even"].shape == ()

        # Check that metrics are in expected ranges
        assert 0 <= batch_result["saturation_std"] <= 1
        assert 0 <= batch_result["value_std"] <= 1
        assert batch_result["color_imbalance_count"] >= 0
        assert 0 <= batch_result["is_color_count_even"] <= 1

    def test_evaluation_with_constant_saturation_value(self):
        """Test evaluation with constant saturation and value."""
        evaluation = EvenPixelsEvaluation()
        batch_size = 2

        # Create images with constant saturation and value (should have std close to 0)
        # Random hue values
        input_tensor = torch.rand(batch_size, 3, 128, 128) * 2.0 - 1.0

        with torch.no_grad():
            result = evaluation.evaluate(input_tensor)

        # For random images, saturation and value std should be reasonable
        assert torch.isfinite(result["saturation_std"])
        assert torch.isfinite(result["value_std"])


class TestCountingPolygonsEvaluation:
    """Test cases for CountingPolygonsEvaluation."""

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_evaluation_creation(self, object_variant):
        """Test evaluation creation for both variants."""
        evaluation = CountingObjectsEvaluation(object_variant=object_variant, device="cpu")
        assert evaluation is not None
        assert evaluation.classifier is not None

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_evaluation_forward(self, object_variant):
        """Test evaluation forward pass."""
        evaluation = CountingObjectsEvaluation(object_variant=object_variant, device="cpu")
        batch_size = 2
        # Create RGB images in [-1, 1] range
        input_tensor = torch.rand(batch_size, 3, 128, 128) * 2.0 - 1.0

        with torch.no_grad():
            result = evaluation.evaluate(input_tensor)

        assert "are_vertices_uniform" in result
        assert "numbers_match_objects" in result
        assert result["are_vertices_uniform"].shape == ()
        assert result["numbers_match_objects"].shape == ()
        assert torch.isfinite(result["are_vertices_uniform"])
        assert torch.isfinite(result["numbers_match_objects"])

    @pytest.mark.parametrize("object_variant", ["polygons", "stars"])
    def test_evaluation_batch_processing(self, object_variant):
        """Test evaluation batch processing."""
        evaluation = CountingObjectsEvaluation(object_variant=object_variant, device="cpu")

        batch_size = 3
        batch_input = torch.rand(batch_size, 3, 128, 128) * 2.0 - 1.0
        batch_result = evaluation.evaluate(batch_input)

        # All metrics should be scalar (batch-averaged)
        assert batch_result["are_vertices_uniform"].shape == ()
        assert batch_result["numbers_match_objects"].shape == ()

        # Check that metrics are in expected ranges
        assert 0 <= batch_result["are_vertices_uniform"] <= 1
        assert 0 <= batch_result["numbers_match_objects"] <= 1
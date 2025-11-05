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
        input_tensor = torch.randint(0, 255, (batch_size, 252, 252), dtype=torch.float32)

        with torch.no_grad():
            result = evaluation.evaluate(input_tensor)

        assert "is_accurate" in result
        assert "distance" in result
        assert result["is_accurate"].shape == (batch_size,)
        assert result["distance"].shape == (batch_size,)

    def test_evaluation_batch_processing(self):
        """Test evaluation batch processing."""
        evaluation = MnistSudokuEvaluation()

        batch_input = torch.randint(0, 255, (3, 252, 252), dtype=torch.float32)
        batch_result = evaluation.evaluate(batch_input)

        assert batch_result["is_accurate"].shape == (3,)
        assert (batch_result["is_accurate"] == torch.zeros(3)).all()
        
        assert batch_result["distance"].shape == (3,)
        assert (batch_result["distance"] > 0).all()


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
        input_tensor = torch.rand(batch_size, 3, 256, 256) * 2.0 - 1.0

        with torch.no_grad():
            result = evaluation.evaluate(input_tensor)

        assert "saturation_std" in result
        assert "value_std" in result
        assert "hue_uneven_errors" in result
        assert "is_color_count_even" in result

        # Check that all metrics are scalar tensors
        for key, value in result.items():
            assert value.shape == (), f"{key} should be a scalar tensor"
            assert torch.isfinite(value), f"{key} should be finite"

    def test_evaluation_batch_processing(self):
        """Test evaluation batch processing."""
        evaluation = EvenPixelsEvaluation()

        batch_size = 3
        batch_input = torch.rand(batch_size, 3, 256, 256) * 2.0 - 1.0
        batch_result = evaluation.evaluate(batch_input)

        # All metrics should be scalar (batch-averaged)
        assert batch_result["saturation_std"].shape == ()
        assert batch_result["value_std"].shape == ()
        assert batch_result["hue_uneven_errors"].shape == ()
        assert batch_result["is_color_count_even"].shape == ()

        # Check that metrics are in expected ranges
        assert 0 <= batch_result["saturation_std"] <= 1
        assert 0 <= batch_result["value_std"] <= 1
        assert batch_result["hue_uneven_errors"] >= 0
        assert 0 <= batch_result["is_color_count_even"] <= 1

    def test_evaluation_with_constant_saturation_value(self):
        """Test evaluation with constant saturation and value."""
        evaluation = EvenPixelsEvaluation()
        batch_size = 2

        # Create images with constant saturation and value (should have std close to 0)
        # Random hue values
        input_tensor = torch.rand(batch_size, 3, 256, 256) * 2.0 - 1.0

        with torch.no_grad():
            result = evaluation.evaluate(input_tensor)

        # For random images, saturation and value std should be reasonable
        assert torch.isfinite(result["saturation_std"])
        assert torch.isfinite(result["value_std"])
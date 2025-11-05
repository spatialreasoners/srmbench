"""
Tests for evaluation modules.
"""

import pytest
import torch

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
"""
Integration tests for the full pipeline.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from srmbench.datasets.mnist_sudoku import MnistSudokuDataset
from srmbench.evaluations.mnist_sudoku_evaluation import MnistSudokuEvaluation


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to torch tensor."""
    # Convert PIL Image to numpy array and then to tensor
    array = np.array(image, dtype=np.float32)
    # Normalize to [0, 1] if values are in [0, 255] range
    if array.max() > 1.0:
        array = array / 255.0
    return torch.from_numpy(array)


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

        assert "is_accurate" in result
        assert "distance" in result
        assert result["is_accurate"].shape == (1,)
        assert result["distance"].shape == (1,)
        
    def test_pipeline_correctness(self):
        """Test that single and batch processing give consistent results."""
        dataset = MnistSudokuDataset()
        evaluation = MnistSudokuEvaluation()

        # Single sample
        samples = torch.stack([_pil_to_tensor(dataset[i][0]) for i in range(3)])
        
        result = evaluation.evaluate(samples)

        assert result["is_accurate"].shape == (3,)
        assert result["distance"].shape == (3,)
        assert (result["is_accurate"] == torch.ones(3)).all()
        assert (result["distance"] == 0).all()

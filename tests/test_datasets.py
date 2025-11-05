"""
Tests for dataset modules.
"""

import pytest
import torch
from PIL import Image

from srmbench.datasets.mnist_sudoku import MnistSudokuDataset


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

"""
Example script for MNIST Sudoku dataset usage.
This matches the example in README.md.

⚠️ NOTE: The current MNIST Sudoku classifier model requires retraining.
Dataset samples should show 100% valid sudokus but the current classifier
shows ~0% accuracy. The dataset itself generates valid sudokus correctly.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from srmbench.datasets import MnistSudokuDataset
from srmbench.evaluations import MnistSudokuEvaluation

# Define transforms for images and masks
image_mask_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Scales from [0,255] to [0,1]
])

# Create dataset with transforms
dataset = MnistSudokuDataset(
    stage="test",
    transform=image_mask_transform,
    mask_transform=image_mask_transform
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
)

# Use with evaluation
evaluation = MnistSudokuEvaluation()

# Evaluate batches (just first batch for testing)
for images, masks in dataloader:
    results = evaluation.evaluate(images)
    
    # duplicate_count = 0 means valid sudoku (no duplicates)
    print(f"Valid Sudoku: {results['is_valid_sudoku'].float().mean():.2%}")
    print(f"Avg Duplicate Count: {results['duplicate_count'].float().mean():.2f}")
    print(f"✅ MNIST Sudoku example works!")
    break  # Just test first batch


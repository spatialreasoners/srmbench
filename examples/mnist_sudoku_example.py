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

# Create dataset
dataset = MnistSudokuDataset(stage="test")

# Define transform: PIL Image (H, W) -> Tensor (H, W) in [0, 1]
# Note: ToImage() converts PIL to Tensor, ToDtype with scale=True normalizes [0,255] -> [0,1]
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Scales from [0,255] to [0,1]
    transforms.Lambda(lambda x: x.squeeze(0)),       # Remove channel dimension
])

# Collate function to handle (image, mask) tuples
def collate_fn(batch):
    images = torch.stack([transform(item[0]) for item in batch])
    masks = [item[1] for item in batch]
    return images, masks

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
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


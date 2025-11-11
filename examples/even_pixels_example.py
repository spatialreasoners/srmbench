"""
Example script for Even Pixels dataset usage.
This matches the example in README.md.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from srmbench.datasets import EvenPixelsDataset
from srmbench.evaluations import EvenPixelsEvaluation

# Define transform: PIL RGB (H, W, 3) -> Tensor (3, H, W) in [-1, 1]
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Scales from [0,255] to [0,1]
    transforms.Lambda(lambda x: x * 2.0 - 1.0),      # Normalize to [-1,1]
])

# Create dataset with transforms
dataset = EvenPixelsDataset(stage="test", transform=transform)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
)

# Use with evaluation
evaluation = EvenPixelsEvaluation()

# Evaluate batches (just first batch for testing)
for images in dataloader:
    results = evaluation.evaluate(images)
    print(f"Saturation STD: {results['saturation_std']:.4f}")
    print(f"Value STD: {results['value_std']:.4f}")
    print(f"Color Imbalance: {results['color_imbalance_count']:.0f} pixels")
    print(f"Perfect Balance: {results['is_color_count_even']:.2%}")
    print(f"✅ Even Pixels example works!")
    break  # Just test first batch


"""
Example script for Even Pixels dataset usage.
This matches the example in README.md.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from srmbench.datasets import EvenPixelsDataset
from srmbench.evaluations import EvenPixelsEvaluation

# Create dataset
dataset = EvenPixelsDataset(stage="test")

# Define transform: PIL RGB (H, W, 3) -> Tensor (3, H, W) in [-1, 1]
# Note: ToImage() converts PIL to Tensor, ToDtype with scale=True normalizes [0,255] -> [0,1]
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Scales from [0,255] to [0,1]
    transforms.Lambda(lambda x: x * 2.0 - 1.0),      # Normalize to [-1,1]
])

# Collate function (dataset returns (image, mask) tuple)
def collate_fn(batch):
    images = torch.stack([transform(item[0]) for item in batch])
    return images

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
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


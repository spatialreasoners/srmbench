"""
Example script for Counting Objects dataset usage.
This matches the example in README.md.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from srmbench.datasets import CountingObjectsFFHQ
from srmbench.evaluations import CountingObjectsEvaluation

# Create dataset (polygons or stars variant)
# NOTE: image_resolution=(128, 128) matches the model training resolution
dataset = CountingObjectsFFHQ(
    stage="test",
    object_variant="polygons",  # or "stars"
    image_resolution=(128, 128),
    are_nums_on_images=True,
)

# Define transform: PIL RGB (H, W, 3) -> Tensor (3, H, W) in [-1, 1]
# Note: ToImage() converts PIL to Tensor, ToDtype with scale=True normalizes [0,255] -> [0,1]
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Scales from [0,255] to [0,1]
    transforms.Lambda(lambda x: x * 2.0 - 1.0),      # Normalize to [-1,1]
])

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda batch: torch.stack([transform(img) for img in batch])
)

# Use with evaluation
evaluation = CountingObjectsEvaluation(object_variant="polygons", device="cpu")

# Evaluate batches (just first batch for testing)
for images in dataloader:
    results = evaluation.evaluate(images, include_counts=True)
    print(f"Vertices Uniform: {results['are_vertices_uniform']:.2%}")
    print(f"Numbers Match Objects: {results['numbers_match_objects']:.2%}")
    print(f"✅ Counting Objects example works!")
    break  # Just test first batch


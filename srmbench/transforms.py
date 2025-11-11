"""
Transform utilities for SRM Benchmarks datasets.

Provides standard transformations for converting PIL Images to tensors
with the appropriate normalization for each evaluation type.
"""

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image


class MnistSudokuTransform:
    """Transform for MNIST Sudoku dataset.
    
    Converts grayscale PIL Image to float tensor normalized to [0, 1].
    Input: PIL Image (H, W) in grayscale
    Output: Tensor (H, W) in range [0, 1]
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),  # Converts to [0, 1]
            transforms.Lambda(lambda x: x.squeeze(0) if x.shape[0] == 1 else x),  # Remove channel dim
        ])
    
    def __call__(self, sample):
        """Transform a (image, mask) tuple."""
        if isinstance(sample, tuple):
            image, mask = sample
            return self.transform(image), mask
        return self.transform(sample)


class EvenPixelsTransform:
    """Transform for Even Pixels dataset.
    
    Converts RGB PIL Image to float tensor normalized to [-1, 1].
    Input: PIL Image (H, W, 3) in RGB
    Output: Tensor (3, H, W) in range [-1, 1]
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),  # Converts to [0, 1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # Convert to [-1, 1]
        ])
    
    def __call__(self, sample):
        """Transform a (image, mask) tuple."""
        if isinstance(sample, tuple):
            image, mask = sample
            return self.transform(image), mask
        return self.transform(sample)


class CountingObjectsTransform:
    """Transform for Counting Objects dataset.
    
    Converts RGB PIL Image to float tensor normalized to [-1, 1].
    Input: PIL Image (H, W, 3) in RGB
    Output: Tensor (3, H, W) in range [-1, 1]
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),  # Converts to [0, 1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0),  # Convert to [-1, 1]
        ])
    
    def __call__(self, sample):
        """Transform image."""
        # CountingObjects datasets return just the image, not a tuple
        return self.transform(sample)


def collate_mnist_sudoku(batch):
    """Collate function for MNIST Sudoku DataLoader.
    
    Args:
        batch: List of (image_tensor, mask) tuples
    
    Returns:
        Tuple of (batched_images, masks)
    """
    images = torch.stack([item[0] for item in batch])
    masks = [item[1] for item in batch]
    return images, masks


def collate_even_pixels(batch):
    """Collate function for Even Pixels DataLoader.
    
    Args:
        batch: List of (image_tensor, mask) tuples
    
    Returns:
        Tuple of (batched_images, masks)
    """
    images = torch.stack([item[0] for item in batch])
    masks = [item[1] for item in batch]
    return images, masks


def collate_counting_objects(batch):
    """Collate function for Counting Objects DataLoader.
    
    Args:
        batch: List of image tensors
    
    Returns:
        Batched images tensor
    """
    return torch.stack(batch)


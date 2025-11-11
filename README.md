# SRM Benchmarks
Package with benchmark datasets to see how good is your image generative model at understanding complex spatial relationships. Those are the datasets used in the ICML 2025 paper [Spatial Reasoning with Denoising Models](https://geometric-rl.mpi-inf.mpg.de/srm/).

## Installation
### From PyPI
```bash
pip install srmbench
```

### From source
```bash
git clone https://github.com/spatialreasoners/srmbench.git
cd srmbench
pip install -e .
```

### Development installation
```bash
git clone https://github.com/spatialreasoners/srmbench.git
cd srmbench
pip install -e ".[dev]"
```

## Usage

### Available Datasets

SRM Benchmarks provides three main datasets for evaluating spatial reasoning capabilities:
- **MNIST Sudoku**: Sudoku puzzles with MNIST digits
- **Even Pixels**: Images with specific color distribution constraints
- **Counting Objects**: Images with polygons or stars to count (with optional numbers overlay)

### Quick Start

#### 1. MNIST Sudoku Dataset

```python
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

# Evaluate batches
for images, masks in dataloader:
    # Here you can apply the mask and reconstruct using your model.
    # For example:
    # images = model(images * masks)

    results = evaluation.evaluate(images)
    # duplicate_count = 0 means valid sudoku (no duplicates)
    print(f"Valid Sudoku: {results['is_valid_sudoku'].float().mean():.2%}")
    print(f"Avg Duplicate Count: {results['duplicate_count'].float().mean():.2f}")
```

#### 2. Even Pixels Dataset

```python
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

# Evaluate batches
for images in dataloader:
    results = evaluation.evaluate(images)
    print(f"Saturation STD: {results['saturation_std']:.4f}")
    print(f"Value STD: {results['value_std']:.4f}")
    print(f"Color Imbalance: {results['color_imbalance_count']:.0f} pixels")
    print(f"Perfect Balance: {results['is_color_count_even']:.2%}")
```

#### 3. Counting Objects Dataset

```python
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from srmbench.datasets import CountingObjectsFFHQ
from srmbench.evaluations import CountingObjectsEvaluation

# Create dataset (polygons or stars variant)
# NOTE: Use image_resolution=(128, 128) to match model training resolution
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

# Use with evaluation (set device="cpu" if no GPU available)
evaluation = CountingObjectsEvaluation(object_variant="polygons", device="cpu")

# Evaluate batches
for images in dataloader:
    results = evaluation.evaluate(images, include_counts=True)
    print(f"Vertices Uniform: {results['are_vertices_uniform']:.2%}")
    print(f"Numbers Match Objects: {results['numbers_match_objects']:.2%}")
```

### Evaluation Metrics

Each evaluation returns different metrics:

**MNIST Sudoku**:
- `is_valid_sudoku`: Boolean indicating whether the sudoku is valid (no duplicate digits in any row, column, or subgrid)
- `duplicate_count`: Total count of duplicate violations (0 = perfect valid sudoku, higher = more duplicates)

**Even Pixels**:
- `saturation_std`: Standard deviation of saturation across the image (should be ~0 for uniform saturation)
- `value_std`: Standard deviation of value/brightness across the image (should be ~0 for uniform brightness)
- `color_imbalance_count`: Number of pixels deviating from a perfect 50/50 split between the two main colors (0 = perfectly balanced)
- `is_color_count_even`: Boolean indicating whether the two main colors have exactly equal pixel counts (1.0 = balanced, 0.0 = unbalanced)

**Counting Objects**:
- `are_vertices_uniform`: Fraction of images where all objects have the same number of vertices
- `numbers_match_objects`: Fraction of images where the displayed numbers match the actual object counts (high for dataset images, low for random images)
- Additional vertex/polygon count distributions (with `include_counts=True`)
- Confidence scores (with `include_confidences=True`)

### Running tests
```bash
pytest
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{wewer25srm,
  title     = {Spatial Reasoning with Denoising Models},
  author    = {Wewer, Christopher and Pogodzinski, Bartlomiej and Schiele, Bernt and Lenssen, Jan Eric},
  booktitle = {International Conference on Machine Learning ({ICML})},
  year      = {2025},
}
```
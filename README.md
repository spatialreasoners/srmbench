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
import numpy as np
import torch
from torch.utils.data import DataLoader
from srmbench.datasets import MnistSudokuDataset
from srmbench.evaluations import MnistSudokuEvaluation

# Create dataset
dataset = MnistSudokuDataset(stage="test")

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)

# Get a single sample
image, mask = dataset[0]  # PIL Images (252x252)

# Use with evaluation
evaluation = MnistSudokuEvaluation()

# Convert PIL images to tensors
def pil_to_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)

# Evaluate a batch (requires tensors in shape: [batch, 252, 252])
for images, masks in dataloader:
    images_tensor = torch.stack([pil_to_tensor(img) for img in images])
    
    results = evaluation.evaluate(images_tensor)
    # duplicate_count = 0 means valid sudoku (no duplicates)
    print(f"Valid Sudoku: {results['is_valid_sudoku'].mean():.2%}")
    print(f"Avg Duplicate Count: {results['duplicate_count'].mean():.2f}")
```

#### 2. Even Pixels Dataset

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from srmbench.datasets import EvenPixelsDataset
from srmbench.evaluations import EvenPixelsEvaluation

# Create dataset with custom parameters
dataset = EvenPixelsDataset(
    stage="test",
    saturation=1.0,
    value=0.7,
    num_bins=256
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)

# Get a single sample
image, mask = dataset[0]  # PIL RGB Images (256x256)

# Use with evaluation
evaluation = EvenPixelsEvaluation(num_bins=256)

# Convert PIL images to tensors and normalize to [-1, 1]
def pil_rgb_to_tensor(image):
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1) * 2.0 - 1.0

# Evaluate a batch (requires tensors in shape: [batch, 3, 256, 256], range [-1, 1])
for images, masks in dataloader:
    images_tensor = torch.stack([pil_rgb_to_tensor(img) for img in images])
    
    results = evaluation.evaluate(images_tensor)
    print(f"Saturation STD: {results['saturation_std']:.4f}")
    print(f"Value STD: {results['value_std']:.4f}")
    print(f"Color Imbalance: {results['color_imbalance_count']:.0f} pixels")
    print(f"Perfect Balance: {results['is_color_count_even']:.2%}")
```

#### 3. Counting Objects Dataset

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from srmbench.datasets import CountingObjectsFFHQ
from srmbench.evaluations import CountingPolygonsEvaluation

# Create dataset (polygons or stars variant)
dataset = CountingObjectsFFHQ(
    stage="test",
    object_variant="polygons",  # or "stars"
    image_resolution=(128, 128),
    are_nums_on_images=True,
    min_vertices=3,
    max_vertices=7,
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)

# Get a single sample
image = dataset[0]  # PIL RGB Image (128x128)

# Use with evaluation
evaluation = CountingPolygonsEvaluation(object_variant="polygons")

# Convert PIL images to tensors and normalize to [-1, 1]
def pil_rgb_to_tensor(image):
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1) * 2.0 - 1.0

# Evaluate a batch (requires tensors in shape: [batch, 3, 128, 128], range [-1, 1])
for images in dataloader:
    images_tensor = torch.stack([pil_rgb_to_tensor(img) for img in images])
    
    results = evaluation.evaluate(images_tensor, include_counts=True)
    print(f"Vertices Uniform: {results['are_vertices_uniform']:.2%}")
    print(f"Numbers Consistent: {results['are_numbers_and_objects_consistent']:.2%}")
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
- `are_numbers_and_objects_consistent`: Fraction of images where the displayed numbers match the actual object counts (high for dataset images, low for random images)
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
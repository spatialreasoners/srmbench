# SRM Benchmarks Examples

This directory contains working examples that demonstrate how to use each dataset and evaluation in the SRM Benchmarks package.

## Running the Examples

Make sure you have installed the package:

```bash
pip install -e .
```

Then run any example:

```bash
python examples/mnist_sudoku_example.py
python examples/even_pixels_example.py
python examples/counting_objects_example.py
```

## Examples

### 1. MNIST Sudoku Example
- **File**: `mnist_sudoku_example.py`
- **Description**: Shows how to use the MNIST Sudoku dataset with DataLoader and evaluation
- **Dataset**: Sudoku puzzles with MNIST digits
- **Evaluation**: Checks if sudoku is valid (no duplicates) and counts violations

### 2. Even Pixels Example
- **File**: `even_pixels_example.py`
- **Description**: Shows how to use the Even Pixels dataset with DataLoader and evaluation
- **Dataset**: Images with specific color distribution constraints
- **Evaluation**: Measures color balance and distribution metrics

### 3. Counting Objects Example
- **File**: `counting_objects_example.py`
- **Description**: Shows how to use the Counting Objects dataset with DataLoader and evaluation
- **Dataset**: Images with polygons or stars overlaid on FFHQ faces
- **Evaluation**: Checks consistency between displayed numbers and actual object counts

## Notes

- All examples use `torchvision.transforms.v2` for PIL Image to Tensor conversion
- Examples process only the first batch for quick testing
- The Counting Objects example uses `device="cpu"` by default (change to `"cuda"` if you have a GPU)
- Transform pipeline includes proper normalization for each dataset type


#!/usr/bin/env python3
"""
Main entry point with global beartype type checking.

This approach uses beartype's import hooks to automatically apply type checking
to all functions and methods without cluttering the code with @beartype decorators.
"""

import os
from pathlib import Path

# Check if we should enable beartype checking
DEBUG = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes", "on")
ENABLE_BEARTYPE = os.getenv("ENABLE_BEARTYPE", "true").lower() in ("true", "1", "yes", "on")

if DEBUG and ENABLE_BEARTYPE:
    try:
        from jaxtyping import install_import_hook
        import torch
        
        # Configure torch for better debugging
        torch.set_printoptions(
            threshold=8,
            edgeitems=2
        )
        
        # Configure beartype and jaxtyping for the srmbench package
        with install_import_hook(
            ("srmbench",),
            ("beartype", "beartype"),
        ):
            # Import the actual modules after setting up the hook
            from srmbench.evaluations.mnist_sudoku_evaluation import MnistSudokuEvaluation
            from srmbench.datasets.mnist_sudoku import MnistSudokuDataset
            
            def test_sudoku():
                """Test function with automatic beartype checking."""
                dataset = MnistSudokuDataset()
                evaluation = MnistSudokuEvaluation()
                
                # This will trigger a beartype error if DEBUG is enabled
                a: int = "a"  # This should fail with beartype
                
                # Process samples in batches for efficiency
                NUM_SAMPLES = 10
                batch_size = 4  # Process 4 samples at a time
                num_batches = (NUM_SAMPLES + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, NUM_SAMPLES)
                    actual_batch_size = end_idx - start_idx
                    
                    # Collect samples for this batch
                    batch_images = []
                    for i in range(start_idx, end_idx):
                        sample = dataset[i]
                        batch_images.append(sample["image"])
                    
                    # Stack into a single tensor
                    batch_tensor = torch.stack(batch_images)  # [batch_size, height, width]
                    
                    # Evaluate the entire batch at once
                    pred = evaluation.evaluate(batch_tensor)
                    
                    # Check results for each sample in the batch
                    for i in range(actual_batch_size):
                        assert pred['is_accurate'][i]
                        assert pred['distance'][i] == 0 
                    
            def main() -> None:
                """Main function with automatic beartype checking."""
                test_sudoku()
                
    except ImportError as e:
        print(f"⚠️  Beartype/jaxtyping not available: {e}")
        print("   Install with: pip install beartype jaxtyping")
        print("   Continuing without enhanced type checking...")
        
        # Fallback to regular imports
        from srmbench.evaluations.mnist_sudoku_evaluation import MnistSudokuEvaluation
        from srmbench.datasets.mnist_sudoku import MnistSudokuDataset
        
        def test_sudoku():
            dataset = MnistSudokuDataset()
            evaluation = MnistSudokuEvaluation()
            
            # Process samples in batches for efficiency
            NUM_SAMPLES = 10
            batch_size = 4  # Process 4 samples at a time
            num_batches = (NUM_SAMPLES + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, NUM_SAMPLES)
                actual_batch_size = end_idx - start_idx
                
                # Collect samples for this batch
                batch_images = []
                for i in range(start_idx, end_idx):
                    sample = dataset[i]
                    batch_images.append(sample["image"])
                
                # Stack into a single tensor
                batch_tensor = torch.stack(batch_images)  # [batch_size, height, width]
                
                # Evaluate the entire batch at once
                pred = evaluation.evaluate(batch_tensor)
                
                # Check results for each sample in the batch
                for i in range(actual_batch_size):
                    assert pred['is_accurate'][i]
                    assert pred['distance'][i] == 0 
                
        def main() -> None:
            test_sudoku()
            
else:
    # No beartype checking - regular imports
    from srmbench.evaluations.mnist_sudoku_evaluation import MnistSudokuEvaluation
    from srmbench.datasets.mnist_sudoku import MnistSudokuDataset
    
    def test_sudoku():
        dataset = MnistSudokuDataset()
        evaluation = MnistSudokuEvaluation()
        
        # Process samples in batches for efficiency
        NUM_SAMPLES = 10
        batch_size = 4  # Process 4 samples at a time
        num_batches = (NUM_SAMPLES + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, NUM_SAMPLES)
            actual_batch_size = end_idx - start_idx
            
            # Collect samples for this batch
            batch_images = []
            for i in range(start_idx, end_idx):
                sample = dataset[i]
                batch_images.append(sample["image"])
            
            # Stack into a single tensor
            batch_tensor = torch.stack(batch_images)  # [batch_size, height, width]
            
            # Evaluate the entire batch at once
            pred = evaluation.evaluate(batch_tensor)
            
            # Check results for each sample in the batch
            for i in range(actual_batch_size):
                assert pred['is_accurate'][i]
                assert pred['distance'][i] == 0 
            
    def main() -> None:
        test_sudoku()

if __name__ == "__main__":
    main()

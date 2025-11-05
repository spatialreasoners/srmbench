"""
SRM Benchmarks package with optional beartype type checking.
"""

from .__version__ import VERSION, VERSION_INFO, __version__, __version_info__

__all__ = [
    "__version__",
    "__version_info__",
    "VERSION",
    "VERSION_INFO",
    "enable_beartype_checking",
    "disable_beartype_checking",
]


def enable_beartype_checking():
    """
    Enable beartype checking for improved type safety and error messages.

    This function should be called before importing other srmbench modules
    if you want enhanced type checking. It uses jaxtyping's import hooks
    to automatically apply beartype decorators to all functions and methods.

    Example:
        import srmbench as srm
        srm.enable_beartype_checking()  # Enable before using other functions

        # Now use srmbench with enhanced type checking
        from srmbench.datasets.mnist_sudoku import MnistSudokuDataset
        dataset = MnistSudokuDataset()  # Automatically type-checked!
    """
    try:
        from jaxtyping import install_import_hook

        # Configure beartype and jaxtyping for the srmbench package
        with install_import_hook(
            ("srmbench",),
            ("beartype", "beartype"),
        ):
            pass  # The hook is now installed

        print("✅ Beartype checking enabled for enhanced type safety")
        return True

    except ImportError:
        print(
            "⚠️  Beartype/jaxtyping not available - install with: pip install beartype jaxtyping"
        )
        print("   Continuing without enhanced type checking...")
        return False


def disable_beartype_checking():
    """
    Disable beartype checking (this is the default behavior).

    When beartype checking is disabled, the code runs normally without
    runtime type checking, which is faster but less safe.
    """
    print("ℹ️  Beartype checking disabled - running without runtime type checking")


# Import modules (without beartype by default)
from .datasets import *
from .evaluations import *

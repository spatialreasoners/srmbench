#!/usr/bin/env python3
"""
Setup script for srmbench package.
This provides backward compatibility for older pip versions.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Datasets and evaluation from the Spatial Reasoning with Denoising Models paper"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="srmbench",
    version="0.1.0",
    author="Bartlomiej Pogodzinski, Christopher Wewer, Bernt Schiele, Jan Eric Lenssen",
    author_email="bpogodzi@mpi-inf.mpg.de",
    description="Datasets and evaluation from the Spatial Reasoning with Denoising Models paper",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/spatialreasoners/srmbench",
    packages=find_packages(include=["srmbench*"]),
    package_data={
        "srmbench": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "all": [
            "srmbench[dev]",
        ],
    },
    entry_points={
        "console_scripts": [
            "srm-bench=srmbench.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="spatial reasoning, diffusion models, flow matching, image generation, benchmark",
    zip_safe=False,
)

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
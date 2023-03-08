# Neural Fields with JAX 

[![codecov](https://codecov.io/gh/jejjohnson/eqx-nerf/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/jejjohnson/eqx-nerf)
[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/eqx-nerf/badge)](https://www.codefactor.io/repository/github/jejjohnson/eqx-nerf)

This package implements some of the standard neural field algorithms in JAX using the `equinox` backend.


---
## Algorithms

* SIREN
* Modulated SIREN
* Positional Encoding
* Multiplicative Filter Networks


---
## Installation

This package isn't pip-worthy (yet) but here are a few options for installation.

**Option I**: Use the `pip` install option (locally)

```bash
https://github.com/jejjohnson/eqx-nerf.git
cd eqx-nerf
pip install -e .
```

**Option II**: Install it from pip directly.

```bash
pip install "git+https://github.com/jejjohnson/py_template.git"
```


---
## Inspiration

* [plum](https://github.com/wesselb/plum)
* [GPJax](https://github.com/JaxGaussianProcesses/GPJax/tree/master)
* [Nvidia-Merlin DataLoader](https://github.com/NVIDIA-Merlin/dataloader/tree/main)
* [xrft](https://github.com/xgcm/xrft/tree/master)

# Neural Fields with JAX 

[![codecov](https://codecov.io/gh/jejjohnson/eqx-nerf/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/jejjohnson/eqx-nerf)
[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/eqx-nerf/badge)](https://www.codefactor.io/repository/github/jejjohnson/eqx-nerf)

This package implements some of the standard neural field algorithms in JAX using the `equinox` backend.


---
## Installation

This package isn't pip-worthy (yet) but here are a few options for installation.

**Option I**: Use the `pip` install option (locally)

```bash
https://github.com/jejjohnson/eqx-nerf.git
cd eqx-nerf
pip install -e .[dev, jlab, test]
```

**Option II**: Install it from pip directly.

```bash
pip install "git+https://github.com/jejjohnson/eqx-nerf.git"
```

---
## Usage

```python
from equinox.nn.linear import Identity
from eqx_nerf import SirenNet
import jax.random as jrandom
import random
import jax

key = jrandom.PRNGKey(random.randint(-1, 1))
net = SirenNet(
    in_size=2,
    out_size=1,
    width_size=8,
    depth=3,
    final_activation=Identity(),
    w0_initial=30.0,                  # 
    key=key
)

# vector convention
key, init_key = jrandom.split(key, 2)
n_dims = 2
x = jrandom.normal(init_key, (n_dims,))
out = net(x)   # (1,)

# batch convention
key, init_key = jrandom.split(key, 2)
n_batch, n_dims = 10, 2
x_batch = jrandom.normal(init_key, (n_batch, n_dims))
out = jax.vmap(net)(x_batch)  # (10,1)
```

---
## Algorithms

* [x] SIREN - [Sitzmann et. al., 2020](https://www.vincentsitzmann.com/siren/)
* [ ] Modulated SIREN - [Mehta et. al., 2021](https://arxiv.org/abs/2104.03960)
* [ ] Multiplicative Filter Networks (MFN) - [Fathony et. al., 2021](https://github.com/boschresearch/multiplicative-filter-networks)
    * [ ] Fourier Net
    * [ ] Gabor Net
* [ ] Neural Implicit Flows - [Pan et. al., 2023](https://arxiv.org/abs/2204.03216)
* [ ] ResNet MFN - [Shekarforoush et. al., 2022](https://shekshaa.github.io/ResidualMFN/)


---
## Inspiration

* [lucidrains/siren-pytorch](https://github.com/lucidrains/siren-pytorch/tree/master)

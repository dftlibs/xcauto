[![Build status](https://github.com/dftlibs/xcauto/workflows/Test/badge.svg)](https://github.com/dftlibs/xcauto/actions)
[![PyPI badge](https://badge.fury.io/py/xcauto.svg)](https://badge.fury.io/py/xcauto)
[![License](https://img.shields.io/badge/license-%20MPL--v2.0-blue.svg)](LICENSE)


# xcauto: Arbitrary order exchange-correlation functional derivatives using JAX

![Automatic for the functions, easy for the people](img/cover.png)

This library computes arbitrary-order exchange-correlation function(al) derivatives
using [JAX](https://jax.readthedocs.io/).

The emphasis of this project is on **ease of use** and **ease of adding
functionals** in **Python**.  The focus is not (yet) on performance.  Our hope is
that this project can make it easier to test new implementations of functional
derivatives but maybe also used directly to provide functional derivatives in a
density functional theory program.


## Acknowledgements and recommended citation

[JAX](https://jax.readthedocs.io/) does all the heavy lifting by
automatically differentiating the exchange-correlation functions. Please
acknowledge their authors when using this code:
https://github.com/google/jax#citing-jax:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and Skye Wanderman-Milne},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.1.55},
  year = {2018},
}
```

- We have used [Libxc](https://www.tddft.org/programs/libxc/) as reference to
  double check the computed derivatives.
- The functional definitions for VWN and PBE were ported to Python based on the
  functional definitions found in [XCFun](https://github.com/dftlibs/xcfun)
  (Copyright Ulf Ekström and contributors, Mozilla Public License v2.0).


## Authors

- Radovan Bast
- Roberto Di Remigio


## Installation

You can install this code from PyPI:
```
$ pip install xcauto
```

Installing a development version:
```
$ git clone https://github.com/dftlibs/xcauto
$ cd xc
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ flit install --symlink
```


## Example

First `pip install xcauto`, then:

```python
# use double precision floats
from jax.config import config
config.update("jax_enable_x64", True)

from xcauto.functionals import pbex_n_gnn, pbec_n_gnn, pbec_a_b_gaa_gab_gbb
from xcauto.derv import derv


def pbe_unpolarized(n, gnn):
    return pbex_n_gnn(n, gnn) + pbec_n_gnn(n, gnn)


print('up to first-order derivatives for spin-unpolarized pbe:')

n = 0.02
gnn = 0.05

d_00 = derv(pbe_unpolarized, [n, gnn], [0, 0])
d_10 = derv(pbe_unpolarized, [n, gnn], [1, 0])
d_01 = derv(pbe_unpolarized, [n, gnn], [0, 1])

print(d_00)  # -0.006987994564372291
print(d_10)  # -0.43578312569769495
print(d_01)  # -0.004509994863217848


print('few derivatives for spin-polarized pbec:')

fun = pbec_a_b_gaa_gab_gbb
a = 0.02
b = 0.05
gaa = 0.02
gab = 0.03
gbb = 0.04

d_00000 = derv(fun, [a, b, gaa, gab, gbb], [0, 0, 0, 0, 0])
d_10000 = derv(fun, [a, b, gaa, gab, gbb], [1, 0, 0, 0, 0])
d_01000 = derv(fun, [a, b, gaa, gab, gbb], [0, 1, 0, 0, 0])
d_00100 = derv(fun, [a, b, gaa, gab, gbb], [0, 0, 1, 0, 0])
d_00010 = derv(fun, [a, b, gaa, gab, gbb], [0, 0, 0, 1, 0])
d_00001 = derv(fun, [a, b, gaa, gab, gbb], [0, 0, 0, 0, 1])

print(d_00000)  # -0.0002365056872298918
print(d_10000)  # -0.020444840022142356
print(d_01000)  # -0.015836702168478496
print(d_00100)  # 0.0030212897704793786
print(d_00010)  # 0.006042579540958757
print(d_00001)  # 0.0030212897704793786
```

For more examples, see the [tests folder](tests).


## Functionals

List of implemented functions:
```
slaterx_n
slaterx_a_b
vwn3_n
vwn3_a_b
vwn5_n
vwn5_a_b
pbex_n_gnn
pbex_a_b_gaa_gab_gbb
pbec_n_gnn
pbec_a_b_gaa_gab_gbb
```

**Where are all the other functionals?** We will be adding more but our hope is
that the community will contribute these also. It is very little work to define
a functional so please send pull requests!  All the derivatives you get "for
free" thanks to [JAX](https://jax.readthedocs.io/).


## Ideas

Here we list few ideas that would be good to explore but which we haven't done
yet:

- Check performance
- Try how the code offloads to GPU or TPU
- Verify numerical stability for small densities
- Adding more functionals
- Directional derivatives
- Contracting derivatives with perturbed densities

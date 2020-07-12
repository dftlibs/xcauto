# The functional definition in this file was ported to Python
# from XCFun, which is Copyright Ulf Ekström and contributors 2009-2020
# and provided under the Mozilla Public License (v2.0)
# see also:
#   - https://github.com/dftlibs/xcfun
#   - https://github.com/dftlibs/xcfun/blob/master/LICENSE.md


import jax.numpy as np

from .slater import slaterx_a


def _b88_a_gaa(a, gaa):
    na43 = a ** (4.0 / 3.0)
    chi2 = gaa * a ** (-8.0 / 3.0)
    chi = np.sqrt(chi2)
    d = 0.0042
    b88 = -(d * na43 * chi2) / (1.0 + 6.0 * d * chi * np.arcsinh(chi))
    return slaterx_a(a) + b88


def b88_a_b_gaa_gbb(a, b, gaa, gbb):
    return _b88_a_gaa(a, gaa) + _b88_a_gaa(b, gbb)


def b88_n_gnn(n, gnn):
    a = 0.5 * n
    b = 0.5 * n
    gaa = 0.25 * gnn
    gbb = gaa
    return b88_a_b_gaa_gbb(a, b, gaa, gbb)

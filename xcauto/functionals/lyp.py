# The functional definition in this file was ported to Python
# from XCFun, which is Copyright Ulf Ekström and contributors 2009-2020
# and provided under the Mozilla Public License (v2.0)
# see also:
#   - https://github.com/dftlibs/xcfun
#   - https://github.com/dftlibs/xcfun/blob/master/LICENSE.md


import jax.numpy as np


def lyp_a_b_gaa_gab_gbb(a, b, gaa, gab, gbb):

    A = 0.04918
    B = 0.132
    C = 0.2533
    Dd = 0.349

    n = a + b
    gnn = gaa + 2.0 * gab + gbb

    CF = 0.3 * (3.0 * np.pi * np.pi) ** (2.0 / 3.0)
    icbrtn = n ** (-1.0 / 3.0)
    P = 1.0 / (1.0 + Dd * icbrtn)
    omega = np.exp(-C * icbrtn) * P * n ** (-11.0 / 3.0)
    delta = icbrtn * (C + Dd * P)
    n2 = n * n
    return -A * (
        4.0 * a * b * P / n
        + B
        * omega
        * (
            a
            * b
            * (
                2.0 ** (11.0 / 3.0) * CF * (a ** (8.0 / 3.0) + b ** (8.0 / 3.0))
                + (47.0 - 7.0 * delta) * gnn / 18.0
                - (2.5 - delta / 18.0) * (gaa + gbb)
                - (delta - 11.0) / 9.0 * (a * gaa + b * gbb) / n
            )
            - 2.0 / 3.0 * n2 * gnn
            + (2.0 / 3.0 * n2 - a * a) * gbb
            + (2.0 / 3.0 * n2 - b * b) * gaa
        )
    )


def lyp_n_gnn(n, gnn):
    a = 0.5 * n
    b = 0.5 * n
    gaa = 0.25 * gnn
    gab = gaa
    gbb = gaa
    return lyp_a_b_gaa_gab_gbb(a, b, gaa, gab, gbb)

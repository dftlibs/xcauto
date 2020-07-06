# The functional definition in this file was ported to Python
# from XCFun, which is Copyright Ulf Ekström and contributors 2009-2020
# and provided under the Mozilla Public License (v2.0)
# see also:
#   - https://github.com/dftlibs/xcfun
#   - https://github.com/dftlibs/xcfun/blob/master/LICENSE.md


import jax.numpy as np


def _vwn_a(p):
    return p[0] * p[2] / (p[0] * p[0] + p[0] * p[2] + p[3]) - 1.0


def _vwn_b(p):
    return 2.0 * (p[0] * p[2] / (p[0] * p[0] + p[0] * p[2] + p[3]) - 1) + 2.0


def _vwn_c(p):
    return (
        2.0
        * p[2]
        * (
            1.0 / np.sqrt(4.0 * p[3] - p[2] * p[2])
            - p[0]
            / (
                (p[0] * p[0] + p[0] * p[2] + p[3])
                * np.sqrt(4.0 * p[3] - p[2] * p[2])
                / (p[2] + 2.0 * p[0])
            )
        )
    )


def _vwn_x(s, p):
    return s * s + p[2] * s + p[3]


def _vwn_y(s, p):
    return s - p[0]


def _vwn_z(s, p):
    return np.sqrt(4.0 * p[3] - p[2] * p[2]) / (2.0 * s + p[2])


def _vwn_f(s, p):
    return (
        0.5
        * p[1]
        * (
            2.0 * np.log(s)
            + _vwn_a(p) * np.log(_vwn_x(s, p))
            - _vwn_b(p) * np.log(_vwn_y(s, p))
            + _vwn_c(p) * np.arctan(_vwn_z(s, p))
        )
    )


def _ufunc(x, a):
    return (1.0 + x) ** a + (1.0 - x) ** a


def vwn5_n(n):
    return vwn5_a_b(0.5 * n, 0.5 * n)


def vwn5_a_b(a, b):
    para = [-0.10498, 0.0621813817393097900698817274255, 3.72744, 12.9352]
    ferro = [-0.325, 0.0310906908696548950349408637127, 7.06042, 18.0578]
    inter = [-0.0047584, -1.0 / (3.0 * np.pi * np.pi), 1.13107, 13.0045]

    n = a + b

    r_s = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    s = r_s ** 0.5

    zeta = (a - b) / n
    g = 1.92366105093154 * (_ufunc(zeta, 4.0 / 3.0) - 2.0)
    zeta4 = zeta ** 4.0
    dd = g * (
        (_vwn_f(s, ferro) - _vwn_f(s, para)) * zeta4
        + _vwn_f(s, inter) * (1.0 - zeta4) * (9.0 / 4.0 * (2.0 ** (1.0 / 3.0) - 1.0))
    )

    return n * (_vwn_f(s, para) + dd)


def vwn3_n(n):
    return vwn3_a_b(0.5 * n, 0.5 * n)


def vwn3_a_b(a, b):
    para = [-0.4092860, 0.0621814, 13.0720, 42.7198]
    ferro = [-0.7432940, 0.0310907, 20.1231, 101.578]
    inter = [-0.0047584, -0.0337737, 1.13107, 13.0045]

    n = a + b

    r_s = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    s = r_s ** 0.5

    zeta = (a - b) / n
    g = 1.92366105093154 * (_ufunc(zeta, 4.0 / 3.0) - 2.0)
    zeta4 = zeta ** 4.0
    dd = g * (_vwn_f(s, ferro) - _vwn_f(s, para))

    return n * (_vwn_f(s, para) + dd)

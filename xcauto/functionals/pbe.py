# The functional definition in this file was ported to Python
# from XCFun, which is Copyright Ulf Ekström and contributors 2009-2020
# and provided under the Mozilla Public License (v2.0)
# see also:
#   - https://github.com/dftlibs/xcfun
#   - https://github.com/dftlibs/xcfun/blob/master/LICENSE.md


import jax.numpy as np


param_gamma = (1.0 - np.log(2.0)) / (np.pi * np.pi)
# param_beta_pbe_paper = 0.066725
param_beta_accurate = 0.06672455060314922
param_beta_gamma = param_beta_accurate / param_gamma


def _prefactor(a):
    return (
        -0.75
        * 2.0 ** (1.0 / 3.0)
        * (3.0 * np.pi * np.pi) ** (1.0 / 3.0)
        * a ** (4.0 / 3.0)
        / np.pi
    )


def _enhancement(R, a, gaa):
    # some codes use this instead:
    # mu = 0.2195149727645171
    mu = 0.066725 * np.pi * np.pi / 3.0
    st2 = (
        gaa
        / a ** (8.0 / 3.0)
        * (6.0 ** (2.0 / 3.0) / (12.0 * np.pi ** (2.0 / 3.0))) ** 2.0
    )
    t1 = 1.0 + mu * st2 / R
    return 1.0 + R - R / t1


def _ufunc(x, a):
    return (1.0 + x) ** a + (1.0 - x) ** a


def _omega(z):
    #   return (_ufunc(z, 4.0 / 3.0) - 2.0) / 0.5198421
    return (_ufunc(z, 4.0 / 3.0) - 2.0) / (2.0 * 2.0 ** (1.0 / 3.0) - 2.0)


def _eopt(sqrtr, t):
    return (
        -2.0
        * t[0]
        * (1.0 + t[1] * sqrtr * sqrtr)
        * np.log(
            1.0
            + 0.5
            / (t[0] * (sqrtr * (t[2] + sqrtr * (t[3] + sqrtr * (t[4] + t[5] * sqrtr)))))
        )
    )


def _pw92eps(zeta, r_s):
    parameters = [
        [0.03109070, 0.21370, 7.59570, 3.5876, 1.63820, 0.49294],
        [0.01554535, 0.20548, 14.1189, 6.1977, 3.36620, 0.62517],
        [0.01688690, 0.11125, 10.3570, 3.6231, 0.88026, 0.49671],
    ]
    # c = 1.709921
    c = 8.0 / (9.0 * (2.0 * 2.0 ** (1.0 / 3.0) - 2.0))
    zeta4 = zeta ** 4.0
    omegaval = _omega(zeta)
    sqrtr = r_s ** 0.5
    e0 = _eopt(sqrtr, parameters[0])
    return (
        e0
        - _eopt(sqrtr, parameters[2]) * omegaval * (1.0 - zeta4) / c
        + (_eopt(sqrtr, parameters[1]) - e0) * omegaval * zeta4
    )


def _A(eps, u3):
    return param_beta_gamma / (np.exp(-eps / (param_gamma * u3)) - 1.0)


# This is [(1+zeta)^(2/3) + (1-zeta)^(2/3)]/2, reorganized.
def _phi(a, b):
    n = a + b
    c = 2.0 ** (-1.0 / 3.0)
    n_m13 = n ** (-1.0 / 3.0)
    a_43 = a ** (2.0 / 3.0)
    b_43 = b ** (2.0 / 3.0)
    return c * n_m13 * n_m13 * (a_43 + b_43)


def _H(d2, eps, u3):
    d2A = d2 * _A(eps, u3)
    return (
        param_gamma
        * u3
        * np.log(1.0 + param_beta_gamma * d2 * (1.0 + d2A) / (1.0 + d2A * (1.0 + d2A)))
    )


def _pbec(a, b, gnn):
    n = a + b
    s = a - b
    zeta = s / n
    r_s = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    eps = _pw92eps(zeta, r_s)
    u = _phi(a, b)
    d2 = (
        (1.0 / 12.0 * 3.0 ** (5.0 / 6.0) / np.pi ** (-1.0 / 6.0)) ** 2.0
        * gnn
        / (u * u * n ** (7.0 / 3.0))
    )
    u3 = u * u * u
    return n * (eps + _H(d2, eps, u3))


def pbex_a_gaa(a, gaa):
    R = 0.804
    return _prefactor(a) * _enhancement(R, a, gaa)


def pbex_n_gnn(n, gnn):
    gaa = 0.25 * gnn
    gbb = gaa
    a = 0.5 * n
    b = 0.5 * n

    return pbex_a_gaa(a, gaa) + pbex_a_gaa(b, gbb)


def pbex_a_b_gaa_gab_gbb(a, b, gaa, gab, gbb):
    # gab not used
    return pbex_a_gaa(a, gaa) + pbex_a_gaa(b, gbb)


def pbec_n_gnn(n, gnn):
    a = 0.5 * n
    b = 0.5 * n
    return _pbec(a, b, gnn)


def pbec_a_b_gaa_gab_gbb(a, b, gaa, gab, gbb):
    gnn = gaa + 2.0 * gab + gbb
    return _pbec(a, b, gnn)

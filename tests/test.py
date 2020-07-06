from jax.config import config

config.update("jax_enable_x64", True)

from xcauto.functionals import (
    slaterx_n,
    slaterx_a_b,
    vwn3_n,
    vwn3_a_b,
    vwn5_n,
    vwn5_a_b,
    pbex_n_gnn,
    pbex_a_b_gaa_gab_gbb,
    pbec_n_gnn,
    pbec_a_b_gaa_gab_gbb,
)
from xcauto.derv import derv

import pytest


def test_slaterx_unpolarized():
    fun = slaterx_n
    n = 0.05

    d_0 = derv(fun, [n], [0])
    d_1 = derv(fun, [n], [1])

    assert d_0 == pytest.approx(-0.01360436879474179)
    assert d_1 == pytest.approx(-0.362783167859781)


def test_slaterx_polarized():
    fun = slaterx_a_b
    a = 0.02
    b = 0.05

    d_00 = derv(fun, [a, b], [0, 0])
    d_10 = derv(fun, [a, b], [1, 0])
    d_01 = derv(fun, [a, b], [0, 1])

    assert d_00 == pytest.approx(-0.022192101517910012)
    assert d_10 == pytest.approx(-0.33677806019212597)
    assert d_01 == pytest.approx(-0.4570781497340833)


def test_vwn3_unpolarized():
    fun = vwn3_n
    n = 0.05

    d_0 = derv(fun, [n], [0])
    d_1 = derv(fun, [n], [1])

    assert d_0 == pytest.approx(-0.0033243334606879206)
    assert d_1 == pytest.approx(-0.07438806748231225)


def test_vwn3_polarized():
    fun = vwn3_a_b
    a = 0.02
    b = 0.05

    d_00 = derv(fun, [a, b], [0, 0])
    d_10 = derv(fun, [a, b], [1, 0])
    d_01 = derv(fun, [a, b], [0, 1])

    assert d_00 == pytest.approx(-0.004585556701793601)
    assert d_10 == pytest.approx(-0.09784372868839261)
    assert d_01 == pytest.approx(-0.06303262897292465)


def test_vwn5_unpolarized():
    fun = vwn5_n
    n = 0.05

    d_0 = derv(fun, [n], [0])
    d_1 = derv(fun, [n], [1])

    assert d_0 == pytest.approx(-0.0024185694846377663)
    assert d_1 == pytest.approx(-0.05545437748839972)


def test_vwn5_polarized():
    fun = vwn5_a_b
    a = 0.02
    b = 0.05

    d_00 = derv(fun, [a, b], [0, 0])
    d_10 = derv(fun, [a, b], [1, 0])
    d_01 = derv(fun, [a, b], [0, 1])

    assert d_00 == pytest.approx(-0.0033313701535310127)
    assert d_10 == pytest.approx(-0.07637590965268805)
    assert d_01 == pytest.approx(-0.04561294583487131)


def test_pbex_unpolarized():
    fun = pbex_n_gnn
    n = 0.05
    gnn = 0.05

    d_00 = derv(fun, [n, gnn], [0, 0])
    d_10 = derv(fun, [n, gnn], [1, 0])
    d_01 = derv(fun, [n, gnn], [0, 1])

    assert d_00 == pytest.approx(-0.019209216943125326)
    assert d_10 == pytest.approx(-0.3664969949653799)
    assert d_01 == pytest.approx(-0.05465579631923576)


def test_pbec_unpolarized():
    fun = pbec_n_gnn
    n = 0.05
    gnn = 0.05

    d_00 = derv(fun, [n, gnn], [0, 0])
    d_10 = derv(fun, [n, gnn], [1, 0])
    d_01 = derv(fun, [n, gnn], [0, 1])

    assert d_00 == pytest.approx(-0.0001881151998459174)
    assert d_10 == pytest.approx(-0.018915857103503467)
    assert d_01 == pytest.approx(0.005695886444674986)


def test_pbex_polarized():
    fun = pbex_a_b_gaa_gab_gbb
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

    assert d_00000 == pytest.approx(-0.030022232676348133)
    assert d_10000 == pytest.approx(-0.4399544626404114)
    assert d_01000 == pytest.approx(-0.4179516165421625)
    assert d_00100 == pytest.approx(-0.037768955504833064)
    assert d_00010 == pytest.approx(0.0)
    assert d_00001 == pytest.approx(-0.07798714870271929)


def test_pbec_polarized():
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

    assert d_00000 == pytest.approx(-0.0002365056872298918)
    assert d_10000 == pytest.approx(-0.020444840022142356)
    assert d_01000 == pytest.approx(-0.015836702168478496)
    assert d_00100 == pytest.approx(0.0030212897704793786)
    assert d_00010 == pytest.approx(0.006042579540958757)
    assert d_00001 == pytest.approx(0.0030212897704793786)

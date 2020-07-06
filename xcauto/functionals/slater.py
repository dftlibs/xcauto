import jax.numpy as np


def slaterx_a_b(a, b):
    cx = (-3.0 / 4.0) * (6.0 / np.pi) ** (1.0 / 3.0)
    return cx * (a ** (4.0 / 3.0) + b ** (4.0 / 3.0))


def slaterx_n(n):
    return slaterx_a_b(0.5 * n, 0.5 * n)

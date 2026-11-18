"""Comprueba la indexación y el estado de SGD incluso en un host solo CPU."""

import importlib.util
import os
import subprocess
import sys
from textwrap import dedent

import pytest


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="Numba no está instalado.")
def test_two_updates_and_last_partial_block():
    source = dedent("""
        import numpy as np
        from p3.cuda_kernel import THREADS, sgd_momentum

        params = np.linspace(-1, 1, 259, dtype=np.float32)
        velocity = np.zeros_like(params)
        expected = params.copy()
        expected_velocity = velocity.copy()
        blocks = (params.size + THREADS - 1) // THREADS
        lr, mu = np.float32(0.01), np.float32(0.9)
        for factor in (1.0, -0.3):
            gradient = np.linspace(0.5, 2, 259, dtype=np.float32) * np.float32(factor)
            next_params = np.empty_like(params)
            next_velocity = np.empty_like(velocity)
            sgd_momentum[blocks, THREADS](params, gradient, velocity, next_params, next_velocity, lr, mu)
            expected_velocity = mu * expected_velocity - lr * gradient
            expected += expected_velocity
            np.testing.assert_allclose(next_params, expected, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(next_velocity, expected_velocity, rtol=1e-6, atol=1e-7)
            params, velocity = next_params, next_velocity
    """)
    subprocess.run([sys.executable, "-c", source], env={**os.environ, "NUMBA_ENABLE_CUDASIM": "1"}, check=True)

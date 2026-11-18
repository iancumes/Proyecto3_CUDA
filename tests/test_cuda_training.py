import numpy as np
import pytest


def test_cuda_sgd_matches_keras_across_updates():
    tf = pytest.importorskip("tensorflow")
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("Se requiere una GPU para comprobar el kernel real.")
    from p3.cuda_train import MomentumCUDA, configure_cuda

    device = configure_cuda(tf)
    with tf.device(device):
        initial = np.linspace(-1, 1, 259, dtype=np.float32)
        expected = tf.Variable(initial)
        actual = tf.Variable(initial)
        keras_sgd = tf.keras.optimizers.SGD(0.01, momentum=0.9)
        cuda_sgd = MomentumCUDA(tf, [actual], 0.01, 0.9)
        for factor in (1.0, -0.3):
            gradient = tf.constant(initial * factor)
            keras_sgd.apply_gradients([(gradient, expected)])
            cuda_sgd.apply_gradients([(gradient, actual)])
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)
        cuda_sgd.reset()
        np.testing.assert_array_equal(cuda_sgd.velocities[0].numpy(), np.zeros_like(initial))

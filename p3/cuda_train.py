"""Interoperabilidad TensorFlow, CuPy y Numba-CUDA en una GPU."""

import numpy as np


def configure_cuda(tf):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("CUDA requiere una GPU NVIDIA. En Colab activa el acelerador T4.")
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    return "/GPU:0"


class MomentumCUDA:
    """Aplica la actualización de Keras mediante un kernel CUDA propio."""

    def __init__(self, tf, variables, learning_rate, momentum):
        import cupy
        from numba import cuda

        from .cuda_kernel import THREADS, sgd_momentum

        if not cuda.is_available():
            raise RuntimeError("TensorFlow detectó una GPU, pero Numba-CUDA no pudo inicializarla.")
        self.tf = tf
        self.cupy = cupy
        self.cuda = cuda
        self.kernel = sgd_momentum
        self.threads = THREADS
        self.learning_rate = np.float32(learning_rate)
        self.momentum = np.float32(momentum)
        self.velocities = [tf.Variable(tf.zeros_like(var), trainable=False) for var in variables]

    def reset(self):
        for velocity in self.velocities:
            velocity.assign(self.tf.zeros_like(velocity))

    def apply_gradients(self, pairs):
        tf, cp, cuda = self.tf, self.cupy, self.cuda
        pending = []
        for (gradient, variable), velocity in zip(pairs, self.velocities, strict=True):
            if gradient is None:
                raise ValueError("CNN2 produjo un gradiente ausente.")
            weight_tensor = tf.convert_to_tensor(variable)
            velocity_tensor = tf.convert_to_tensor(velocity)
            if any("GPU:0" not in t.device for t in (weight_tensor, gradient, velocity_tensor)):
                raise RuntimeError("El optimizador CUDA recibió un tensor fuera de GPU:0.")

            # TensorFlow entrega DLPack listo para leer. Los arreglos son vistas en GPU.
            weight = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(weight_tensor))
            grad = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(gradient))
            old_velocity = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(velocity_tensor))
            new_weight = cp.empty_like(weight)
            new_velocity = cp.empty_like(old_velocity)
            flat = (weight, grad, old_velocity, new_weight, new_velocity)
            views = [cuda.as_cuda_array(array.reshape(-1), sync=False) for array in flat]
            blocks = (weight.size + self.threads - 1) // self.threads
            self.kernel[blocks, self.threads](*views, self.learning_rate, self.momentum)
            pending.append((variable, velocity, flat))

        # La salida tiene que estar lista antes de entregarla a TensorFlow.
        cuda.synchronize()
        for variable, velocity, buffers in pending:
            new_weight, new_velocity = buffers[-2:]
            variable.assign(tf.experimental.dlpack.from_dlpack(new_weight.__dlpack__()))
            velocity.assign(tf.experimental.dlpack.from_dlpack(new_velocity.__dlpack__()))

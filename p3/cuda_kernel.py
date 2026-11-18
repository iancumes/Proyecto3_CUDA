"""SGD con momentum en CUDA Python. Un hilo actualiza un parámetro."""

from numba import cuda

THREADS = 256


@cuda.jit
def sgd_momentum(weight, gradient, velocity, next_weight, next_velocity, learning_rate, momentum):
    index = cuda.grid(1)
    if index < weight.size:
        current = momentum * velocity[index] - learning_rate * gradient[index]
        next_velocity[index] = current
        next_weight[index] = weight[index] + current

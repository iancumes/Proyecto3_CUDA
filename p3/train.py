"""Entrenamiento de CNN2 con pesos, lotes, pérdida y precisión comunes."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from src.config import BATCH_SIZE, RANDOM_SEED
from .data import CLASS_INDEX

LEARNING_RATE = 0.01
MOMENTUM = 0.9


def open_split(directory, split):
    images = np.load(directory / f"{split}_x.npy", mmap_mode="r")
    labels = np.load(directory / f"{split}_y.npy")
    return images, labels


def run_training(arrays, mode, epochs, output, test_arrays=None):
    if mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf
    from src.models import build_cnn2

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.set_synchronous_execution(True)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    device = "/CPU:0"
    if mode != "cpu":
        from .cuda_train import configure_cuda
        device = configure_cuda(tf)

    # Se inicializan los pesos en CPU para garantizar exactamente el mismo punto de partida.
    with tf.device("/CPU:0"):
        original = build_cnn2(learning_rate=LEARNING_RATE, dropout=0.0)
        initial = original.get_weights()
    del original
    with tf.device(device):
        model = build_cnn2(learning_rate=LEARNING_RATE, dropout=0.0)
        model.set_weights(initial)
        if mode == "cuda":
            from .cuda_train import MomentumCUDA
            optimizer = MomentumCUDA(tf, model.trainable_variables, LEARNING_RATE, MOMENTUM)
        else:
            optimizer = model.optimizer
            optimizer.build(model.trainable_variables)

    train_x, train_y = open_split(arrays, "train")
    val_x, val_y = open_split(arrays, "validation")
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    timings = {"h2d_s": 0.0, "compute_s": 0.0, "d2h_s": 0.0}

    def transfer(images, labels):
        with tf.device("/CPU:0"):
            x = tf.convert_to_tensor(images)
            y = tf.one_hot(labels, len(CLASS_INDEX), dtype=tf.float32)
        if mode == "cpu":
            return x, y
        with tf.device(device):
            return tf.identity(x), tf.identity(y)

    def update(x, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(x, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def evaluate(images, labels):
        loss_sum = 0.0
        correct = 0
        for lower in range(0, len(labels), BATCH_SIZE):
            upper = lower + BATCH_SIZE
            x, y = transfer(images[lower:upper], labels[lower:upper])
            p = model(x, training=False)
            size = len(labels[lower:upper])
            loss_sum += float(loss_fn(y, p).numpy()) * size
            correct += int(tf.reduce_sum(tf.cast(
                tf.argmax(p, axis=1) == tf.argmax(y, axis=1), tf.int32
            )).numpy())
        return {"loss": loss_sum / len(labels), "accuracy": correct / len(labels)}

    with tf.device(device):
        warm_x, warm_y = transfer(train_x[:BATCH_SIZE], train_y[:BATCH_SIZE])
        if mode != "cpu" and ("GPU:0" not in warm_x.device or "GPU:0" not in model.trainable_variables[0].handle.device):
            raise RuntimeError("TensorFlow no ubicó los tensores de entrenamiento en la GPU.")
        float(update(warm_x, warm_y).numpy())
        last_size = len(train_y) % BATCH_SIZE
        if last_size:
            float(update(*transfer(train_x[-last_size:], train_y[-last_size:])).numpy())
        model.set_weights(initial)
        if mode == "cuda":
            optimizer.reset()
        else:
            for value in optimizer.variables:
                if "momentum" in value.name or value is optimizer.iterations:
                    value.assign(tf.zeros_like(value))

        history = []
        training_s = 0.0
        validation_s = 0.0
        for epoch in range(epochs):
            order = np.random.default_rng(RANDOM_SEED + epoch).permutation(len(train_y))
            start_epoch = perf_counter()
            for lower in range(0, len(order), BATCH_SIZE):
                batch = order[lower:lower + BATCH_SIZE]
                start = perf_counter()
                images, labels = train_x[batch], train_y[batch]
                # Medir las copias de forma independiente de la lectura del array.
                with tf.device("/CPU:0"):
                    x_cpu = tf.convert_to_tensor(images)
                    y_cpu = tf.one_hot(labels, len(CLASS_INDEX), dtype=tf.float32)
                copied = perf_counter()
                if mode == "cpu":
                    x, y = x_cpu, y_cpu
                else:
                    x, y = tf.identity(x_cpu), tf.identity(y_cpu)
                    timings["h2d_s"] += perf_counter() - copied
                before_update = perf_counter()
                loss = update(x, y)
                timings["compute_s"] += perf_counter() - before_update
            before_return = perf_counter()
            last_batch_loss = float(loss.numpy())
            if mode != "cpu":
                timings["d2h_s"] += perf_counter() - before_return
            training_s += perf_counter() - start_epoch

            before_validation = perf_counter()
            metrics = evaluate(val_x, val_y)
            validation_s += perf_counter() - before_validation
            if not np.isfinite(last_batch_loss) or not np.isfinite(metrics["loss"]):
                raise RuntimeError("El entrenamiento produjo una pérdida no finita.")
            history.append({"epoch": epoch + 1, "last_batch_loss": last_batch_loss, **metrics})
            print(f"{mode} época {epoch + 1}/{epochs}: {history[-1]}", flush=True)

        result = {
            "mode": mode, "epochs": epochs, "examples": len(train_y), "batch_size": BATCH_SIZE,
            "parameters": model.count_params(), "training_s": training_s, "validation_s": validation_s,
            "imgs_per_s": epochs * len(train_y) / training_s,
            "device": model.trainable_variables[0].handle.device, "tensorflow": tf.__version__,
            "history": history, **timings,
        }
        if test_arrays is not None:
            result["held_out_test"] = evaluate(*open_split(test_arrays, "test"))
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu", "cuda", "gpu-tf"], required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-arrays", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs debe ser positivo.")
    run_training(args.arrays, args.mode, args.epochs, args.output, args.test_arrays)


if __name__ == "__main__":
    main()

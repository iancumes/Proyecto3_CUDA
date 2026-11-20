"""Distribuye la decodificación de JPEG entre procesos de CPU."""

from concurrent.futures import ProcessPoolExecutor


def decode_parallel(paths, workers):
    from .data import decode_image

    # map conserva el orden de entrada. Cada tarea procesa un grupo de imágenes.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(decode_image, paths, chunksize=32)

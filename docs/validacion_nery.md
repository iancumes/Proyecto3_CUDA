# Validación del aporte CUDA de Nery Molina

Fecha local: 18 de noviembre de 2026, America/Guatemala.

Se aplicaron las tres etapas de `Nery_Molina.zip` en la rama
`aporte/nery-cuda`, desde el commit base
`101fa8641e6c4a2e9269f46acb91aae00f27c957`. Los cinco archivos del aporte
conservan el contenido del ZIP, normalizando únicamente los finales de línea.
El entrenador de la base ya selecciona `MomentumCUDA` para `--mode cuda`
y reinicia las velocidades después del calentamiento.

## Pruebas locales

Se creó un entorno virtual con Python 3.13.11 y se instalaron las dependencias
de `requirements-p3.txt`, Numba 0.62.1 y NumPy 2.3.5. Para reproducirlo:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-p3.txt numba==0.62.1 numpy==2.3.5
.venv/bin/python -m pytest -q tests/test_cuda_training.py tests/test_cuda_kernel_sim.py -rs
```

Resultado: **1 passed, 1 skipped**. La simulación comprobó dos pasos de SGD,
la persistencia del momentum y el bloque parcial con 259 parámetros.
La prueba contra Keras se omitió porque TensorFlow no está instalado;
este host tampoco dispone de una GPU NVIDIA ni de CUDA utilizable.

Se conservan la [salida de pytest](evidencia_nery/pytest.txt) y el
[registro de hardware y versiones](evidencia_nery/environment.json).

También se ejecutaron dos pasos del kernel en simulación con 1, 256 y 259
parámetros, comparando pesos y velocidades contra NumPy. Los tres casos
pasaron. La importación del módulo de integración, el diagnóstico ante la
ausencia de GPU, la compilación de sintaxis y `pip check` pasaron.

## Revisión del aporte CPU disponible

Se revisaron `p3/data_parallel.py`, `tests/test_data_parallel.py` y
`docs/preparacion_cpu.md` en `validacion-colab`, commit
`a11aed07a3feeb836a92f69d7ad5ffbed3c0fc5a`, desde una copia temporal separada.
`ProcessPoolExecutor.map` conserva el orden de entrada y reutiliza la misma
función de decodificación que la referencia secuencial. La prueba de igualdad
de hashes, etiquetas, forma y tipo de los arrays pasó:
**1 passed**. Se conserva su [salida](evidencia_nery/revision_cpu_pytest.txt).

Observación de cobertura: la prueba crea 29 imágenes por partición y el
preparador usa `chunksize=32`. Cada partición cabe en un solo grupo, por lo
que esa prueba comprueba equivalencia pero no demuestra el uso simultáneo
de varios procesos. Conviene añadir más de 32 imágenes por partición para
ejercitar varios grupos. No se midió aceleración del preparador.

## Validación pendiente en GPU NVIDIA

La simulación local no comprueba la compilación del kernel para una GPU real,
la interoperabilidad DLPack, la ubicación de los tensores ni la equivalencia
numérica con Keras en GPU. Tampoco se realizaron el entrenamiento completo
ni mediciones de rendimiento. No se incluyen cifras de aceleración ni
capturas de GPU.

En la rama integrada `validacion-colab`, con una GPU NVIDIA y las dependencias
TensorFlow, CuPy y `numba-cuda[cu12]`, ejecutar las pruebas del ZIP sin
`NUMBA_ENABLE_CUDASIM` en el proceso principal. Guardar la salida de pytest,
`nvidia-smi` y los registros del cuaderno como evidencia de esa ejecución.
La simulación se habilita únicamente en el subproceso de su prueba.

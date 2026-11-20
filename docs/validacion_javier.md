# Validación de la preparación CPU de Javier Valladares

Fecha local: 18 de noviembre de 2026, America/Guatemala.

## Integración

Se integraron los tres archivos de `Javier_Valladares.zip` en la rama
`aporte/javier-preparacion`, sobre `main` en el commit
`77a61bd8ab96f407438561eebcc60b9eec78244c`. Conservan el contenido del ZIP,
normalizando únicamente los finales de línea. Se usó la base actual para
conservar el cuaderno, el benchmark y el generador de informes que se agregaron
después del commit mencionado por el paquete.

`p3.data.prepare_arrays` ya llama a `decode_parallel` cuando `workers > 1`.
El módulo agregado completa esa conexión: reutiliza `decode_image` y entrega
los resultados de `ProcessPoolExecutor.map` en el orden de entrada. No cambia
las etiquetas, las particiones ni la transformación de píxeles.

## Pruebas de preparación

```bash
python -m pip install -r requirements-p3.txt
python -m pytest -q tests
```

Resultado en esta rama: **9 passed**. La [salida registrada](evidencia_javier/pytest.txt)
y el [entorno con versiones](evidencia_javier/environment.json) documentan la ejecución.

La prueba original del paquete usa 29 imágenes por partición y el preparador
agrupa hasta 32 archivos por tarea. Por eso se agregó
`tests/test_data_parallel_integration.py`, que comprueba:

- 87 imágenes por partición: tres grupos, con dos procesos disponibles.
- Igualdad bit a bit de imágenes y hashes entre preparación serial y paralela.
- Orden del manifiesto mezclado, etiquetas, forma y tipos de los arrays.
- Conversión de RGB, escala de grises y RGBA, con normalización a `[0, 1]`.
- Propagación de errores por archivos ausentes o corruptos, en ambos modos.
- Rechazo de cero trabajadores y números negativos.
- Ejecución de la CLI, escritura de `preparation.json` y preparación separada
  de test mediante `--test`.

Cubrir varios grupos no mide el solapamiento temporal de los procesos ni
demuestra aceleración. La prueba verifica la corrección de sus resultados.

## Flujo CPU completo con una muestra sintética

Se generó un manifiesto de 29 clases con 87 imágenes de entrenamiento,
29 de validación y 29 reservadas para test. La preparación serial, la
paralela con dos procesos y la preparación ejecutada por el benchmark dieron
el mismo SHA-256:

```text
88a7759d808947ea498a183116b9ac1aa2286bf72cd26f58add40594f4efea1f
```

Con TensorFlow 2.21.0 se ejecutó:

```bash
python -m p3.benchmark --cpu-only \
  --manifest outputs/validacion_javier/manifest.csv \
  --out outputs/validacion_javier/benchmark --repetitions 1 --epochs 1
python -m p3.report --results outputs/validacion_javier/benchmark \
  --output outputs/validacion_javier/benchmark/Informe_CPU_sintetico.pdf
```

El manifiesto sintético y sus imágenes quedan en `outputs/validacion_javier/`
de la copia local, fuera del control de versiones. Para ejecutar este flujo
con otro manifiesto, sustituir la ruta de entrada y usar una carpeta de salida
nueva. El entrenamiento requiere instalar `tensorflow` además de las
dependencias de `requirements-p3.txt`.

CNN2 completó una época en CPU, con 2,120,413 parámetros y dos lotes de
entrenamiento, de 64 y 23 imágenes. Se verificaron pérdidas y métricas finitas,
el dispositivo CPU y la coincidencia de hashes. Se conservan
[el entrenamiento](evidencia_javier/entrenamiento_cpu_sintetico.json),
[la preparación paralela](evidencia_javier/preparacion_paralela.json),
[los registros CSV](evidencia_javier/smoke_runs.csv) y
[el resumen](evidencia_javier/smoke_summary.csv).

El generador produjo el informe preliminar de tres páginas y sus gráficos.
Se renderizaron y revisaron visualmente las tres páginas. Estos resultados
comprueban el funcionamiento del flujo; no miden exactitud sobre ASL ni
rendimiento de la ejecución completa.

También pasaron la revisión de sintaxis de los módulos, la lectura de ambos
cuadernos como JSON, `git diff --check` y la comprobación de dependencias
con `uv pip check`.

## Compatibilidad con el aporte CUDA disponible

Se combinó esta rama con `origin/aporte/nery-cuda` en
`4db9906`, en un worktree temporal separado. La combinación no presentó
conflictos. No se incorporaron esos archivos a la rama de Javier.

La suite combinada produjo **10 passed, 1 skipped**. La
[salida completa](evidencia_javier/compatibilidad_nery_pytest.txt) identifica
la prueba omitida: comparación del optimizador contra Keras en una GPU real.
La simulación del kernel comprobó dos actualizaciones y el bloque parcial.

## Pendientes para completar el proyecto

En la base `main` revisada todavía no están `p3/cuda_kernel.py` ni
`p3/cuda_train.py`; existen en la rama de Nery. Se debe integrar el aporte
correspondiente para habilitar `--mode cuda` en `main`.

Este Mac no dispone de GPU NVIDIA, `nvcc` ni `nvidia-smi`. Quedan pendientes
la prueba del kernel y de DLPack en GPU real, el dataset ASL completo y los
tres pares de dos épocas con CPU y CUDA en el mismo entorno. De esa ejecución
deben salir los tiempos, las capturas y el informe final.

El cuaderno actual clona explícitamente `validacion-colab`. Esa rama contiene
los tres archivos originales del aporte CPU, pero no las nuevas pruebas de
esta integración. Para verificar los cambios finales en Colab, hay que usar
una rama que incluya los aportes definitivos y confirmar el SHA que imprime
la primera celda; abrir un cuaderno de otra rama no cambia por sí solo la
rama que clona su código.

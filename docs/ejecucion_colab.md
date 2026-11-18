# Ejecutar el proyecto

Abre [Proyecto3_CUDA.ipynb en Colab](https://colab.research.google.com/github/iancumes/Proyecto3_CUDA/blob/validacion-colab/Proyecto3_CUDA.ipynb), elige un entorno GPU T4 y ejecuta todo. La rama validacion-colab reúne los cambios para que se puedan probar juntos. La rama main contiene la base común para que Javier y Nery apliquen sus ZIP y abran sus respectivos PR.

El cuaderno descarga el dataset, audita los archivos, compara la salida de los preparadores y verifica el kernel. Después corre tres pares alternados y descarga un ZIP con los registros, los CSV, las gráficas y el PDF. Guarda también el cuaderno ejecutado desde el menú Archivo.

Durante la medición captura las salidas del monitor de preparación, CPU y CUDA. Nombra los archivos captura_preparacion.png, captura_cpu.png y captura_cuda.png y colócalos junto a runs.csv. El generador del PDF los anexa automáticamente.

Desde la raíz de la copia de validación se puede reproducir el informe de una ejecución guardada:

```bash
python -m pip install -r requirements-p3.txt
python -m p3.report --results ruta/a/resultados --output Informe_Proyecto3_CUDA.pdf
```

Los tiempos y las capturas deben corresponder a la misma ejecución. Los logs JSONL registran la utilización por núcleo, porcentaje de GPU activa y memoria ocupada. El porcentaje que reporta nvidia-smi mide actividad durante el intervalo de muestreo. La ocupación de multiprocesadores requiere otra métrica de perfilado.

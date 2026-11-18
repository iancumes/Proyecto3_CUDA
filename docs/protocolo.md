# Protocolo de comparación

Se eligió CNN2 (dos convoluciones, sin dropout ni aumento de datos). Se usa el manifiesto ASL de 29 clases y la semilla 123 del laboratorio. En cada repetición, ambos modos comienzan con los mismos pesos y reciben los mismos lotes de 64 en el mismo orden. Se usan dos épocas para comparar rendimiento con un costo de ejecución razonable. La referencia histórica entrenó diez épocas.

Los datos se auditan y dividen una vez. Las repeticiones vuelven a leer, decodificar, redimensionar y normalizar los JPEG de entrenamiento y validación. Cada versión prepara sus propios arrays antes de entrenar. La huella SHA-256 comprueba su equivalencia. Se registra la exactitud de validación y se puede evaluar el conjunto test reservado en una ejecución aparte.

Se alterna el orden de CPU y CUDA en tres pares. El tiempo de preparación incluye la escritura de arrays en el almacenamiento local. Los intervalos cronometrados empiezan después de instalar dependencias, descargar y dividir el dataset y, para el entrenamiento, después del calentamiento y reinicio de pesos. El tiempo total del flujo medido suma preparación, entrenamiento y validación. Se informa preparación, entrenamiento y total por separado.

El tiempo de pared de entrenamiento incluye lectura de lotes, transferencias y sincronización de cálculos. La descomposición de GPU mide por separado las copias CPU→GPU, las actualizaciones sincronizadas y la devolución de la última pérdida de cada época. La validación tiene su propio cronómetro. Los tiempos de transferencia del entrenamiento no incluyen las copias de validación.

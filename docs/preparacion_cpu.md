# Preparación de imágenes

| Operación | Datos | Repeticiones |
|---|---|---:|
| Auditar archivos | Archivos fuente de las 29 clases de Kaggle | Una inspección por archivo al generar el manifiesto |
| Seleccionar y dividir | Imágenes válidas por clase | 29 selecciones sin reemplazo. 360, 120 y 120 imágenes por clase en train, validation y test |
| Leer y decodificar | JPEG de train y validation | 13,920 imágenes por preparación |
| Convertir, redimensionar y normalizar | Píxeles RGB y etiquetas | 13,920 conversiones. 12,288 componentes de color por salida |
| Escribir el lote preparado | Tensores de imagen y vectores de etiquetas | Un par de arrays por partición |

Las dos versiones usan Pillow para decodificar y redimensionar con la interpolación más cercana que usa CNN2 en Lab3Datos. La segunda versión entrega grupos de 32 archivos a un conjunto de procesos. Los procesos pueden trabajar sobre imágenes independientes y el controlador conserva el orden del manifiesto. La escritura y un hash de los archivos permiten comprobar la igualdad bit a bit.

El disco, las cachés del sistema, las copias entre procesos y el número de núcleos asignados por Colab influyen en el tiempo de preparación. El monitor registra la carga por núcleo durante esta fase. Las imágenes se convierten a tensores en el host y los lotes se transfieren cuando empieza la matemática del entrenamiento.

# Laboratorio 3 - Deep Learning: reconocimiento de lenguaje de señas ASL

Proyecto reproducible para CC3084 Data Science. Construye y compara clasificadores
de imágenes estáticas del alfabeto ASL con la metodología vista en clase:
`keras.Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`,
`ImageDataGenerator`, `flow_from_directory`, SGD con momentum y EarlyStopping.

El notebook principal funciona como informe y usa resultados producidos por ejecuciones
reales. El conjunto `test` se reserva hasta que la selección por `validation` queda
guardada en `outputs/metrics/selection_before_test.json`.

## Dataset

Se utiliza [ASL Alphabet de Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet):
aproximadamente 87,000 fotografías JPG RGB de 200×200, organizadas en 29 clases
(A-Z, DELETE, NOTHING y SPACE). Las imágenes no se incluyen en Git.

Descarga automática sin duplicar una copia existente:

```bash
source .venv/bin/activate
python scripts/download_dataset.py
```

También se puede descomprimir manualmente de modo que exista una carpeta llamada
`asl_alphabet_train` dentro de `data/raw/`. El localizador admite tanto la estructura
simple como la carpeta doble distribuida por Kaggle.

## Estructura

```text
.
├── data/
│   ├── raw/                 # dataset original, ignorado por Git
│   ├── train/               # 60 %, enlaces físicos reproducibles
│   ├── validation/          # 20 %
│   ├── test/                # 20 %
│   └── custom_test/         # fotos reales, pendientes
├── notebooks/
│   └── Lab03_ASL.ipynb      # informe ejecutado
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/              # modelos pesados, ignorados por Git
├── scripts/
│   ├── download_dataset.py
│   └── generate_notebook.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── experiments.py
│   ├── visualization.py
│   └── run_lab.py
├── requirements.txt
└── README.md
```

## Instalación

En macOS Apple Silicon se recomienda Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducir todo

Desde la raíz del proyecto:

```bash
source .venv/bin/activate
python scripts/download_dataset.py
python -m src.run_lab
python scripts/generate_notebook.py
jupyter nbconvert --to notebook --execute notebooks/Lab03_ASL.ipynb \
  --output Lab03_ASL.ipynb --output-dir notebooks \
  --ExecutePreprocessor.timeout=-1
```

La primera ejecución audita las 87,000 imágenes, crea una submuestra aleatoria de
600 por clase, genera los splits 60/20/20, entrena todos los modelos y escribe
métricas/gráficos. Las ejecuciones posteriores reutilizan artefactos válidos. Para
reentrenar deliberadamente, desde Python:

```python
from src.run_lab import run_complete_lab
run_complete_lab(force=True)
```

Configuración central: semilla 123, RGB 64×64, batch 64, 600 imágenes por clase.
El entrenamiento usa solo `data/train`; `validation` selecciona configuraciones;
`test` se consulta al final. Augmentation se aplica solo al generador de train y no
incluye `horizontal_flip`.

## Modelos y evaluación

Se ejecutan:

- CNN Modelo 1: una convolución, SGD 0.01.
- CNN Modelo 2: dos convoluciones, SGD 0.01.
- CNN con Dropout 0.1, SGD 0.001.
- CNN con Dropout 0.2, SGD 0.001.
- CNN con Dropout 0.1 y SGD 0.01 para aislar el efecto del learning rate.
- Mejor CNN de validación con augmentation moderado.
- Fully Connected con Dense(128), sin convoluciones.
- Fully Connected con Dense(256) y Dropout 0.1 como variación controlada.
- SVM lineal y SVM RBF, ambos con `StandardScaler` y PCA(100) ajustados solo en train.

Las métricas consolidadas están en `outputs/metrics/final_comparison.csv`. Los
classification reports, matrices de confusión, manifiesto de splits, prueba de no
overlap, históricos y tiempos quedan en la misma carpeta. El mejor modelo Keras se
guarda como `outputs/models/best_asl_model.keras` y se valida cargándolo de nuevo.

## Predicción de una fotografía

```python
from tensorflow import keras
from src.evaluation import predecir_imagen
from src.preprocessing import build_generators

model = keras.models.load_model("outputs/models/best_asl_model.keras")
_, _, test_it = build_generators(augmentation=False, shuffle_train=False)
predecir_imagen("ruta/a/A_01.jpg", model, test_it.class_indices)
```

La función convierte a RGB, redimensiona a 64×64, normaliza a [0,1], agrega la
dimensión de batch y presenta clase y confianza.

## Fotografías propias

PENDIENTE: fotografías reales de los integrantes. Colócalas aquí:

```text
data/custom_test/<nombre_del_integrante>/
```

Cada integrante debe agregar al menos cinco letras distintas con nombres como
`A_01.jpg`, `B_01.jpg`, `C_01.jpg`, `D_01.jpg` y `E_01.jpg`. Después ejecuta de
nuevo `python -m src.run_lab`; se crearán `custom_test_predictions.csv` y la
accuracy de las fotos etiquetadas. No se usan imágenes del dataset como sustituto.

## Alcance

Este laboratorio clasifica fotografías estáticas. No es un traductor ASL completo:
un producto real necesita captura de video, detección y seguimiento de manos,
modelado temporal de señas dinámicas, composición de palabras, contexto, manejo de
incertidumbre y evaluación con usuarios diversos.

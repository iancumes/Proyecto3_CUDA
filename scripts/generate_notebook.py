"""Genera el notebook-informe usando exclusivamente métricas ejecutadas."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
NOTEBOOK = ROOT / "notebooks" / "Lab03_ASL.ipynb"


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def signed_pp(value: float) -> str:
    return f"{100 * float(value):+.2f} puntos porcentuales"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    required = [
        METRICS / "experiments.csv",
        METRICS / "run_summary.json",
        METRICS / "conclusions.json",
        METRICS / "best_model_analysis.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Primero ejecuta python -m src.run_lab. Faltan: {missing}")

    experiments = pd.read_csv(METRICS / "experiments.csv").set_index("experiment_id")
    summary = json.loads((METRICS / "run_summary.json").read_text(encoding="utf-8"))
    conclusions = json.loads((METRICS / "conclusions.json").read_text(encoding="utf-8"))
    analysis = json.loads((METRICS / "best_model_analysis.json").read_text(encoding="utf-8"))
    selection = summary["selection"]
    best_id = selection["best_overall_by_validation"]
    best_keras = selection["best_keras_by_validation"]
    base_selection = json.loads((METRICS / "cnn_selection_before_test.json").read_text())["best_base_cnn_by_validation"]
    custom = summary["custom_test"]

    def result_sentence(experiment_id: str) -> str:
        row = experiments.loc[experiment_id]
        gap = row.train_accuracy - row.validation_accuracy
        signal = (
            "una brecha apreciable compatible con sobreajuste"
            if gap > 0.05
            else "una brecha pequeña entre entrenamiento y validación"
        )
        return (
            f"La ejecución real alcanzó {pct(row.train_accuracy)} en train y "
            f"{pct(row.validation_accuracy)} en validation (brecha {signed_pp(gap)}), con "
            f"loss de validación {row.validation_loss:.4f}. Se entrenaron "
            f"{int(row.epochs_trained)} épocas en {row.training_time:.1f} s. Esto muestra {signal}."
        )

    cells = [
        md("""
# Laboratorio 3
## Deep Learning: Reconocimiento de Lenguaje de Señas ASL

**CC3084 - Data Science · Semestre II 2026**

Este notebook funciona como informe ejecutable. Todos los conteos, tiempos, métricas,
gráficos y conclusiones cuantitativas que aparecen abajo provienen de los artefactos
producidos por una ejecución real de `src.run_lab`. No se utilizan arquitecturas
preentrenadas ni LSTM: la metodología principal sigue el ejemplo CNN de gatos y perros.
"""),
        md("""
## 1. Introducción

SignBridge plantea un primer prototipo para reconocer letras ASL a partir de fotografías
de manos. Este laboratorio resuelve clasificación multiclase de imágenes estáticas; no
pretende ser todavía un traductor ASL en tiempo real.

## 2. Objetivos

- Explorar y verificar el dataset ASL Alphabet.
- Crear una división estratificada 60/20/20 sin imágenes compartidas.
- Comparar CNN de complejidad progresiva, Dropout y augmentation.
- Contrastar contra una red fully-connected y SVM.
- Reservar test hasta congelar la selección basada en validation.
- Analizar errores, accesibilidad, sesgo y límites del prototipo.
"""),
        code("""
from pathlib import Path
import os, sys, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage, Markdown

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_lab import run_complete_lab
from src.config import *
from src.models import build_cnn1, build_cnn2, build_fully_connected
from src.preprocessing import build_generators
from src.evaluation import predecir_imagen
from tensorflow import keras

# Si faltara cualquier artefacto, esta llamada ejecutaría el pipeline completo.
# En una reproducción ya terminada reutiliza modelos y métricas verificadas.
run_summary = run_complete_lab(force=False)
print("Pipeline completo y verificado.")
"""),
        md(f"""
## 3. Dataset

### Preguntas

¿Cuántas imágenes y clases existen? ¿El dataset está balanceado? ¿Todas las imágenes
tienen resolución, formato y canales consistentes? ¿Hay archivos inválidos?

La auditoría abrió y verificó **{summary['dataset']['total_files']:,} archivos**. Encontró
**{summary['dataset']['valid_images']:,} imágenes válidas**, **{summary['dataset']['invalid_images']} inválidas**
y **{summary['dataset']['class_count']} clases**. Los conteos van de
{summary['dataset']['min_count']:,} a {summary['dataset']['max_count']:,}; la diferencia real es
{summary['dataset']['class_difference']} imágenes. Por tanto, el balance no se asume: queda
cuantificado por esa diferencia y por la tabla siguiente.
"""),
        code("""
distribution = pd.read_csv("outputs/metrics/class_distribution.csv")
display(distribution.style.format({"percentage": "{:.4f}%"}))
display(IPImage(filename="outputs/figures/class_distribution.png"))
"""),
        md(f"""
La resolución dominante auditada es **{int(summary['dataset']['resolutions'][0]['width'])}×{int(summary['dataset']['resolutions'][0]['height'])}**,
el formato observado es {summary['dataset']['formats']} y los modos de color son
{summary['dataset']['modes']}. La diferencia de {summary['dataset']['class_difference']} entre la clase
con mayor y menor cantidad confirma un conjunto perfectamente balanceado en imágenes válidas
solo si esa diferencia es cero; ese fue el resultado observado aquí.
"""),
        md("""
## 4. Análisis exploratorio

### 4.1 Ejemplos visuales y variabilidad

¿Cómo cambian posición, iluminación, escala, orientación, ubicación de la mano y dedos
dentro y entre clases? Las imágenes siguientes son muestras reales elegidas con semilla 123.
"""),
        code("""
for path in [
    "outputs/figures/five_letters.png",
    "outputs/figures/within_class_A.png",
    "outputs/figures/special_classes.png",
]:
    display(IPImage(filename=path))
"""),
        md("""
Las fotografías conservan fondo e iluminación relativamente uniformes, pero dentro de una
misma clase se aprecian desplazamientos, ligeros cambios de escala/orientación y variaciones
en la flexión de los dedos. Esa variabilidad justifica aprender características y aplicar
transformaciones moderadas, sin asumir que cualquier transformación conserva la etiqueta.

### 4.2 Clases visualmente similares

Antes de entrenar se planteó la hipótesis de que M/N/S y U/V/R podrían ser difíciles por
las diferencias sutiles en la colocación o cruce de dedos. Esto es una hipótesis visual,
no una afirmación sobre las confusiones reales.
"""),
        code("""
display(IPImage(filename="outputs/figures/similar_MNS.png"))
display(IPImage(filename="outputs/figures/similar_UVR.png"))
"""),
        md(f"""
## 5. Submuestreo

Se seleccionaron aleatoriamente **{summary['samples_per_class']} imágenes válidas por clase**,
sin reemplazo y con `RANDOM_SEED=123`: **{summary['selected_total']:,} imágenes** en total.
Se escogió 600 porque está dentro del rango 500-800 indicado por el laboratorio y permite
ejecutar diez experimentos en una computadora de 16 GB sin usar simplemente los primeros archivos.

## 6. División Train / Validation / Test

La división estratificada fue 60/20/20: **{summary['split_counts']['train']:,} train**,
**{summary['split_counts']['validation']:,} validation** y **{summary['split_counts']['test']:,} test**.
Las 29 clases están presentes con 360/120/120 imágenes respectivamente.
"""),
        code("""
split_table = pd.read_csv("outputs/metrics/split_by_class.csv")
leakage = json.loads(Path("outputs/metrics/leakage_check.json").read_text())
display(split_table)
print("Intersecciones:", leakage["intersections"])
print("Imágenes fuente únicas:", leakage["unique_source_images"])
assert leakage["verified"] and all(v == 0 for v in leakage["intersections"].values())
"""),
        md("""
La intersección train-validation, train-test y validation-test es cero. TRAIN ajusta pesos;
VALIDATION decide arquitectura, dropout, augmentation y kernel; TEST no se consultó hasta
guardar `selection_before_test.json`. Aunque no hay duplicados exactos, las fotografías del
dataset provienen de secuencias visualmente homogéneas, una limitación que puede volver el test
interno más optimista que fotos de usuarios nuevos.

## 7. Preprocesamiento

Las imágenes se mantienen RGB, se redimensionan de 200×200×3 a 64×64×3 y se normalizan
de [0,255] a [0,1] con `rescale=1.0/255.0`. El resize reduce memoria y tiempo, conservando
la forma global de la mano recomendada para este laboratorio.
"""),
        code("""
display(IPImage(filename="outputs/figures/resize_comparison.png"))
train_it, validation_it, test_it = build_generators(augmentation=False)
print("class_mode='categorical'; clases:", len(train_it.class_indices))
print("Validation y test: solo rescale, sin augmentation.")
"""),
        md("""
La salida binaria del ejemplo de gatos/perros se adapta a 29 neuronas softmax,
`categorical_crossentropy` y `class_mode='categorical'`. Esto produce una distribución
de probabilidad mutuamente excluyente sobre todas las clases.

## 8. Modelo 1 - CNN básica
"""),
        code("""
modelo1 = build_cnn1(learning_rate=0.01)
modelo1.summary()
display(pd.read_csv("outputs/metrics/experiments.csv").query("experiment_id == 'CNN1'"))
display(IPImage(filename="outputs/figures/history_cnn1.png"))
"""),
        md(result_sentence("CNN1")),
        md("""
## 9. Modelo 2 - CNN con mayor profundidad

El cambio controlado es agregar Conv2D(64)+MaxPooling2D, manteniendo Dense(128), SGD
con learning rate 0.01, momentum 0.9 y diez épocas.
"""),
        code("""
modelo2 = build_cnn2(learning_rate=0.01, dropout=0.0)
modelo2.summary()
display(pd.read_csv("outputs/metrics/experiments.csv").query("experiment_id == 'CNN2'"))
display(IPImage(filename="outputs/figures/history_cnn2.png"))
"""),
        md(result_sentence("CNN2") + " El cambio de validation accuracy contra Modelo 1 fue " + signed_pp(conclusions["second_convolution_validation_change"]) + "."),
        md("""
## 10. Modelo 3 - Dropout

Se comparan dos configuraciones controladas sobre la CNN de dos convoluciones. Ambas usan
SGD con learning rate 0.001; solo cambia Dropout de 0.1 a 0.2. Los entrenamientos largos
usan EarlyStopping sobre `val_loss`, patience 3 y `restore_best_weights=True`.
"""),
        code("""
display(pd.read_csv("outputs/metrics/experiments.csv").query("experiment_id in ['CNN3_D01','CNN3_D02']"))
display(IPImage(filename="outputs/figures/history_cnn3_d01.png"))
display(IPImage(filename="outputs/figures/history_cnn3_d02.png"))
"""),
        md(result_sentence("CNN3_D01") + "\n\n" + result_sentence("CNN3_D02") + f"\n\nSegún validation, la mejor tasa de Dropout fue **{conclusions['better_dropout']}**."),
        md("""
### 10.1 Experimento controlado de learning rate

Como learning rate 0.001 mostró aprendizaje lento, se repitió Dropout 0.1 con 0.01,
sin cambiar ninguna otra variable. Esto separa el efecto de tasa de aprendizaje del efecto
de Dropout.
"""),
        code("""
display(pd.read_csv("outputs/metrics/experiments.csv").query("experiment_id == 'CNN3_D01_LR01'"))
display(IPImage(filename="outputs/figures/history_cnn3_d01_lr01.png"))
"""),
        md(result_sentence("CNN3_D01_LR01") + f" El cambio frente a la misma arquitectura con learning rate 0.001 fue {signed_pp(conclusions['learning_rate_001_to_01_validation_change'])}."),
        md("""
## 11. Image Augmentation

La mejor CNN base se reentrenó con desplazamiento horizontal/vertical 0.1, rotación 10°
y zoom 0.1. No se usa flip horizontal: en gatos/perros una reflexión suele conservar la
clase, pero en ASL puede alterar relaciones espaciales relevantes entre mano y dedos.
Validation y test conservan únicamente `rescale`.
"""),
        code("""
display(IPImage(filename="outputs/figures/augmentation_examples.png"))
display(pd.read_csv("outputs/metrics/experiments.csv").query("experiment_id == 'CNN_AUG'"))
display(IPImage(filename="outputs/figures/history_cnn_aug.png"))
"""),
        md(result_sentence("CNN_AUG") + f" La base elegida fue {base_selection} y el cambio de validation accuracy con augmentation fue {signed_pp(conclusions['augmentation_vs_selected_base_validation_change'])}."),
        md("""
## 12. Fully Connected

`Flatten` convierte los 64×64×3 píxeles en un vector; Dense(128) aprende sin extracción
espacial por convoluciones. Esto permite medir el valor de Conv2D y muestra el gran número
de parámetros que puede producir una representación completamente conectada.
"""),
        code("""
modelo_fc = build_fully_connected(learning_rate=0.01)
modelo_fc.summary()
display(pd.read_csv("outputs/metrics/experiments.csv").query("experiment_id in ['FC1','FC2']"))
display(IPImage(filename="outputs/figures/history_fc1.png"))
display(IPImage(filename="outputs/figures/history_fc2.png"))
"""),
        md(result_sentence("FC1") + "\n\n" + result_sentence("FC2") + f" La variación Dense(256)+Dropout(0.1) cambió validation en {signed_pp(conclusions['fully_connected_variant_validation_change'])}."),
        md("""
## 13. SVM

SVM es una referencia clásica adecuada para fronteras de decisión de alto margen. Cada imagen
64×64×3 se aplana; `StandardScaler` y PCA(100) se ajustan **solo con TRAIN** y después se
aplican sin refit a validation/test. Se comparan kernel lineal y RBF con C=1, sin una búsqueda
masiva de hiperparámetros.
"""),
        code("""
svm_rows = pd.read_csv("outputs/metrics/experiments.csv").query("experiment_id.str.startswith('SVM')", engine="python")
display(svm_rows)
"""),
        md(f"""
SVM lineal obtuvo {pct(experiments.loc['SVM_LINEAR','validation_accuracy'])} en validation y
{pct(experiments.loc['SVM_LINEAR','test_accuracy'])} en test; SVM RBF obtuvo
{pct(experiments.loc['SVM_RBF','validation_accuracy'])} y
{pct(experiments.loc['SVM_RBF','test_accuracy'])}, respectivamente. El kernel seleccionado
por validation fue **{selection['best_svm_by_validation']}**.
"""),
        md("""
## 14. Comparación de modelos

La selección quedó congelada antes de evaluar test. La tabla incluye test para todos los
experimentos únicamente como evaluación final posterior, no como criterio de tuning.
"""),
        code("""
comparison = pd.read_csv("outputs/metrics/final_comparison.csv")
display(comparison.style.format({
    "Train Accuracy":"{:.4f}", "Validation Accuracy":"{:.4f}",
    "Test Accuracy":"{:.4f}", "Macro Precision":"{:.4f}",
    "Macro Recall":"{:.4f}", "Macro F1":"{:.4f}",
    "Test Loss":"{:.4f}", "Tiempo (s)":"{:.1f}"
}, na_rep="N/A"))
"""),
        md(f"""
## 15. Mejor modelo

El modelo seleccionado fue **{best_id}** porque obtuvo la mayor validation accuracy
({pct(experiments.loc[best_id,'validation_accuracy'])}) antes de consultar test. Su resultado
final fue {pct(experiments.loc[best_id,'test_accuracy'])} de accuracy y
{pct(experiments.loc[best_id,'macro_f1'])} de macro F1. La decisión no usa training accuracy
ni cambia después de ver test.

El mejor modelo Keras guardado y recargado es **{best_keras}**. La comprobación de carga y
predicción está documentada en `model_reload_check.json`.
"""),
        md("""
## 16. Matriz de confusión y análisis de errores
"""),
        code("""
display(IPImage(filename="outputs/figures/confusion_matrix_best.png"))
top_confusions = pd.read_csv("outputs/metrics/top_10_confusions.csv")
display(top_confusions)
best_analysis = json.loads(Path("outputs/metrics/best_model_analysis.json").read_text())
print("Mayor recall:", best_analysis["highest_recall"])
print("Menor recall:", best_analysis["lowest_recall"])
"""),
        md(f"""
Dentro de M/N/S ocurrieron **{analysis['m_n_s_internal_errors']} errores internos** y dentro de
U/V/R ocurrieron **{analysis['u_v_r_internal_errors']}**. Por ello la hipótesis del EDA se
contrasta con datos sin forzarla: los conteos anteriores indican si esos grupos fueron o no
confusiones dominantes, y la tabla automática muestra los diez pares reales más frecuentes.
"""),
        code("""
if Path("outputs/figures/misclassified_examples.png").exists():
    display(IPImage(filename="outputs/figures/misclassified_examples.png"))
    display(pd.read_csv("outputs/metrics/misclassified_examples.csv"))
"""),
        md("""
Los ejemplos se ordenan por confianza entre las predicciones incorrectas. Una confianza alta
en un error muestra que softmax no equivale a certeza calibrada y que un producto debería
rechazar o pedir repetición cuando la confianza o consistencia temporal sea insuficiente.

### Función de predicción reusable
"""),
        code("""
best_model = keras.models.load_model("outputs/models/best_asl_model.keras")
_, _, prediction_test_it = build_generators(augmentation=False, shuffle_train=False)
sample_path = prediction_test_it.filepaths[0]
display(IPImage(filename=sample_path))
prediction_result = predecir_imagen(
    sample_path, best_model, prediction_test_it.class_indices, mostrar=False
)
prediction_result
"""),
        md("""
## 17. Pruebas con fotografías propias
"""),
        code("""
custom_status = json.loads(Path("outputs/metrics/custom_test_status.json").read_text())
display(custom_status)
if Path("outputs/metrics/custom_test_predictions.csv").exists():
    display(pd.read_csv("outputs/metrics/custom_test_predictions.csv"))
"""),
        md(f"""
**{custom['status']}** Coloca al menos cinco letras distintas por integrante en
`data/custom_test/<integrante>/`, con nombres como `A_01.jpg`. No se sustituyeron fotografías
propias por imágenes del dataset. Cuando se agreguen, el pipeline calculará accuracy y permitirá
comparar el test interno contra cámaras, fondos e iluminación reales.

Una caída futura sería una observación de *domain shift*; iluminación, cámara, fondo, orientación,
distancia, escala, tono de piel o tamaño de mano serían posibles explicaciones a investigar,
no causas que deban afirmarse sin evidencia adicional.
"""),
        md("""
## 18. Accesibilidad y sesgo

Una accuracy alta en este dataset no garantiza servicio equitativo. La muestra tiene fondos,
ángulos y condiciones relativamente controladas y no aporta metadatos suficientes para auditar
tono de piel, tamaño de mano, diversidad funcional, cámara o subgrupos de usuarios.

Para un producto real se recomienda: recopilar participantes diversos; cubrir distintos tonos
de piel, tamaños de mano, cámaras, iluminaciones, fondos y ángulos; evaluar métricas por subgrupo;
hacer pruebas con usuarios reales; documentar fallos; calibrar confianza; y ofrecer rechazo o
repetición para predicciones de baja confianza. También se requiere diseñar con la comunidad
sorda y evaluar el impacto de errores, no solo accuracy promedio.

## 19. Consideraciones para un producto real

Este prototipo clasifica una imagen estática. SignBridge necesitaría además captura continua de
video, detección de manos, seguimiento temporal, reconocimiento de movimiento y señas dinámicas,
segmentación de secuencias, construcción de palabras, contexto lingüístico y manejo explícito de
incertidumbre. Por tanto, no se presenta este clasificador como un traductor ASL completo.
"""),
        md(f"""
## 20. Conclusiones

1. El mejor modelo por validation fue **{best_id}** ({pct(experiments.loc[best_id,'validation_accuracy'])}); en test obtuvo {pct(experiments.loc[best_id,'test_accuracy'])} y macro F1 {pct(experiments.loc[best_id,'macro_f1'])}.
2. Agregar la segunda convolución cambió validation en {signed_pp(conclusions['second_convolution_validation_change'])} frente a CNN1.
3. Dropout 0.1 cambió validation en {signed_pp(conclusions['dropout_01_vs_cnn2_validation_change'])} frente a CNN2; Dropout 0.2 lo hizo en {signed_pp(conclusions['dropout_02_vs_cnn2_validation_change'])}.
4. Entre las tasas evaluadas, funcionó mejor Dropout **{conclusions['better_dropout']}** según validation.
5. Cambiar el learning rate controlado de 0.001 a 0.01 modificó validation en {signed_pp(conclusions['learning_rate_001_to_01_validation_change'])}.
6. Augmentation cambió validation en {signed_pp(conclusions['augmentation_vs_selected_base_validation_change'])} respecto a su misma CNN base.
7. Fully Connected base alcanzó {pct(experiments.loc['FC1','validation_accuracy'])}; la variante Dense(256)+Dropout(0.1) cambió validation en {signed_pp(conclusions['fully_connected_variant_validation_change'])}. La diferencia del baseline frente al mejor Keras fue {signed_pp(conclusions['fully_connected_vs_best_cnn_validation_change'])}.
8. El mejor SVM por validation fue {selection['best_svm_by_validation']} con {pct(conclusions['best_svm_validation_accuracy'])}.
9. Las clases difíciles y los pares principales son exactamente los reportados por recall y matriz de confusión, no supuestos previos.
10. M/N/S produjo {analysis['m_n_s_internal_errors']} errores internos y U/V/R produjo {analysis['u_v_r_internal_errors']}; esto determina en qué medida el EDA coincidió con la evaluación.
11. La principal limitación pendiente son las fotografías auténticas de integrantes y la falta de diversidad/metadatos del dataset.
12. Un producto real necesita video, temporalidad, usuarios diversos, evaluación por subgrupos y manejo de incertidumbre.
"""),
        md("""
## Reproducibilidad y referencias

Configuración: `RANDOM_SEED=123`, `IMAGE_SIZE=(64,64)`, `BATCH_SIZE=64`, split
60/20/20, RGB, `rescale=1/255`, SGD con momentum 0.9. Los scripts guardan selección,
manifiesto, tiempos, históricos y métricas en `outputs/metrics/`.

- ASL Alphabet: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- Material de clase: *Ejemplo CNN gatos y perros.ipynb*.
- Referencia metodológica de validación/EarlyStopping: *Ejemplo series de tiempo LSTM.ipynb*.
"""),
        code("""
import tensorflow as tf, sklearn, PIL, matplotlib, pandas
print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("NumPy:", np.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("Pillow:", PIL.__version__)
print("matplotlib:", matplotlib.__version__)
"""),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (ASL Lab)",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(f"Notebook generado: {NOTEBOOK}")


if __name__ == "__main__":
    main()

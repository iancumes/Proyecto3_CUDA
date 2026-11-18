"""Crea el informe de rendimiento a partir de registros de una ejecución."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PHASES = {
    "prep_s": "Preparación",
    "training_s": "Entrenamiento",
    "validation_s": "Validación",
    "total_s": "Total",
}
BLUE = "#164b64"
TEAL = "#087f75"


def load_csv(path):
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    return []


def plot_phases(summary, directory):
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    modes = [mode for mode in ("cpu", "cuda") if f"{mode}_mean_s" in summary[0]]
    keys = [PHASES[row["phase"]] for row in summary]
    bottom = list(range(len(keys)))
    for index, mode in enumerate(modes):
        avg = [float(row[f"{mode}_mean_s"]) for row in summary]
        sd = [float(row[f"{mode}_sd_s"] or 0) for row in summary]
        positions = [value + index * 0.34 for value in bottom]
        ax.barh(positions, avg, height=0.32, xerr=sd, color=(BLUE, TEAL)[index],
                label="CPU" if mode == "cpu" else "CUDA", capsize=3)
    ax.set_yticks([value + (len(modes) - 1) * 0.17 for value in bottom], keys)
    ax.invert_yaxis()
    ax.set_xlabel("Tiempo medio (s)")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    path = directory / "tiempos_por_fase.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_resources(results):
    figures = []
    for suffix in ("cpu_prep", "cuda_prep", "cpu_train", "cuda_train"):
        candidates = sorted(results.glob(f"*_{suffix}_resources.jsonl"))
        if not candidates:
            continue
        samples = [json.loads(line) for line in candidates[0].read_text(encoding="utf-8").splitlines()]
        if not samples:
            continue
        start = datetime.fromisoformat(samples[0]["utc"])
        seconds = [(datetime.fromisoformat(row["utc"]) - start).total_seconds() for row in samples]
        fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
        for index in range(len(samples[0]["cpu_pct_by_core"])):
            if len(samples[0]["cpu_pct_by_core"]) > 8 and index >= 8:
                break
            ax.plot(seconds, [row["cpu_pct_by_core"][index] for row in samples],
                    label=f"CPU {index}", linewidth=1, alpha=0.6, marker="." if len(samples) == 1 else None)
        gpu_rows = [row for row in samples if "gpu_util_pct" in row]
        if gpu_rows:
            gpu_seconds = [(datetime.fromisoformat(row["utc"]) - start).total_seconds() for row in gpu_rows]
            ax.plot(gpu_seconds, [float(row["gpu_util_pct"]) for row in gpu_rows],
                    color=TEAL, label="GPU activa", linewidth=1.8, marker=".")
            ax.plot(gpu_seconds, [100 * float(row["gpu_memory_mib"]) / float(row["gpu_total_mib"])
                                  for row in gpu_rows],
                    color=BLUE, label="Memoria GPU ocupada", linewidth=1.6, linestyle="--")
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Segundos desde la primera muestra")
        ax.set_ylabel("Porcentaje")
        ax.set_title(f'{samples[0]["phase"]} · {start:%H:%M:%S} UTC', loc="left")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncols=4, fontsize=7)
        path = results / f"monitor_{suffix}.png"
        fig.savefig(path, dpi=155)
        plt.close(fig)
        figures.append(path)
    return figures


def build_report(results, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = load_csv(results / "summary.csv")
    runs = load_csv(results / "runs.csv")
    complete = bool(summary and runs and "cuda_mean_s" in summary[0]
                    and all(int(row["examples"]) == 10440 for row in runs))
    env_path = results / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor(BLUE)
    styles["Heading2"].textColor = colors.HexColor(BLUE)
    styles["BodyText"].spaceAfter = 9
    styles.add(ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=9, leading=12, spaceAfter=0))
    styles.add(ParagraphStyle("White", parent=styles["Cell"], textColor=colors.white))
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))

    def table(headers, rows, widths):
        headings = [Paragraph(escape(value), styles["White"]) for value in headers]
        body = [[Paragraph(escape(str(value)), styles["Cell"]) for value in row] for row in rows]
        element = Table([headings, *body], colWidths=[width * cm for width in widths], repeatRows=1,
                        hAlign="LEFT")
        element.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eef4f6"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([element, Spacer(1, 12)])

    paragraph("Proyecto 3 CUDA", "Title")
    paragraph("Reconocimiento de señas ASL con CNN2", "Heading2")
    paragraph("Universidad del Valle de Guatemala · CC3069<br/>Ian Cumes, Javier Valladares y Nery Molina")
    if not complete:
        paragraph("<b>Informe técnico preliminar.</b> La comparación CPU–CUDA y las capturas de GPU "
                  "se completan con los registros de Colab.")
    paragraph("1. Solución del laboratorio", "Heading2")
    paragraph("El laboratorio original compara varias redes y SVM. Se reutilizó CNN2, una red sin "
              "dropout ni aumento de datos en este experimento. La selección fija 600 imágenes por "
              "cada una de 29 clases y separa 60 %, 20 % y 20 % con semilla 123. El test queda reservado.")
    table(["Operación", "Función"], [
        ("01 · Selección y división", "Audita archivos y produce un manifiesto estratificado."),
        ("02 · Preparación", "Decodifica cada JPEG, convierte a RGB, redimensiona a 64 × 64 y normaliza."),
        ("03 · Forward", "Aplica convoluciones y pooling, aplana activaciones y predice 29 clases."),
        ("04 · Retropropagación", "Calcula la pérdida y sus gradientes para entrenar la red."),
        ("05 · Optimización", "Aplica SGD con momentum a los parámetros."),
        ("06 · Validación", "Calcula pérdida y exactitud con la partición reservada para validación."),
    ], [5, 11.5])
    table(["Operación repetitiva", "Datos", "Repeticiones en ASL completo"], [
        ("Preparar imágenes", "JPEG de entrenamiento y validación", "13,920 por preparación"),
        ("Forward y gradientes", "Lotes de hasta 64 imágenes", "164 pasos por época"),
        ("SGD con momentum", "Pesos y estado del optimizador", "2,120,413 parámetros por paso"),
        ("Validar", "3,480 imágenes", "55 lotes por época"),
    ], [5, 5.5, 6])
    story.append(PageBreak())
    paragraph("2. División del trabajo y diseño CUDA", "Heading2")
    paragraph("Leer archivos, decodificar JPEG y preparar etiquetas son tareas independientes por imagen. "
              "El CPU las distribuye en grupos de 32 mediante un conjunto de procesos. La referencia usa la "
              "misma transformación en serie. El hash de imagen y etiqueta comprueba que los arrays coinciden.")
    paragraph("En GPU, TensorFlow ejecuta convoluciones, capas densas y sus gradientes. El kernel de Numba-CUDA "
              "actualiza cada parámetro con v = 0.9v - 0.01g y w = w + v. Se programan 256 hilos por bloque y "
              "ceil(N/256) bloques por tensor. Cada hilo comprueba su índice y escribe su propia posición. "
              "DLPack y CUDA Array Interface intercambian vistas de memoria GPU entre TensorFlow, CuPy y Numba.")
    paragraph("3. Metodología y resultados", "Heading2")
    paragraph("El protocolo ejecuta tres pares de dos épocas con lotes de 64, precisión float32 y los mismos "
              "pesos iniciales y orden de lotes. Se calienta cada modo y se reinician los pesos y velocidades. "
              "El tiempo total suma preparación, entrenamiento y validación. El entrenamiento GPU incluye "
              "transferencia de lotes, sincronización y lectura de la pérdida al final de cada época.")
    if env:
        paragraph(f'<b>Entorno medido.</b> {escape(env["cpu"] or "CPU")} · {env["available_cores"]} núcleos lógicos '
                  f'· Python {env["python"]} · {escape(env.get("gpu", "ejecución local en CPU"))}.')
    if summary:
        if int(runs[0]["examples"]) != 10440:
            paragraph(f'<b>Prueba funcional.</b> La muestra contiene {runs[0]["examples"]} imágenes '
                      "de entrenamiento. Las cifras de esta prueba solo verifican el flujo con una muestra pequeña.")
        displayed_modes = ["cpu", "cuda"] if complete else ["cpu"]
        headers = ["Fase", *[f"{mode.upper()} media ± DE (s)" for mode in displayed_modes]]
        if complete:
            headers.append("Speedup")
        rows = []
        for row in summary:
            values = [PHASES[row["phase"]]]
            for mode in displayed_modes:
                avg = float(row[f"{mode}_mean_s"])
                sd = row[f"{mode}_sd_s"]
                values.append(f"{avg:.3f} ± {float(sd):.3f}" if sd else f"{avg:.3f}")
            if complete:
                values.append(f'{float(row["speedup"]):.2f}×')
            rows.append(values)
        table(headers, rows, [16.5 / len(headers)] * len(headers))
        story.append(Image(str(plot_phases(summary, results)), width=16 * cm, height=7.2 * cm))
    if complete:
        story.append(PageBreak())
        paragraph("4. Interpretación y evidencia", "Heading2")
        cuda_rows = [row for row in runs if row["mode"] == "cuda"]
        by_phase = {row["phase"]: float(row["speedup"]) for row in summary}
        descriptions = [
            f"{PHASES[key]} fue {(factor if factor >= 1 else 1 / factor):.2f} veces "
            f"{'más rápida' if factor >= 1 else 'más lenta'}"
            for key, factor in by_phase.items() if key in ("prep_s", "training_s", "total_s")
        ]
        paragraph(". ".join(descriptions) + ".")
        avg_train = mean(float(row["training_s"]) for row in cuda_rows)
        avg_transfer = mean(float(row["h2d_s"]) + float(row["d2h_s"]) for row in cuda_rows)
        accuracy = defaultdict(list)
        for row in runs:
            accuracy[row["mode"]].append(float(row["val_accuracy"]))
        paragraph(f"Las transferencias explícitas consumieron {avg_transfer:.3f} s en promedio "
                  f"({100 * avg_transfer / avg_train:.1f} % del entrenamiento CUDA). La exactitud media "
                  f"de validación fue {100 * mean(accuracy['cpu']):.2f} % en CPU y "
                  f"{100 * mean(accuracy['cuda']):.2f} % en CUDA.")
    elif summary:
        paragraph("Estas cifras corresponden a la ejecución local CPU. La comparación de speedup requiere "
                  "repetir CPU y CUDA en el mismo entorno de Colab.")
    paragraph("La lectura de JPEG se beneficia de procesar archivos independientes, aunque depende de E/S "
              "y del costo de mover resultados entre procesos. La matemática de CNN2 reutiliza datos en "
              "convoluciones y operaciones matriciales, mientras que SGD recorre pesos independientes. "
              "Lanzamientos, sincronizaciones, transferencias y tamaño del lote condicionan el resultado.")
    resources = plot_resources(results) if results.is_dir() else []
    for index, path in enumerate(resources):
        if index % 2 == 0:
            story.append(PageBreak())
            paragraph("Monitoreo registrado durante la ejecución", "Heading2")
        story.extend([Image(str(path), width=16 * cm, height=6 * cm), Spacer(1, 8)])
    captures = sorted(results.glob("captura_*.png"))
    for path in captures:
        story.append(PageBreak())
        paragraph(f"Captura · {escape(path.stem.removeprefix('captura_'))}", "Heading2")
        image = Image(str(path))
        ratio = min(16 * cm / image.imageWidth, 20 * cm / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
        story.append(image)
    paragraph("Fuentes", "Heading2")
    paragraph('<link href="https://github.com/Javiervalladares1/Lab3Datos">Lab3Datos · código y notebook originales</link>'
              '<br/><link href="https://nvidia.github.io/numba-cuda/user/kernels.html">NVIDIA · Numba-CUDA y modelo de ejecución</link>')
    SimpleDocTemplate(str(output), title="Proyecto 3 CUDA · Informe de rendimiento",
                      author="Ian Cumes, Javier Valladares y Nery Molina",
                      leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm,
                      bottomMargin=1.7 * cm).build(story)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.results, args.output)

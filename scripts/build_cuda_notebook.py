"""Genera el cuaderno de entrega a partir de las instrucciones del proyecto."""

import json
from pathlib import Path
from textwrap import dedent


def cell(kind, text):
    base = {"cell_type": kind, "metadata": {}, "source": dedent(text).strip().splitlines(keepends=True)}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


notebook = {
    "cells": [
        cell("markdown", """
            # Proyecto 3 CUDA
            **CC3069 · Universidad del Valle de Guatemala**
            Ian Cumes · Javier Valladares · Nery Molina

            Comparamos el entrenamiento de CNN2 sobre ASL Alphabet con dos flujos.
            La referencia prepara imágenes y entrena en CPU secuencial. El flujo
            optimizado prepara imágenes en varios núcleos de CPU y entrena en GPU
            con un kernel de SGD con momentum escrito en CUDA Python.

            En Colab, selecciona **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**.
            Luego ejecuta todo. La descarga y auditoría del dataset se hacen una vez por sesión.
        """),
        cell("code", """
            import importlib.util
            import os
            from pathlib import Path
            import subprocess
            import sys

            def run(*args):
                subprocess.run(args, check=True)

            def py(*args):
                run(sys.executable, "-u", *args)

            repo = Path("/content/Proyecto3_CUDA")
            branch = "validacion-colab"
            if not repo.exists():
                run("git", "clone", "--branch", branch, "--single-branch",
                    "https://github.com/iancumes/Proyecto3_CUDA.git", str(repo))
            else:
                run("git", "-C", str(repo), "fetch", "origin", branch)
                run("git", "-C", str(repo), "switch", "--detach", "FETCH_HEAD")
            os.chdir(repo)
            run("git", "rev-parse", "HEAD")
            run("nvidia-smi")
            deps = ["numba-cuda[cu12]", "reportlab", "pytest"]
            if importlib.util.find_spec("cupy") is None:
                deps.append("cupy-cuda12x")
            if importlib.util.find_spec("tensorflow") is None:
                deps.append("tensorflow[and-cuda]")
            py("-m", "pip", "install", "-q", *deps)
        """),
        cell("markdown", """
            ## 1. Preparación de imágenes · CPU

            El manifiesto conserva los 29 grupos del laboratorio. El redimensionado
            RGB a 64 × 64 y la normalización usan la misma función en ambas
            versiones. Los workers cambian con el modo. El SHA-256 de los arrays
            confirma que ambas preparaciones entregan los mismos tensores.
        """),
        cell("code", """
            from IPython.display import Code, display

            for source in ("p3/data.py", "p3/data_parallel.py"):
                display(Code(filename=source, language="python"))
            py("scripts/download_dataset.py")
            py("-m", "p3.data", "--setup")
        """),
        cell("markdown", """
            ## 2. Entrenamiento · CPU y CUDA

            CNN2 es la red original de Lab3Datos. El kernel CUDA implementa la
            actualización del peso y del momentum. La activación, el forward, el
            cálculo de gradientes y la salida de 29 clases usan TensorFlow.
        """),
        cell("code", """
            for source in ("src/models.py", "p3/train.py", "p3/cuda_kernel.py", "p3/cuda_train.py"):
                display(Code(filename=source, language="python"))
            py("-m", "pytest", "-q", "tests")
        """),
        cell("markdown", """
            ## 3. Benchmark y evidencia

            Se miden tres pares por dos épocas, con calentamiento, lotes de 64 y
            pesos iniciales iguales. La salida en vivo muestra CPU por núcleo y
            utilización y memoria de GPU. Captura la pantalla durante preparación,
            entrenamiento secuencial y entrenamiento CUDA para anexarla al PDF.
        """),
        cell("code", """
            from datetime import datetime, timezone
            import pandas as pd

            label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = Path("outputs") / f"p3_{label}"
            py("-m", "p3.benchmark", "--out", str(output), "--gpu-first")
            display(pd.read_csv(output / "runs.csv"))
            display(pd.read_csv(output / "summary.csv"))
        """),
        cell("markdown", """
            ## 4. Informe, gráficos y exportación

            La descarga reúne PDF, registros y gráficos. Guarda también una copia del
            cuaderno ejecutado desde **Archivo → Descargar → Descargar .ipynb**.
        """),
        cell("code", """
            from zipfile import ZipFile, ZIP_DEFLATED
            from google.colab import files
            from IPython.display import Image

            pdf = output / "Informe_Proyecto3_CUDA.pdf"
            py("-m", "p3.report", "--results", str(output), "--output", str(pdf))
            for png in sorted(output.glob("*.png")):
                display(Image(filename=str(png)))
            export = output.with_suffix(".zip")
            with ZipFile(export, "w", ZIP_DEFLATED) as archive:
                for path in sorted(output.iterdir()):
                    if path.is_file():
                        archive.write(path, path.name)
            files.download(str(export))
        """),
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
Path("Proyecto3_CUDA.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

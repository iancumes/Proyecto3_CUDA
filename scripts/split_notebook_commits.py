from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "Lab03_ASL.ipynb"
DESTINATION = ROOT / "notebooks" / "versiones_por_commit"

STAGES = (
    {
        "number": 1,
        "folder": "01_datos_eda_preprocesamiento",
        "last_source_cell": 14,
        "title": "Datos, EDA y preprocesamiento",
        "scope": (
            "Auditoría del dataset, análisis exploratorio, submuestreo balanceado, "
            "split 60/20/20, control de data leakage, resize y normalización."
        ),
        "commit": "feat: add reproducible ASL data pipeline and EDA",
    },
    {
        "number": 2,
        "folder": "02_modelos_y_experimentos",
        "last_source_cell": 34,
        "title": "Modelos y experimentos",
        "scope": (
            "Todo el trabajo del commit 1 más CNN1, CNN2, dropout, learning rate, "
            "image augmentation, fully connected y SVM."
        ),
        "commit": "feat: add ASL model experiments and baselines",
    },
    {
        "number": 3,
        "folder": "03_evaluacion_y_reporte_final",
        "last_source_cell": 50,
        "title": "Evaluación y reporte final",
        "scope": (
            "Todo el trabajo de los commits 1 y 2 más comparación final, selección "
            "del modelo, matriz de confusión, errores, predicción, accesibilidad y conclusiones."
        ),
        "commit": "docs: complete ASL evaluation and laboratory report",
    },
)


def stage_cell(stage: dict) -> dict:
    source = [
        f"> **Versión correspondiente al commit {stage['number']} de 3 — {stage['title']}**\n",
        ">\n",
        f"> **Alcance:** {stage['scope']}\n",
        f"> **Mensaje de commit sugerido:** `{stage['commit']}`\n",
        ">\n",
        "> Esta es una versión progresiva del notebook. El notebook final original permanece en "
        "`notebooks/Lab03_ASL.ipynb`.\n",
    ]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def readme_text(stage: dict) -> str:
    return (
        f"# Commit {stage['number']} de 3 — {stage['title']}\n\n"
        f"## Alcance\n\n{stage['scope']}\n\n"
        f"## Notebook\n\n`Lab03_ASL.ipynb`\n\n"
        f"## Mensaje de commit sugerido\n\n`{stage['commit']}`\n\n"
        "Esta carpeta representa una versión progresiva del trabajo. Los resultados incluidos "
        "provienen del notebook ejecutado; no se alteraron métricas ni salidas.\n"
    )


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_cells = notebook["cells"]

    for stage in STAGES:
        if stage["last_source_cell"] >= len(source_cells):
            raise ValueError(f"El corte del commit {stage['number']} excede el notebook fuente")

        target_dir = DESTINATION / stage["folder"]
        target_dir.mkdir(parents=True, exist_ok=True)

        snapshot = copy.deepcopy(notebook)
        selected = copy.deepcopy(source_cells[: stage["last_source_cell"] + 1])
        snapshot["cells"] = [selected[0], stage_cell(stage), *selected[1:]]
        snapshot.setdefault("metadata", {})["lab3_commit_stage"] = {
            "number": stage["number"],
            "total": len(STAGES),
            "title": stage["title"],
            "source": "notebooks/Lab03_ASL.ipynb",
            "last_source_cell": stage["last_source_cell"],
            "suggested_commit_message": stage["commit"],
        }

        notebook_path = target_dir / "Lab03_ASL.ipynb"
        notebook_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        (target_dir / "README.md").write_text(readme_text(stage), encoding="utf-8")
        print(f"Commit {stage['number']}: {notebook_path.relative_to(ROOT)} ({len(snapshot['cells'])} celdas)")


if __name__ == "__main__":
    main()

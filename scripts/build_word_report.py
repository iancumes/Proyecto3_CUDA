from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
OUTPUT = ROOT / "entrega" / "Informe_Laboratorio_3_ASL.docx"

BLUE = "2E74B5"
BLUE_DARK = "1F4D78"
TEXT = "1F2937"
GRAY = "667085"
LIGHT = "F2F4F7"
PALE_BLUE = "EAF2F8"
PALE_YELLOW = "FFF4CE"
WHITE = "FFFFFF"


def load_json(name: str):
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


dataset = load_json("dataset_summary.json")
leakage = load_json("leakage_check.json")
preprocess = load_json("preprocessing_config.json")
best_analysis = load_json("best_model_analysis.json")
reload_check = load_json("model_reload_check.json")
custom_status = load_json("custom_test_status.json")
experiments = pd.read_csv(METRICS / "experiments.csv").set_index("experiment_id")
split_by_class = pd.read_csv(METRICS / "split_by_class.csv")
top_confusions = pd.read_csv(METRICS / "top_10_confusions.csv")


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    old_grid = table._tbl.find(qn("w:tblGrid"))
    if old_grid is not None:
        table._tbl.remove(old_grid)
    grid = OxmlElement("w:tblGrid")
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    table._tbl.insert(1, grid)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            width = widths[min(idx, len(widths) - 1)]
            cell.width = Inches(width / 1440)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)


def set_alt_text(shape, title: str, description: str):
    inline = shape._inline
    doc_pr = inline.docPr
    doc_pr.set("title", title)
    doc_pr.set("descr", description)


def add_field(paragraph, instruction: str):
    run = paragraph.add_run()
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char, instr, fld_sep, fld_end])


def configure_section(section, first_page=False):
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)
    section.different_first_page_header_footer = first_page
    header = section.header
    header.is_linked_to_previous = False
    p = header.paragraphs[0]
    p.text = "CC3084 · LABORATORIO 3 · RECONOCIMIENTO ASL"
    p.style = doc.styles["Header"]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.runs[0].font.color.rgb = RGBColor.from_string(GRAY)
    p.runs[0].font.size = Pt(8)
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Informe de resultados  ·  ")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string(GRAY)
    add_field(p, "PAGE")


def setup_document():
    global doc
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    configure_section(section, first_page=True)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(TEXT)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    for name, size, color, before, after in (
        ("Title", 30, BLUE_DARK, 0, 18),
        ("Subtitle", 13, GRAY, 0, 8),
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 11.5, BLUE_DARK, 8, 4),
    ):
        st = styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor.from_string(color)
        st.font.bold = name != "Subtitle"
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True
    styles["Caption"].font.name = "Calibri"
    styles["Caption"].font.size = Pt(9)
    styles["Caption"].font.italic = True
    styles["Caption"].font.color.rgb = RGBColor.from_string(GRAY)
    styles["Caption"].paragraph_format.space_after = Pt(8)
    styles["Caption"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styles["Header"].font.name = "Calibri"
    styles["Footer"].font.name = "Calibri"
    for sty_name in ("List Bullet", "List Number"):
        st = styles[sty_name]
        st.font.name = "Calibri"
        st.font.size = Pt(10.5)
        st.paragraph_format.left_indent = Inches(0.5)
        st.paragraph_format.first_line_indent = Inches(-0.25)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.line_spacing = 1.08
    if "Table Note" not in [s.name for s in styles]:
        st = styles.add_style("Table Note", WD_STYLE_TYPE.PARAGRAPH)
        st.font.name = "Calibri"
        st.font.size = Pt(8)
        st.font.color.rgb = RGBColor.from_string(GRAY)
        st.paragraph_format.space_after = Pt(0)
    core = doc.core_properties
    core.title = "Laboratorio 3 - Deep Learning: Reconocimiento de Lenguaje de Señas (ASL)"
    core.subject = "Informe de resultados de CC3084 Data Science"
    core.author = "Equipo de Data Science"
    core.keywords = "ASL, CNN, Deep Learning, SVM, clasificación de imágenes"
    return doc


def add_text(text: str, bold_prefix: str | None = None, style=None):
    p = doc.add_paragraph(style=style)
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        r.bold = True
        p.add_run(text[len(bold_prefix):])
    else:
        p.add_run(text)
    return p


def add_bullets(items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_numbered(items):
    # Use a fresh numbering instance so each independent list starts at 1.
    numbering = doc.part.numbering_part.element
    base_num = numbering.num_having_numId(5)  # built-in "List Number"
    abstract_num_id = int(base_num.abstractNumId.val)
    new_num = numbering.add_num(abstract_num_id)
    new_num.add_lvlOverride(ilvl=0).add_startOverride(1)
    new_num_id = int(new_num.numId)
    for item in items:
        p = doc.add_paragraph(item, style="List Number")
        p_pr = p._p.get_or_add_pPr()
        num_pr = p_pr.get_or_add_numPr()
        num_pr.get_or_add_ilvl().val = 0
        num_pr.get_or_add_numId().val = new_num_id


def add_table(headers, rows, widths, font_size=8.5, alignments=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    set_table_geometry(table, widths)
    header = table.rows[0]
    set_repeat_table_header(header)
    for idx, value in enumerate(headers):
        cell = header.cells[idx]
        set_cell_shading(cell, LIGHT)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(value))
        r.bold = True
        r.font.size = Pt(font_size)
        r.font.color.rgb = RGBColor.from_string(BLUE_DARK)
    for row_values in rows:
        row = table.add_row()
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        for idx, value in enumerate(row_values):
            cell = row.cells[idx]
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            if alignments:
                p.alignment = alignments[idx]
            r = p.add_run(str(value))
            r.font.size = Pt(font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_callout(title: str, text: str, fill=PALE_BLUE):
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    set_table_geometry(table, [9000])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(BLUE_DARK)
    p = cell.add_paragraph(text)
    p.paragraph_format.space_after = Pt(0)
    return table


def add_figure(filename: str, width: float, caption: str, alt: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    shape = p.add_run().add_picture(str(FIGURES / filename), width=Inches(width))
    set_alt_text(shape, caption, alt)
    cap = doc.add_paragraph(caption, style="Caption")
    return cap


def add_page_break():
    doc.add_page_break()


def pct(value):
    return f"{float(value) * 100:.2f}%"


def metric_row(exp_id, label):
    r = experiments.loc[exp_id]
    return [
        label,
        pct(r.train_accuracy),
        pct(r.validation_accuracy),
        pct(r.test_accuracy),
        pct(r.macro_precision),
        pct(r.macro_recall),
        pct(r.macro_f1),
        "—" if pd.isna(r.test_loss) else f"{r.test_loss:.3f}",
    ]


setup_document()

# Cover
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(70)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("UNIVERSIDAD DEL VALLE DE GUATEMALA")
r.bold = True
r.font.size = Pt(12)
r.font.color.rgb = RGBColor.from_string(BLUE_DARK)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("CC3084 · DATA SCIENCE · SEGUNDO SEMESTRE 2026")
r.font.size = Pt(10)
r.font.color.rgb = RGBColor.from_string(GRAY)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(85)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("LABORATORIO 3")
r.bold = True
r.font.size = Pt(14)
r.font.color.rgb = RGBColor.from_string(BLUE)
p = doc.add_paragraph(style="Title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Deep Learning: Reconocimiento de Lenguaje de Señas (ASL)")
p = doc.add_paragraph(style="Subtitle")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Informe de resultados")

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(85)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Dataset: ASL Alphabet · 29 clases")
r.font.size = Pt(11)
r.font.color.rgb = RGBColor.from_string(TEXT)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Notebook ejecutado sin excepciones · métricas obtenidas de corridas reales")
r.font.size = Pt(9)
r.font.color.rgb = RGBColor.from_string(GRAY)

add_page_break()

doc.add_heading("1. Introducción", level=1)
add_text(
    "Este laboratorio desarrolla un clasificador de imágenes estáticas para reconocer las 26 letras del alfabeto ASL y las clases SPACE, DELETE y NOTHING. La metodología sigue el flujo visto en clase: carga con ImageDataGenerator y flow_from_directory, redes Sequential con Conv2D, MaxPooling2D, Flatten, Dense y Dropout, optimización SGD con momentum y comparación contra una red fully connected y SVM."
)
add_text(
    "El análisis parte de una auditoría completa del dataset, utiliza un submuestreo balanceado y reproducible, separa train, validation y test sin traslape, y selecciona hiperparámetros exclusivamente con validation. Test se consulta únicamente después de congelar la selección."
)

doc.add_heading("2. Objetivos", level=1)
add_bullets([
    "Construir y evaluar modelos capaces de reconocer 29 clases ASL a partir de fotografías RGB.",
    "Comparar el efecto de profundidad convolucional, dropout, learning rate e image augmentation mediante experimentos controlados.",
    "Contrastar las CNN con una red fully connected y con SVM, utilizando accuracy y métricas macro por clase.",
    "Analizar errores, sesgos, limitaciones y los requisitos necesarios para convertir el prototipo en un producto real.",
])

doc.add_heading("3. Dataset y análisis exploratorio", level=1)
doc.add_heading("3.1 Auditoría del dataset", level=2)
add_text(
    "Se utilizó ASL Alphabet de Kaggle. La auditoría recorrió todos los archivos antes del modelado y confirmó que no existen imágenes inválidas o corruptas. Todas comparten resolución, formato y número de canales."
)
add_table(
    ["Propiedad", "Resultado"],
    [
        ["Imágenes válidas", f"{dataset['valid_images']:,}"],
        ["Imágenes inválidas", f"{dataset['invalid_images']:,}"],
        ["Clases", "29: A–Z, DELETE, NOTHING y SPACE"],
        ["Imágenes por clase", "3,000 (3.45% del total por clase)"],
        ["Resolución original", "200 × 200 píxeles"],
        ["Formato y canales", "JPG, RGB, 3 canales"],
        ["Diferencia máx.–mín. por clase", "0 imágenes"],
    ],
    [3000, 6000],
    font_size=9,
)
add_text(
    "El dataset está perfectamente balanceado: las 29 clases contienen exactamente 3,000 imágenes, por lo que la diferencia entre la clase más frecuente y la menos frecuente es cero. Cada clase representa 3,000/87,000 = 3.45% del conjunto."
)
add_figure(
    "class_distribution.png",
    6.35,
    "Figura 1. Distribución de imágenes por clase.",
    "Gráfico de barras con 29 barras de igual altura, 3,000 imágenes por clase.",
)

doc.add_heading("3.2 Ejemplos y variabilidad visual", level=2)
add_text(
    "Las muestras permiten comparar letras distintas y examinar la variación dentro de una misma clase. El fondo es relativamente homogéneo, pero cambian la exposición, las sombras, la escala de la mano y su ubicación en el encuadre. También hay variaciones pequeñas en flexión y separación de los dedos."
)
add_figure(
    "five_letters.png",
    2.75,
    "Figura 2. Ejemplos reales de cinco letras distintas.",
    "Mosaico vertical de cinco fotografías de manos pertenecientes a cinco letras ASL distintas.",
)
add_figure(
    "within_class_A.png",
    6.3,
    "Figura 3. Variabilidad dentro de la clase A.",
    "Seis ejemplos de la letra A que muestran pequeñas variaciones de iluminación, posición y escala.",
)
add_text(
    "La clase no se define por valores idénticos de píxel, sino por la configuración de la mano. Por ello, el modelo debe aprender rasgos espaciales robustos a cambios moderados de posición e iluminación."
)

doc.add_heading("3.3 Clases especiales y grupos similares", level=2)
add_figure(
    "special_classes.png",
    4.35,
    "Figura 4. Ejemplos de SPACE, DELETE y NOTHING.",
    "Ejemplos del dataset correspondientes a las tres clases especiales.",
)
add_text(
    "NOTHING funciona como clase de rechazo para encuadres sin una seña objetivo; SPACE y DELETE representan acciones. Estas clases amplían el problema de A–Z a 29 salidas."
)
add_figure(
    "similar_MNS.png",
    4.35,
    "Figura 5. Hipótesis visual: M, N y S.",
    "Ejemplos de M, N y S dispuestos lado a lado para comparar la posición del pulgar y los dedos cerrados.",
)
add_figure(
    "similar_UVR.png",
    4.35,
    "Figura 6. Hipótesis visual: U, V y R.",
    "Ejemplos de U, V y R dispuestos lado a lado para comparar separación y cruce de dedos.",
)
add_text(
    "Antes del entrenamiento se planteó que M/N/S podrían confundirse por depender de diferencias finas en la posición del pulgar, y U/V/R por la separación o cruce de dos dedos. Esta es una hipótesis del EDA; se contrasta posteriormente con la matriz de confusión, sin asumir que será cierta."
)

add_page_break()
doc.add_heading("4. Submuestreo y división de datos", level=1)
doc.add_heading("4.1 Selección reproducible", level=2)
add_text(
    "Para mantener tiempos razonables en la computadora disponible, se seleccionaron aleatoriamente 600 imágenes por clase con RANDOM_SEED = 123. La muestra conserva el balance original y reúne 17,400 imágenes, equivalentes al 20% del dataset completo. No se tomaron simplemente los primeros archivos de cada carpeta."
)
add_table(
    ["Conjunto", "Total", "Por clase", "Proporción"],
    [
        ["Dataset original", "87,000", "3,000", "100%"],
        ["Submuestra", "17,400", "600", "20% del original"],
        ["Train", "10,440", "360", "60% de la submuestra"],
        ["Validation", "3,480", "120", "20% de la submuestra"],
        ["Test", "3,480", "120", "20% de la submuestra"],
    ],
    [2600, 1800, 1800, 2800],
    font_size=9,
)

doc.add_heading("4.2 Distribución y estratificación por clase", level=2)
split_rows = [[r.display_class, "3,000", "3.45%", int(r.train), int(r.validation), int(r.test)] for r in split_by_class.itertuples()]
add_table(
    ["Clase", "Original", "%", "Train", "Validation", "Test"],
    split_rows,
    [1800, 1400, 1200, 1400, 1800, 1400],
    font_size=7.8,
    alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 5,
)
add_text(
    "La tabla documenta las 29 clases, su distribución original y el split estratificado. Todas aparecen en los tres conjuntos con la misma proporción. Train se emplea para ajustar pesos; validation para comparar arquitecturas, hiperparámetros y early stopping; test se reserva para la evaluación final."
)

add_page_break()
doc.add_heading("4.3 Verificación de data leakage", level=2)
add_table(
    ["Intersección", "Archivos compartidos"],
    [
        ["Train ∩ Validation", leakage["intersections"]["train_validation"]],
        ["Train ∩ Test", leakage["intersections"]["train_test"]],
        ["Validation ∩ Test", leakage["intersections"]["validation_test"]],
    ],
    [5200, 3800],
    font_size=9,
)
add_text(
    f"La comprobación también confirmó {leakage['unique_source_images']:,} rutas de origen únicas para {leakage['total_manifest_rows']:,} filas del manifiesto. Por tanto, no existe traslape entre splits."
)

doc.add_heading("5. Preprocesamiento y carga", level=1)
add_text(
    "Las imágenes se redimensionaron de 200 × 200 × 3 a 64 × 64 × 3 para reducir memoria y tiempo, conservando la forma general de la mano. Se mantuvo RGB y se normalizaron los píxeles de 0–255 a 0–1 mediante rescale=1.0/255.0."
)
add_figure(
    "resize_comparison.png",
    5.9,
    "Figura 7. Imagen original y versión redimensionada a 64 × 64.",
    "Comparación visual de una fotografía ASL a 200 por 200 y a 64 por 64 píxeles.",
)
add_text(
    "La carga se realizó con ImageDataGenerator y flow_from_directory, target_size=(64,64), batch_size=64 y class_mode='categorical'. Al tratarse de 29 clases, la salida es Dense(29, activation='softmax') y la pérdida categorical_crossentropy; esto sustituye la salida sigmoid binaria del ejemplo de gatos y perros."
)
add_callout(
    "Control de evaluación",
    "Validation y test solo aplican rescale; no reciben augmentation. El orden de clases se mantuvo fijo y test no intervino en la selección de modelos.",
)

add_page_break()
doc.add_heading("6. Modelos CNN", level=1)
doc.add_heading("6.1 CNN Modelo 1: línea base", level=2)
add_table(
    ["Capa", "Configuración", "Salida conceptual"],
    [
        ["Conv2D", "32 filtros, 3×3, ReLU, he_uniform, same", "64×64×32"],
        ["MaxPooling2D", "2×2", "32×32×32"],
        ["Flatten", "—", "32,768"],
        ["Dense", "128, ReLU, he_uniform", "128"],
        ["Dense", "29, Softmax", "29 clases"],
    ],
    [1900, 4800, 2300],
    font_size=8.8,
)
add_text("Optimizador: SGD(learning_rate=0.01, momentum=0.9). Parámetros: 4,199,069. Entrenamiento: 10 épocas.")
add_figure(
    "history_cnn1.png",
    6.35,
    "Figura 8. Curvas de accuracy y loss de CNN Modelo 1.",
    "Dos gráficos con las curvas de entrenamiento y validación de accuracy y loss durante diez épocas.",
)
r = experiments.loc["CNN1"]
add_text(
    f"CNN1 alcanzó {pct(r.train_accuracy)} en train, {pct(r.validation_accuracy)} en validation y {pct(r.test_accuracy)} en test. La diferencia train–validation fue de 4.72 puntos porcentuales; el nivel bajo en ambos conjuntos indica capacidad insuficiente más que un sobreajuste severo."
)

doc.add_heading("6.2 CNN Modelo 2: segunda convolución", level=2)
add_table(
    ["Capa", "Configuración"],
    [
        ["Bloque 1", "Conv2D(32, 3×3, ReLU, he_uniform, same) + MaxPool(2×2)"],
        ["Bloque 2", "Conv2D(64, 3×3, ReLU, he_uniform, same) + MaxPool(2×2)"],
        ["Clasificador", "Flatten + Dense(128, ReLU, he_uniform) + Dense(29, Softmax)"],
        ["Optimización", "SGD(lr=0.01, momentum=0.9), categorical_crossentropy"],
    ],
    [2200, 6800],
    font_size=8.8,
)
add_figure(
    "history_cnn2.png",
    6.35,
    "Figura 9. Curvas de accuracy y loss de CNN Modelo 2.",
    "Curvas de entrenamiento y validación de la CNN con dos capas convolucionales.",
)
r = experiments.loc["CNN2"]
add_text(
    f"CNN2 obtuvo {pct(r.train_accuracy)} en train, {pct(r.validation_accuracy)} en validation y {pct(r.test_accuracy)} en test. Frente a CNN1, agregar una segunda convolución aumentó validation en 34.63 puntos porcentuales. La brecha train–validation subió a 8.28 puntos, señal de mayor sobreajuste, pero la ganancia de capacidad y generalización fue sustancial."
)

doc.add_heading("6.3 Dropout y learning rate", level=2)
add_text(
    "Sobre la arquitectura de CNN2 se añadió Dropout después de cada MaxPooling. Se compararon tasas 0.1 y 0.2 con learning rate 0.001. Además, para aislar el efecto del learning rate se repitió Dropout 0.1 con learning rate 0.01. Las corridas de 15 épocas utilizaron EarlyStopping sobre val_loss, patience=3 y restore_best_weights=True."
)
drop_rows = []
for exp_id, label in (("CNN3_D01", "Dropout 0.1 · LR 0.001"), ("CNN3_D02", "Dropout 0.2 · LR 0.001"), ("CNN3_D01_LR01", "Dropout 0.1 · LR 0.01")):
    r = experiments.loc[exp_id]
    drop_rows.append([label, pct(r.train_accuracy), pct(r.validation_accuracy), pct(r.test_accuracy), int(r.epochs_trained), int(r.best_epoch)])
add_table(
    ["Configuración", "Train", "Validation", "Test", "Épocas", "Mejor época"],
    drop_rows,
    [2900, 1200, 1500, 1200, 1100, 1100],
    font_size=8.4,
    alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 5,
)
add_figure(
    "history_cnn3_d01.png",
    6.35,
    "Figura 10. Dropout 0.1 con learning rate 0.001.",
    "Curvas de accuracy y loss para CNN2 con dropout 0.1 y learning rate 0.001.",
)
add_figure(
    "history_cnn3_d02.png",
    6.35,
    "Figura 11. Dropout 0.2 con learning rate 0.001.",
    "Curvas de accuracy y loss para CNN2 con dropout 0.2 y learning rate 0.001.",
)
add_figure(
    "history_cnn3_d01_lr01.png",
    6.35,
    "Figura 12. Dropout 0.1 con learning rate 0.01.",
    "Curvas de accuracy y loss del mejor experimento CNN, con dropout 0.1 y learning rate 0.01.",
)
add_text(
    "A learning rate 0.001, Dropout 0.1 superó a 0.2 por 6.87 puntos de validation; la regularización 0.2 fue excesiva para esta corrida. Manteniendo Dropout 0.1, cambiar learning rate de 0.001 a 0.01 produjo +49.97 puntos de validation, por lo que la velocidad de aprendizaje fue decisiva."
)
add_text(
    "La comparación controlada para evaluar dropout es CNN2 contra CNN3_D01_LR01, porque ambas usan learning rate 0.01. Dropout 0.1 elevó validation de 83.07% a 86.15% y redujo la brecha train–validation de 8.28 a 6.33 puntos. En este contexto, sí ayudó."
)

add_page_break()
doc.add_heading("7. Image augmentation", level=1)
add_text(
    "La mejor CNN según validation se reentrenó con transformaciones moderadas solo en train: width_shift_range=0.1, height_shift_range=0.1, rotation_range=10 y zoom_range=0.1. No se activó horizontal_flip: una reflexión conserva normalmente la clase de un gato o perro, pero en ASL puede alterar relaciones espaciales relevantes entre mano, dedos y orientación."
)
add_figure(
    "augmentation_examples.png",
    6.1,
    "Figura 13. Imagen original y variaciones aumentadas.",
    "Cuadrícula con una seña original y versiones con desplazamientos, rotación y zoom moderados.",
)
add_figure(
    "history_cnn_aug.png",
    6.35,
    "Figura 14. Curvas de la CNN con augmentation.",
    "Curvas de accuracy y loss para la mejor arquitectura reentrenada con aumento moderado de datos.",
)
r_no = experiments.loc["CNN3_D01_LR01"]
r_aug = experiments.loc["CNN_AUG"]
add_text(
    f"Sin augmentation, la CNN obtuvo {pct(r_no.train_accuracy)} train y {pct(r_no.validation_accuracy)} validation; con augmentation obtuvo {pct(r_aug.train_accuracy)} y {pct(r_aug.validation_accuracy)}. La brecha bajó de 6.33 a 1.97 puntos, pero validation descendió 5.26 puntos. Las transformaciones regularizaron el modelo, aunque con esta configuración y tiempo de entrenamiento no mejoraron la generalización."
)

doc.add_heading("8. Red fully connected", level=1)
add_text(
    "La línea base sin convoluciones usa Flatten(64×64×3), Dense y Softmax. Al no incluir Conv2D, trata los píxeles como un vector y no aprende extractores espaciales locales. Se probaron dos configuraciones para variar capacidad y regularización."
)
fc_rows = []
for exp_id, label in (("FC1", "Flatten + Dense 128 + Softmax"), ("FC2", "Flatten + Dense 256 + Dropout 0.1 + Softmax")):
    r = experiments.loc[exp_id]
    fc_rows.append([label, f"{int(r.parameters):,}", pct(r.train_accuracy), pct(r.validation_accuracy), pct(r.test_accuracy), pct(r.macro_f1)])
add_table(
    ["Arquitectura", "Parámetros", "Train", "Validation", "Test", "Macro F1"],
    fc_rows,
    [3100, 1400, 1100, 1400, 1100, 1100],
    font_size=8.2,
    alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 5,
)
add_figure(
    "history_fc1.png",
    6.35,
    "Figura 15. Curvas de Fully Connected 128.",
    "Curvas de accuracy y loss de la red fully connected con una capa densa de 128 unidades.",
)
add_figure(
    "history_fc2.png",
    6.35,
    "Figura 16. Curvas de Fully Connected 256 con dropout 0.1.",
    "Curvas de accuracy y loss de la red fully connected con 256 unidades y dropout 0.1.",
)
add_text(
    "FC2 mejoró validation en 18.33 puntos respecto de FC1, pero ambas redes quedaron muy por debajo de las CNN. FC1 permaneció en 3.45%, equivalente a acertar una de 29 clases, y FC2 llegó a 21.78%; el patrón es de underfitting. La extracción espacial convolucional resultó esencial."
)

doc.add_heading("9. SVM", level=1)
add_text(
    "SVM se seleccionó como algoritmo clásico porque puede construir fronteras de decisión de gran margen en espacios de alta dimensión. El pipeline fue: imagen 64×64 RGB → flatten → StandardScaler → PCA(100) → SVC(C=1). StandardScaler y PCA se ajustaron exclusivamente con train y luego se aplicaron sin refit a validation y test."
)
svm_rows = []
for exp_id, label in (("SVM_LINEAR", "Linear"), ("SVM_RBF", "RBF")):
    r = experiments.loc[exp_id]
    svm_rows.append([label, "1", "100", pct(r.train_accuracy), pct(r.validation_accuracy), pct(r.test_accuracy), pct(r.macro_f1), f"{r.training_time:.2f} s"])
add_table(
    ["Kernel", "C", "PCA", "Train", "Validation", "Test", "Macro F1", "Tiempo"],
    svm_rows,
    [1100, 700, 750, 1100, 1400, 1100, 1300, 1100],
    font_size=8.1,
    alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 7,
)
add_text(
    "RBF superó a linear por 4.74 puntos de validation y 4.94 puntos de test. SVM RBF fue competitivo frente a la CNN con augmentation, pero quedó 5.43 puntos por debajo de la mejor CNN en validation."
)

add_page_break()
doc.add_heading("10. Comparación final y selección", level=1)
comparison_rows = [
    metric_row("CNN1", "CNN Modelo 1"),
    metric_row("CNN2", "CNN Modelo 2"),
    metric_row("CNN3_D01", "CNN Dropout 0.1 · LR .001"),
    metric_row("CNN3_D02", "CNN Dropout 0.2 · LR .001"),
    metric_row("CNN3_D01_LR01", "CNN Dropout 0.1 · LR .01"),
    metric_row("CNN_AUG", "CNN + augmentation"),
    metric_row("FC1", "Fully Connected 128"),
    metric_row("FC2", "Fully Connected 256 + D0.1"),
    metric_row("SVM_LINEAR", "SVM Linear"),
    metric_row("SVM_RBF", "SVM RBF"),
]
add_table(
    ["Modelo", "Train", "Validation", "Test", "Prec. macro", "Recall macro", "F1 macro", "Loss test"],
    comparison_rows,
    [2550, 850, 1150, 850, 1150, 1150, 1000, 1050],
    font_size=7.7,
    alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 7,
)
resource_rows = []
for exp_id, label in (("CNN1", "CNN1"), ("CNN2", "CNN2"), ("CNN3_D01", "CNN D0.1 LR.001"), ("CNN3_D02", "CNN D0.2 LR.001"), ("CNN3_D01_LR01", "CNN D0.1 LR.01"), ("CNN_AUG", "CNN augmentation"), ("FC1", "FC1"), ("FC2", "FC2"), ("SVM_LINEAR", "SVM Linear"), ("SVM_RBF", "SVM RBF")):
    r = experiments.loc[exp_id]
    resource_rows.append([
        label,
        "—" if pd.isna(r.epochs_trained) else int(r.epochs_trained),
        "—" if pd.isna(r.best_epoch) else int(r.best_epoch),
        "—" if pd.isna(r.parameters) else f"{int(r.parameters):,}",
        f"{r.training_time:.2f}",
        "Sí" if bool(r.augmentation) else "No",
    ])
add_table(
    ["Modelo", "Épocas", "Mejor época", "Parámetros", "Tiempo (s)", "Augmentation"],
    resource_rows,
    [2600, 1100, 1400, 1800, 1200, 1300],
    font_size=8.0,
    alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 5,
)
add_callout(
    "Modelo seleccionado: CNN3_D01_LR01",
    "La selección se congeló por mayor validation accuracy, usando validation loss como desempate, antes de consultar test. El modelo combina dos convoluciones, Dropout 0.1, learning rate 0.01 y no usa augmentation. Alcanzó 86.15% en validation, 86.44% en test y 86.09% de macro F1.",
)
add_text(
    "El ganador no se eligió por training accuracy. Además de liderar validation, mantuvo métricas macro consistentes y un comportamiento por clase superior al resto. La evaluación de test solo se agregó después de la selección."
)

doc.add_heading("11. Matriz de confusión y análisis de errores", level=1)
add_figure(
    "confusion_matrix_best.png",
    6.25,
    "Figura 17. Matriz de confusión del mejor modelo sobre test (29 × 29).",
    "Matriz de confusión de 29 clases para CNN3_D01_LR01, con etiquetas reales y predichas.",
)
add_text(
    "Las clases con mayor recall fueron C (98.33%), NOTHING (97.50%), N (97.50%), F (97.50%) e I (96.67%). Las de menor recall fueron M (37.50%), E (50.00%), X (74.17%), O (76.67%) y W (78.33%)."
)
conf_rows = [[r.actual_class, r.predicted_class, int(r.errors)] for r in top_confusions.itertuples()]
add_table(
    ["Clase real", "Predicción", "Errores"],
    conf_rows,
    [3100, 3100, 1800],
    font_size=8.5,
    alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER],
)
add_text(
    "La hipótesis M/N/S sí se confirmó: hubo 66 errores internos en el grupo y M→N fue la confusión dominante con 61 casos. En U/V/R hubo 29 errores internos; U→V aportó 18. También apareció W→V con 21 errores, mostrando que las confusiones reales no se limitaron a los grupos previstos en el EDA."
)
add_figure(
    "misclassified_examples.png",
    6.15,
    "Figura 18. Ejemplos reales de errores del mejor modelo.",
    "Cuadrícula de imágenes mal clasificadas con clase real, predicción y confianza del modelo.",
)
add_text(
    "Los errores de alta confianza son relevantes porque indican que una salida softmax alta no garantiza que la predicción sea correcta fuera o dentro del dominio. Un producto debe definir un umbral de confianza y permitir rechazo o confirmación cuando exista incertidumbre."
)

doc.add_heading("12. Función de predicción", level=1)
add_text(
    "El proyecto incluye una función reusable para fotografías nuevas. Su procedimiento es:"
)
add_numbered([
    "Cargar la ruta de imagen y convertirla a RGB.",
    "Redimensionar a 64 × 64, convertir a arreglo y normalizar con 1/255.",
    "Agregar la dimensión de batch y ejecutar model.predict().",
    "Aplicar argmax, convertir el índice al nombre de clase y recuperar la probabilidad máxima.",
    "Mostrar la imagen, la clase predicha y la confianza.",
])
add_text(
    f"El modelo guardado se cargó correctamente y produjo una salida de forma {reload_check['prediction_shape']} cuya suma de probabilidades fue {reload_check['probabilities_sum']:.7f}, lo que verifica el flujo de inferencia."
)

doc.add_heading("13. Fotografías propias", level=1)
add_callout(
    "PENDIENTE: fotografías reales de los integrantes",
    "No se encontraron fotografías propias y no se sustituyeron con imágenes del dataset. Cada integrante debe colocar al menos cinco letras distintas en data/custom_test/<integrante>/ usando nombres como A_01.jpg, B_01.jpg y C_01.jpg. Después debe ejecutarse la evaluación preparada para obtener predicciones, accuracy y comparación contra test interno.",
    fill=PALE_YELLOW,
)
add_text(
    "Cuando se agreguen las fotografías, la discusión debe separar observaciones de posibles causas y comparar iluminación, cámara, fondo, orientación, distancia, escala, tono de piel, tamaño de mano y posición. Una caída respecto de 86.44% en test podría indicar domain shift, pero su causa deberá sustentarse con las imágenes y resultados reales."
)

doc.add_heading("14. Accesibilidad y sesgo", level=1)
add_text(
    "Una accuracy alta en este dataset no garantiza desempeño adecuado para todas las personas. El fondo relativamente controlado y la posible falta de diversidad de usuarios, tonos de piel, tamaños de mano, cámaras, iluminación y ángulos pueden limitar la generalización. Además, la clasificación por imagen estática no cubre completamente la variación temporal y lingüística del ASL."
)
add_bullets([
    "Recopilar datos con usuarios diversos, distintos tonos de piel, tamaños de mano, cámaras, fondos e iluminación.",
    "Evaluar métricas por subgrupos y documentar diferencias, no solo un promedio global.",
    "Realizar pruebas con usuarios reales y revisar errores junto con personas que usan ASL.",
    "Incorporar manejo explícito de baja confianza: rechazo, confirmación o repetición de captura.",
    "Evitar presentar el sistema como sustituto universal de intérpretes o de formas de comunicación elegidas por la comunidad sorda.",
])

doc.add_heading("15. Del prototipo a un producto real", level=1)
add_text(
    "El laboratorio clasifica una imagen estática en una de 29 clases; no es un traductor ASL completo. Un producto tipo SignBridge requeriría captura continua de video, detección y seguimiento de manos, modelado temporal de movimientos, reconocimiento de señas dinámicas, segmentación de secuencias, construcción de palabras, contexto lingüístico y manejo de incertidumbre."
)
add_text(
    "La recomendación inmediata es construir primero un módulo de captura guiada que detecte la mano, valide encuadre e iluminación y rechace predicciones de baja confianza. Esta mejora es concreta, comprobable y reduce errores antes de incorporar secuencias y lenguaje."
)

doc.add_heading("16. Conclusiones", level=1)
add_numbered([
    "El mejor modelo fue CNN3_D01_LR01: dos convoluciones, Dropout 0.1 y SGD con learning rate 0.01. Obtuvo 86.15% de validation, 86.44% de test y 86.09% de macro F1.",
    "Agregar una segunda convolución ayudó: validation aumentó de 48.45% a 83.07%, aunque la brecha train–validation creció.",
    "Dropout 0.1 ayudó en la comparación controlada con LR 0.01: aumentó validation 3.07 puntos y redujo la brecha de 8.28 a 6.33 puntos.",
    "Con LR 0.001, Dropout 0.1 superó a 0.2 por 6.87 puntos; 0.2 regularizó demasiado en esta configuración.",
    "El learning rate fue determinante: con Dropout 0.1, pasar de 0.001 a 0.01 mejoró validation en 49.97 puntos.",
    "Augmentation redujo la brecha train–validation, pero no mejoró validation: cayó 5.26 puntos frente a la misma CNN sin aumento.",
    "Las redes fully connected mostraron underfitting y confirmaron la ventaja de conservar estructura espacial mediante convoluciones.",
    "SVM RBF fue mejor que SVM Linear y competitivo, pero no superó a la mejor CNN.",
    "M fue la clase más difícil y M→N la principal confusión; el grupo U/V/R también presentó errores, por lo que parte de las hipótesis del EDA se confirmó.",
    "El prototipo está limitado por el dominio del dataset y por trabajar con imágenes estáticas; faltan fotografías propias y evaluación real antes de afirmar generalización.",
])

doc.add_heading("17. Reproducibilidad y material de entrega", level=1)
add_table(
    ["Elemento", "Ubicación o configuración"],
    [
        ["Notebook ejecutado", "notebooks/Lab03_ASL.ipynb"],
        ["Código del análisis", "src/ y scripts/"],
        ["Métricas y figuras", "outputs/metrics/ y outputs/figures/"],
        ["Seed / imagen / batch", "123 / 64×64 RGB / 64"],
        ["Splits", "60% train / 20% validation / 20% test"],
        ["Dependencias", "requirements.txt"],
        ["Instrucciones", "README.md y data/custom_test/README.md"],
        ["Repositorio remoto", "PENDIENTE: no hay URL remota configurada"],
    ],
    [3100, 5900],
    font_size=8.7,
)
add_text(
    "El notebook contiene código, resultados e interpretaciones; los CSV conservan el registro de experimentos y las evaluaciones por clase. Los modelos grandes se regeneran localmente y no se versionan. Antes de enviar, debe añadirse la URL del repositorio remoto y completarse la prueba con fotografías de los integrantes."
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUTPUT)
print(OUTPUT)

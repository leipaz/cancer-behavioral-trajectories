#!/usr/bin/env python3
"""Informe del baseline de k-means sin LDA.

    python scripts/vqvae/write_hdo_kmeans_baseline_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "results/vq-vae/training_provenance/hdo_kmeans_baseline"
OUT = FOLDER / "informe_kmeans_baseline.docx"


def _num(value: float, digits: int = 2) -> str:
    text = f"{value:.{digits}f}".replace(".", ",")
    return text


def _bracket(block: dict, digits: int = 2, suffix: str = "") -> str:
    return f"{_num(block['point'], digits)}{_num_ci(block, digits)}{suffix}"


def _num_ci(block: dict, digits: int = 2) -> str:
    return f" ({_num(block['ci_low'], digits)} a {_num(block['ci_high'], digits)})"
NAVY = RGBColor(0x1F, 0x4E, 0x79)


def shade(cell, color: str) -> None:
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), color)
    element.set(qn("w:val"), "clear")
    cell._tc.get_or_add_tcPr().append(element)


def add_text(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.08
    run = paragraph.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    paragraph.paragraph_format.space_after = Pt(6)
    for run in paragraph.runs:
        run.font.color.rgb = NAVY
        run.font.name = "Calibri"


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(header)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, "1F4E79")
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(value)
            run.font.name = "Calibri"
            run.font.size = Pt(10)
            if i % 2 == 1:
                shade(cell, "F4F7FB")
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run("Baseline de k-means sobre los días de HDO")
    run.bold = True
    run.font.name = "Calibri"
    run.font.size = Pt(18)
    run.font.color.rgb = NAVY

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(12)
    sub = subtitle.add_run("Un solo modelo, sin LDA, puntuado en los 1013 puntos de la alarma habitual.")
    sub.italic = True
    sub.font.name = "Calibri"
    sub.font.size = Pt(11)
    sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    add_heading(doc, "Qué se ha calculado")
    add_text(
        doc,
        "Este cálculo es un baseline sencillo. Sustituye el VQ-VAE por un k-means de seis grupos sobre las ocho variables diarias de HDO y no entrena ningún LDA. La predicción consiste en aplicar ese k-means a los días de la ventana de 90 días de cada punto de decisión. Se puntúa en los mismos 1013 puntos en los que la alarma habitual, con el vocabulario externo ya congelado, tiene un AUROC de 0,70.",
    )
    add_text(
        doc,
        "Es un modelo único, ajustado con las 175 pacientes de desarrollo. No es la validación anidada del SAP. Esa otra serie, k-means y después LDA dentro de cada pliegue, está resumida al final para que los dos números no se mezclen.",
    )

    add_heading(doc, "De dónde salen los días")
    add_text(
        doc,
        "La cohorte es DB1: 175 pacientes con perfil. Las tres sin perfil, 31015, 61004 y 63006, no entran. El calendario de cada paciente es la ventana del embedding a0, de date_start a date_end, la misma longitud que la secuencia de códigos del VQ-VAE. Son 52.236 días. Si un día de esa ventana no tiene fila en el resumen diario, el día se conserva y sus ocho variables quedan vacías.",
    )
    add_text(
        doc,
        "Las variables son sleep_start, location_distance, location_time_home, sleep_duration, activity_walking, app_usage_total, location_clusters_count y steps_steps_total. Primero se aplica el recorte congelado del VQ-VAE: un valor fuera de rango pasa a vacío. Después, las siete variables positivas se transforman con log1p. sleep_start no, porque ya es un valor con signo.",
    )

    add_heading(doc, "Estandarización y media")
    add_text(
        doc,
        "Cada variable se estandariza antes del k-means. La media y la desviación típica se calculan con todos los valores observados de esa variable, en los 52.236 días de las 175 pacientes. Un día vacío no entra en esa media. Después de restar la media y dividir por la desviación, lo que sigue vacío se rellena con cero. En esa escala, cero es la media. Hay 161.623 celdas rellenadas así.",
    )
    add_text(
        doc,
        "La media no se calcula por paciente ni por pliegue. Es una sola media por variable, hecha con toda la cohorte.",
    )

    add_heading(doc, "El k-means")
    add_text(
        doc,
        "Se ajusta un k-means de seis grupos sobre los 52.236 días ya estandarizados e imputados. Cada día es un punto de ocho coordenadas. El algoritmo es Lloyd, con inicialización k-means++, una sola arrancada, 300 iteraciones como máximo y tolerancia 0,0001. La semilla es 20261031. La inercia final es 159.567.",
    )
    add_text(
        doc,
        "El grupo de una paciente, el que luego entra en la curva, es el grupo más frecuente de su secuencia entera. Si dos grupos empatan en número de días, se queda el de índice más bajo.",
    )

    add_heading(doc, "Qué grupos se llaman desfavorables")
    add_text(
        doc,
        "Hace falta una regla para pasar de un grupo a una alarma, porque el k-means no usa el desenlace clínico. Se usa la regla 3/3 del resto del proyecto. Entran las pacientes con Tiempo_obs_DB1 de al menos 28 días: 170 de las 175. Se ordenan los seis grupos por la Kaplan–Meier de progresión a 365 días. Los tres con la supervivencia más alta son favorables y los tres con la más baja son desfavorables. El criterio que separó los grupos fue esa supervivencia a 365 días, sin empate.",
    )
    add_table(
        doc,
        ["Grupo", "Pacientes", "Eventos", "Supervivencia a 365 días", "Etiqueta"],
        [
            ["0", "8", "3", "0,56", "Desfavorable"],
            ["1", "92", "29", "0,75", "Favorable"],
            ["2", "8", "4", "0,43", "Desfavorable"],
            ["3", "23", "5", "0,81", "Favorable"],
            ["4", "25", "9", "0,62", "Favorable"],
            ["5", "14", "8", "0,32", "Desfavorable"],
        ],
    )
    add_text(
        doc,
        "Los grupos desfavorables son el 0, el 2 y el 5. El grupo 1 concentra a 92 de las 170 pacientes y queda favorable. Estas curvas usan el desenlace de la misma cohorte que después se evalúa.",
    )

    add_heading(doc, "Cómo se puntúa la ventana")
    add_text(
        doc,
        "No hay LDA. En cada punto de decisión se toma la misma ventana de 90 días que la alarma habitual: tres bloques de 30 días de perfil, sin solaparse, desde el primer día. Si la secuencia no acaba en un múltiplo de 30, el último bloque son los últimos 30 días. A cada día de esos bloques se le asigna el grupo del k-means ya ajustado. No se vuelve a entrenar.",
    )
    add_text(
        doc,
        "La carga de un bloque es la proporción de sus días que caen en un grupo desfavorable. La puntuación del punto es la suma de las tres cargas, así que va de 0 a 3. El día de decisión es el índice del bloque por 30. El desenlace es el de la alarma: horizonte de 120 días y reloj Tiempo_obs_digital. Un punto es positivo si la progresión cae dentro de ese horizonte, negativo si el seguimiento cubre el horizonte, y se descarta si el día de decisión ya ha pasado el fin del seguimiento.",
    )
    add_text(
        doc,
        "Los puntos son los 1013 pares paciente-bloque congelados de la alarma del vocabulario externo: 155 positivos y 858 negativos, de 114 pacientes. De esas 114, 45 tienen una progresión en la cohorte.",
    )

    add_heading(doc, "El corte y la comparación")
    add_text(
        doc,
        "El corte del baseline es el de Youden sobre estos mismos 1013 puntos, y a igualdad se toma el corte más bajo. Sale 1,70. Se avisa cuando la puntuación llega a ese valor o lo supera. La antelación es la mediana, entre las pacientes con progresión a las que el corte detecta, de los días que van desde el primer aviso verdadero hasta el día del evento.",
    )
    add_text(
        doc,
        "El pipeline normal de la tabla es la alarma habitual: códigos del VQ-VAE, LDA congelado, topics 3, 4 y 5 desfavorables, y la suma de tres meses. El corte es 1,60. Se lee en los mismos 1013 pares y con el mismo desenlace. Su AUROC en estos puntos es 0,702.",
    )

    comparison = json.loads((FOLDER / "comparison.json").read_text(encoding="utf-8"))
    boot = comparison["bootstrap"]
    add_heading(doc, "Intervalos, el mismo bootstrap de la alarma")
    add_text(
        doc,
        "El intervalo es el de la figura de la alarma habitual: bootstrap por paciente, 2000 remuestreos, semilla 42, y los percentiles 2,5 y 97,5. En cada remuestreo se extraen las 114 pacientes con reemplazo y se conservan todos sus puntos de decisión. Las dos puntuaciones usan la misma extracción, sobre estos 1013 puntos. El AUROC es exactamente esa función. La sensibilidad, la especificidad y la antelación usan las mismas extracciones, con el corte fijo: 1,70 para el k-means y 1,60 para la alarma. El corte no se vuelve a elegir dentro del remuestreo.",
    )

    add_heading(doc, "Resultados en los mismos 1013 puntos")
    add_table(
        doc,
        ["", "k-means", "Alarma habitual"],
        [
            ["AUROC", _bracket(boot["AUROC_baseline"]), _bracket(boot["AUROC_normal"])],
            ["Sensibilidad", _bracket(boot["sensitivity_baseline"]), _bracket(boot["sensitivity_normal"])],
            ["Especificidad", _bracket(boot["specificity_baseline"]), _bracket(boot["specificity_normal"])],
            ["Verdaderos positivos", "40", "105"],
            ["Falsos negativos", "115", "50"],
            ["Falsos positivos", "78", "329"],
            ["Verdaderos negativos", "780", "529"],
            ["Antelación mediana", _bracket(boot["median_lead_baseline"], 1, " días"), _bracket(boot["median_lead_normal"], 1, " días")],
            ["Progresiones detectadas", "14 de 45", "36 de 45"],
            ["Corte", "1,70, Youden en estos puntos", "1,60, el de la alarma"],
        ],
    )
    add_text(
        doc,
        "El paréntesis es el intervalo del 95 %. La diferencia de AUROC, k-means menos alarma habitual, es "
        f"{_bracket(boot['AUROC_baseline_minus_normal'])}. "
        "La puntuación media del k-means es 0,98 en los puntos positivos y 0,77 en los negativos. La separación es pequeña, y por eso el AUROC queda en 0,58. El corte de Youden, 1,70, avisa poco: solo el 12 % de los puntos lo cruzan. La antelación de 91 días es la mediana de las 14 progresiones detectadas, no de las 45.",
    )

    add_heading(doc, "La media está calculada en los mismos datos que se evalúan")
    add_text(
        doc,
        "Sí. La media de cada variable, y también la desviación típica, se calcularon con todos los días observados de las 175 pacientes. Esos días incluyen a las 114 pacientes y a las ventanas de los 1013 puntos que después se puntúan. Un valor faltante de una paciente evaluada se rellena con una media en la que esa paciente ya ha entrado.",
    )
    add_text(
        doc,
        "No es solo la media. El k-means se ajustó con esos mismos 52.236 días. La regla 3/3 usó el desenlace de esas mismas pacientes para decidir qué grupos son desfavorables. El corte de Youden se eligió con los mismos 1013 puntos. El 0,58 describe a un modelo que ha visto la cohorte sobre la que se le mide. No es el resultado de un conjunto dejado fuera.",
    )
    add_text(
        doc,
        "La alarma habitual de la tabla tampoco es una prueba externa: su LDA se ajustó con HDO y el 0,70 es esa alarma leída en estos 1013 puntos. Las dos columnas comparten los puntos. La diferencia es que la media, los grupos y el corte del k-means se estimaron mirando también los días y los desenlaces que luego entran en la cifra.",
    )

    add_heading(doc, "El otro k-means, el de la validación anidada")
    add_text(
        doc,
        "Hay un segundo cálculo, guardado en hdo_only_lda, que no hace esta imputación global. Ahí el k-means sustituye al VQ-VAE y, en un segundo paso, un LDA de seis topics se ajusta sobre la secuencia de grupos. La media y la desviación de cada pliegue se calculan solo con las pacientes de entrenamiento. Las 35 de validación reciben esa media ya hecha, el k-means ya hecho y el LDA ya hecho. La etiqueta 3/3 y el corte de Youden también salen del entrenamiento. VAL-SET no entra en ninguno de esos pasos.",
    )
    add_text(
        doc,
        "En esa serie, la media de las diez repeticiones sobre los 1013 puntos es AUROC 0,618, sensibilidad 0,46, especificidad 0,69 y antelación 92 días. El vocabulario externo, reajustado en los mismos pliegues, da 0,617, 0,64, 0,52 y 95 días. Ese 0,62 es la validación anidada. El 0,70 es la alarma única leída sobre toda la cohorte. El 0,58 de este informe es el baseline sin LDA, con la media calculada en toda la cohorte.",
    )

    add_heading(doc, "Dónde está cada cosa")
    add_text(
        doc,
        "El script es scripts/vqvae/run_hdo_kmeans_baseline.py. Los puntos, los grupos y las cifras están en results/vq-vae/training_provenance/hdo_kmeans_baseline/. La alarma habitual se lee de results/alarm/runs/Evento_DB1_clock/Bt_vs_mensual/decision_points_collapsed_teb2.csv, restringida a los 1013 pares. La serie anidada está en results/vq-vae/training_provenance/hdo_only_lda/.",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()

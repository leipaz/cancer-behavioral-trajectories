#!/usr/bin/env python3
"""Self-contained report for the ten SAP repeats of the stratified nested validation.

    python scripts/03_analysis/lda/run_val2_sap10_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "results/lda/val_2/anidada_estratificada_sap10"
SINGLE = PROJECT_ROOT / "results/lda/val_2/anidada_estratificada/alarma/rendimiento_por_vuelta.csv"
NAVY = "1F4E79"
ZEBRA = "F4F7FB"
PAGE_WIDTH = Cm(29.7)
PAGE_HEIGHT = Cm(21.0)
MARGIN = Cm(1.4)
CONTENT_CM = 29.7 - 2 * 1.4
ROLE = {"held_out": "held out", "external_test": "external"}
HOSPITAL = {
    "CHUAC": "CHUAC",
    "Hospital Clínico Unversitario de Valencia": "Clínico Valencia",
    "Hospital San Pedro de Alcántara": "San Pedro",
    "Hospital Universitario Son Espases": "Son Espases",
    "Hospital Universitario de Fuenlabrada": "Fuenlabrada",
    "Hospital Unversitario Virgen de la Macarena": "Macarena",
    "Hospital Unversitario de La Princesa": "La Princesa",
    "Hospital de Navarra": "Navarra",
    "ICO Bellvitge": "ICO Bellvitge",
}


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def band(point: float, low: float, high: float) -> str:
    return f"{num(point)} ({num(low)}–{num(high)})"


def shade(cell, color: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:fill"), color)
    props.append(fill)


def no_wrap(cell) -> None:
    props = cell._tc.get_or_add_tcPr()
    props.append(OxmlElement("w:noWrap"))


def set_width(cell, width: Cm) -> None:
    props = cell._tc.get_or_add_tcPr()
    tcw = OxmlElement("w:tcW")
    tcw.set(qn("w:w"), str(int(width.twips)))
    tcw.set(qn("w:type"), "dxa")
    props.append(tcw)


def add_table(doc: Document, header: list[str], rows: list[list[str]], widths: list[Cm], size: int = 9, wrap: bool = False) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table.autofit = False
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    for child in list(tbl_pr):
        if child.tag == qn("w:tblW"):
            tbl_pr.remove(child)
    total = sum(int(width.twips) for width in widths)
    tbl_w = OxmlElement("w:tblW")
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_w)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    for existing in tbl.findall(qn("w:tblGrid")):
        tbl.remove(existing)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), "D0D7E2")
        borders.append(element)
    tbl_pr.append(borders)
    grid = OxmlElement("w:tblGrid")
    for width in widths:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(int(width.twips)))
        grid.append(column)
    tbl.insert(1, grid)

    def fill(cell, text: str, *, bold: bool, white: bool, width: Cm) -> None:
        set_width(cell, width)
        if not wrap:
            no_wrap(cell)
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_before = Pt(1)
        paragraph.paragraph_format.space_after = Pt(1)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.bold = bold
        run.font.size = Pt(size)
        run.font.name = "Calibri"
        if white:
            run.font.color.rgb = RGBColor(255, 255, 255)

    for index, title in enumerate(header):
        cell = table.rows[0].cells[index]
        fill(cell, title, bold=True, white=True, width=widths[index])
        shade(cell, NAVY)
    marker = OxmlElement("w:tblHeader")
    table.rows[0]._tr.get_or_add_trPr().append(marker)
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            cell = table.rows[r + 1].cells[c]
            fill(cell, value, bold=(c == 0), white=False, width=widths[c])
            if r % 2 == 1:
                shade(cell, ZEBRA)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(12)
    paragraph.paragraph_format.space_after = Pt(4)
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        run.font.name = "Calibri"


def add_text(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11)


def widths_for(weights: list[float]) -> list[Cm]:
    total = sum(weights)
    return [Cm(CONTENT_CM * weight / total) for weight in weights]


def partition_tables(partitions: pd.DataFrame) -> tuple[list[list[str]], list[str], list[list[str]]]:
    external = partitions[partitions["side"] == "external"]
    train = partitions[partitions["side"] == "train"].copy()
    train["inner_fold"] = train["inner_fold"].astype(int)
    event_rows = []
    for fold in range(5):
        outer = external[external["outer_fold"] == fold]
        inner = train[train["outer_fold"] == fold]
        counts = inner.groupby("inner_fold")["event"].sum()
        event_rows.append([
            str(fold),
            str(int(len(outer))),
            str(int(outer["event"].sum())),
            str(int(inner["event"].sum())),
            *[str(int(counts.get(i, 0))) for i in range(4)],
        ])
    hospitals = sorted(external["hospital"].unique(), key=lambda name: HOSPITAL.get(name, name))
    header = ["Hospital", *[str(fold) for fold in range(5)]]
    hospital_rows = []
    for name in hospitals:
        counts = external[external["hospital"] == name].groupby("outer_fold").size()
        hospital_rows.append([HOSPITAL.get(name, name), *[str(int(counts.get(fold, 0))) for fold in range(5)]])
    return event_rows, header, hospital_rows


def main() -> None:
    seeds = pd.read_csv(OUT / "seeds.csv")
    single = pd.read_csv(SINGLE)
    single = single[single["role"] == "external_test"]
    single_mean = single[["AUROC", "sensitivity", "specificity"]].mean()
    reserved = pd.read_csv(OUT / "modelos_sin_etiqueta" / "modelos_sin_etiqueta.csv")
    reserved_of = {int(row.repeat): row for row in reserved.itertuples(index=False)}

    summary = []
    packs = []
    for repeat in range(10):
        folder = OUT / f"rep_{repeat:02d}"
        rend = pd.read_csv(folder / "alarma" / "rendimiento_por_vuelta.csv")
        intervals = pd.read_csv(folder / "alarma" / "intervalos_bootstrap.csv")
        cuts = pd.read_csv(folder / "alarma" / "cortes.csv")
        partitions = pd.read_csv(folder / "partitions.csv")
        leads = pd.read_csv(folder / "alarma" / "anticipacion_por_paciente.csv")["first_tp_lead"].dropna()
        external = rend[rend["role"] == "external_test"]
        summary.append({
            "repeat": repeat,
            "AUROC": float(external.AUROC.mean()),
            "sensitivity": float(external.sensitivity.mean()),
            "specificity": float(external.specificity.mean()),
            "PPV": float(external.PPV.mean()),
            "NPV": float(external.NPV.mean()),
            "detected": int(external.patients_detected.sum()),
            "false_alarm": int(external.patients_false_alarm.sum()),
            "lead": float(np.median(leads)),
        })
        packs.append((rend, intervals, cuts, partitions, seeds.loc[seeds["repeat"] == repeat].iloc[0]))
    summary = pd.DataFrame(summary)

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = PAGE_WIDTH
    section.page_height = PAGE_HEIGHT
    section.left_margin = MARGIN
    section.right_margin = MARGIN
    section.top_margin = MARGIN
    section.bottom_margin = MARGIN
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    add_heading(doc, "VAL-2. Validación anidada estratificada, diez repeticiones")
    add_text(
        doc,
        "El apartado 12 del SAP pide una validación anidada de cinco pliegues a nivel de paciente, repetida diez veces, "
        "con semillas derivadas de 20261031. Esta carpeta es esa serie. La entrega de una sola partición, en anidada_estratificada, no se ha modificado.",
    )
    add_heading(doc, "La vuelta", level=2)
    add_text(
        doc,
        "Entran 175 pacientes con perfil, en cinco grupos de 35. Sin perfil quedan 31015, 61004 y 63006. "
        "Cada paciente es prueba externa en una sola vuelta. En cada vuelta se entrenan cuatro LDA internos, con 105 pacientes, y un reajuste con las 140. Son 25 modelos por repetición y 250 en la serie. "
        "El diccionario de cada LDA se construye solo con los documentos de su entrenamiento. K = 6 y 500 pasadas. Cada LDA numera sus topics por su cuenta.",
    )
    add_table(
        doc,
        ["Papel", "Pacientes", "Qué hace ese LDA"],
        [
            ["train", "105", "Ajustan un LDA interno. Con su desenlace se etiqueta favorable o desfavorable."],
            ["held out", "35", "Quedan fuera de ese LDA. Reciben esa etiqueta y eligen el corte."],
            ["refit", "140", "Ajustan el reajuste. La etiqueta se calcula otra vez con su desenlace."],
            ["external", "35", "El reajuste se les aplica. El corte elegido con las 140 se conserva."],
        ],
        widths_for([2.2, 2.2, 10]),
        size=10,
        wrap=True,
    )
    add_text(
        doc,
        "Las semillas salen de numpy Generator(20261031), tres enteros por repetición: partición externa, base de la partición interna y arranque del LDA. "
        "La partición interna de la vuelta f usa esa base más f. La semilla solo baraja dentro de cada cruce de hospital, evento y grupo tumoral. "
        "Cada paciente entra en el grupo con menos gente y, a igualdad, en el que tiene menos de su hospital, luego menos de su evento y luego menos de su grupo tumoral. "
        "El hospital es el primer dígito del id. La lista de semillas está en seeds.csv.",
    )
    add_heading(doc, "Etiqueta, bloques y corte", level=2)
    add_text(
        doc,
        "La etiqueta es la regla 3/3. En cada modelo, los tres topics con la Kaplan–Meier a 365 días más alta son favorables y los tres con la más baja son desfavorables. "
        "La curva usa solo a las pacientes que entrenaron ese modelo y el tiempo Tiempo_obs_DB1, desde el inicio del tratamiento. Una paciente con menos de 28 días de esa observación no entra en la curva. "
        "Si el seguimiento acaba antes del día 365 y la curva sigue por encima de cero, ese valor es un techo. El topic se marca desfavorable solo cuando el tercero de los topics que sí llegan queda por encima del techo. Si no, el modelo se deja sin etiqueta.",
    )
    add_text(
        doc,
        "Los bloques siguen la regla del mensual colapsado. Cada mes completo son 30 días de perfil, sin solaparse, desde el primer día. "
        "Si el perfil no acaba en un múltiplo de 30, el último mes se guarda igualmente: con los últimos 30 embeddings si hay al menos 30 días, o con el bloque corto entero si no se llega. "
        "El pUF de un bloque es la suma de las probabilidades de los topics desfavorables del modelo que dio esas probabilidades. "
        "La carga de una decisión es la suma de los tres pUF consecutivos, y la decisión se fecha en el día 30 por el número de mes. "
        "Los meses 1 y 2, y los meses finales incompletos, están en el archivo de bloques. Entran en una decisión solo si esa pareja está en las 1013. "
        "block_probabilities.csv guarda la probabilidad de cada bloque y bloques_puf.csv le añade el pUF. En cada repetición son 9125 filas.",
    )
    add_text(
        doc,
        "El corte y las métricas usan las 1013 parejas de objetivo.csv: 114 pacientes, 155 positivas y 858 negativas. "
        "En el archivo de decisiones cada pareja aparece cinco veces, una externa y cuatro internas: 5065 filas. "
        "El horizonte es de 120 días y el reloj de la decisión es Tiempo_obs_digital. Hay un corte por vuelta, elegido con las decisiones internas. Si dos cortes empatan en Youden, se guarda el más bajo. "
        "El intervalo de cada vuelta usa 2000 remuestreos de pacientes, semilla 20261020.",
    )

    add_heading(doc, "Las diez repeticiones, prueba externa")
    add_text(
        doc,
        "Cada fila es la media de las cinco vueltas externas de esa repetición. Detectadas y falsas alarmas son la suma: "
        "45 pacientes con un punto positivo en las 1013 y 69 sin evento clínico. El adelanto es la mediana, en días, de las detectadas.",
    )
    add_table(
        doc,
        ["Repetición", "AUROC", "Sens.", "Espec.", "VPP", "VPN", "Detectadas", "Falsas alarmas", "Adelanto"],
        [
            [
                str(int(row.repeat)),
                num(row.AUROC, 3),
                num(row.sensitivity),
                num(row.specificity),
                num(row.PPV),
                num(row.NPV),
                f"{int(row.detected)} de 45",
                f"{int(row.false_alarm)} de 69",
                f"{row.lead:.0f} días",
            ]
            for row in summary.itertuples(index=False)
        ] + [[
            "Media",
            num(summary.AUROC.mean(), 3),
            num(summary.sensitivity.mean()),
            num(summary.specificity.mean()),
            num(summary.PPV.mean()),
            num(summary.NPV.mean()),
            f"{summary.detected.mean():.1f} de 45",
            f"{summary.false_alarm.mean():.1f} de 69",
            f"{summary.lead.mean():.0f} días",
        ]],
        widths_for([2.2, 1.6, 1.5, 1.5, 1.5, 1.5, 2.4, 2.8, 2.2]),
        size=10,
    )
    add_text(
        doc,
        f"Entre las diez medias el AUROC va de {num(summary.AUROC.min(), 3)} a {num(summary.AUROC.max(), 3)} "
        f"y la desviación es {num(summary.AUROC.std(ddof=1), 3)}. "
        f"La partición única ya entregada, semilla externa 20261103, tiene AUROC externo {num(single_mean.AUROC, 3)}, "
        f"sensibilidad {num(single_mean.sensitivity)} y especificidad {num(single_mean.specificity)}, "
        f"y detecta {int(single.patients_detected.sum())} de {int(single.patients_with_event.sum())}.",
    )

    metric_header = ["Vuelta", "Papel", "Decisiones", "TP", "FN", "FP", "TN", "Sens.", "Espec.", "VPP", "VPN", "AUROC"]
    metric_widths = widths_for([1.3, 1.8, 2.0, 1.0, 1.0, 1.0, 1.0, 3.2, 3.2, 3.2, 3.2, 3.2])
    for repeat, (rend, intervals, cuts, partitions, seed) in enumerate(packs):
        add_heading(doc, f"Repetición {repeat}")
        add_text(
            doc,
            f"Semilla externa {int(seed.outer_seed)}. Semilla interna {int(seed.inner_seed)} más el número de vuelta. Semilla del LDA {int(seed.lda_seed)}.",
        )
        add_text(doc, "Reparto. Las 35 de cada vuelta externa y, dentro de las 140, los eventos de cada grupo interno.")
        add_table(
            doc,
            ["Vuelta", "Pacientes", "Eventos en las 35", "Eventos en las 140", "Interno 0", "Interno 1", "Interno 2", "Interno 3"],
            partition_tables(partitions)[0],
            widths_for([1.4, 1.8, 2.6, 2.8, 1.8, 1.8, 1.8, 1.8]),
            size=10,
        )
        hospital_rows_header = partition_tables(partitions)
        add_text(doc, "Pacientes de cada hospital en las cinco vueltas externas.")
        add_table(
            doc,
            hospital_rows_header[1],
            hospital_rows_header[2],
            widths_for([3.2, 1.2, 1.2, 1.2, 1.2, 1.2]),
            size=10,
        )
        if repeat in reserved_of:
            item = reserved_of[repeat]
            add_text(
                doc,
                f"El modelo interno {item.model} queda sin etiqueta: {item.topics_sin_soporte_a_365}. "
                f"{int(item.decisiones_1013_sin_puntuacion)} decisiones internas de las 1013 no tienen puntuación y no entran en el corte ni en la tabla de resultados. "
                f"El reajuste {item.reajuste_externo_de_la_misma_vuelta} sí tiene mapa 3/3, y las 1013 externas puntúan.",
            )
        add_text(doc, "Corte elegido con las decisiones internas de las 1013. Se conserva para las 35 externas.")
        cut_rows = []
        for row in cuts.sort_values("outer_fold").itertuples(index=False):
            cut_rows.append([
                str(int(row.outer_fold)),
                num(row.threshold, 3),
                str(int(row.n_decisions)),
                num(row.sensitivity),
                num(row.specificity),
                num(row.youden),
            ])
        add_table(
            doc,
            ["Vuelta", "Corte", "Decisiones", "Sensibilidad", "Especificidad", "Youden"],
            cut_rows,
            widths_for([1.4, 1.8, 2.2, 2.4, 2.4, 1.8]),
            size=10,
        )
        merged = rend.merge(intervals, on=["outer_fold", "role"], suffixes=("", "_i"))
        merged["_role"] = merged["role"].map({"held_out": 0, "external_test": 1})
        merged = merged.sort_values(["outer_fold", "_role"])
        result_rows = []
        for row in merged.itertuples(index=False):
            result_rows.append([
                str(int(row.outer_fold)),
                ROLE[row.role],
                str(int(row.n_decisions)),
                str(int(row.TP)),
                str(int(row.FN)),
                str(int(row.FP)),
                str(int(row.TN)),
                band(row.sensitivity, row.sensitivity_lo, row.sensitivity_hi),
                band(row.specificity, row.specificity_lo, row.specificity_hi),
                band(row.PPV, row.PPV_lo, row.PPV_hi),
                band(row.NPV, row.NPV_lo, row.NPV_hi),
                band(row.AUROC, row.AUROC_lo, row.AUROC_hi),
            ])
        add_text(doc, "Resultados. Held out es el corte en los datos que lo eligieron. External es la comprobación, con el reajuste y ese corte.")
        add_table(doc, metric_header, result_rows, metric_widths, size=8)
        patient_rows = []
        ordered = rend.copy()
        ordered["_role"] = ordered["role"].map({"held_out": 0, "external_test": 1})
        for row in ordered.sort_values(["outer_fold", "_role"]).itertuples(index=False):
            patient_rows.append([
                str(int(row.outer_fold)),
                ROLE[row.role],
                str(int(row.patients_with_event)),
                str(int(row.patients_detected)),
                str(int(row.patients_false_alarm)),
            ])
        add_text(doc, "Por paciente. Con evento: al menos un punto positivo en las 1013. Detectadas: al menos una alarma en uno de esos puntos.")
        add_table(
            doc,
            ["Vuelta", "Papel", "Con evento", "Detectadas", "Sin evento y con falsa alarma"],
            patient_rows,
            widths_for([1.6, 2.2, 2.4, 2.4, 5.5]),
            size=10,
        )

    add_heading(doc, "Tres modelos sin etiqueta")
    add_text(
        doc,
        "De los 250 LDA, estos tres internos no se parten en 3/3. Las curvas están en modelos_sin_etiqueta/. "
        "La adjudicación clínica de la entrega única no se repite: esos mapas nombran los topics de aquellos 25 modelos y no se trasladan a un LDA nuevo.",
    )
    add_table(
        doc,
        ["Repetición", "Modelo", "Topic sin año completo", "Decisiones sin puntuación", "Reajuste externo"],
        [
            [
                str(int(row.repeat)),
                row.model,
                row.topics_sin_soporte_a_365,
                str(int(row.decisiones_1013_sin_puntuacion)),
                row.reajuste_externo_de_la_misma_vuelta,
            ]
            for row in reserved.itertuples(index=False)
        ],
        widths_for([2, 3.2, 6.5, 4.2, 3.2]),
        size=10,
    )
    add_text(
        doc,
        "Cada repetición guarda su reparto en partitions.csv y outer_folds.csv, los 25 modelos en models/, "
        "las 5065 decisiones en alarma/decisiones.csv, el corte en cortes.csv y el rendimiento en rendimiento_por_vuelta.csv.",
    )
    destination = OUT / "informe.docx"
    doc.save(destination)
    print(f"→ {destination}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figures + Word report for missingness point 2 (coverage + availability comparator).

Reads existing CSVs under results/missingness/Real_coverage (coverage + availability_comparator/)
and writes figures + report.docx into results/missingness/Real_coverage/.

    .venv/bin/python scripts/03_analysis/alarm/build_punto2_informe.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsmap, qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[3]
COV = ROOT / "results/missingness/Real_coverage"
CMP = ROOT / "results/missingness/Real_coverage/availability_comparator"
OUT = ROOT / "results/missingness/Real_coverage"

# Same palette / layout as results/missingness/Sensitivity_to_artificial_missingness/report.docx
HEADER_FILL = "2F5D50"
ZEBRA = "F4F1EA"
INK = "1C1B19"
TABLE_WIDTH_DXA = 10206  # ~18 cm usable width with 1.5 cm margins on A4
FIG_WIDTH_IN = 7.0
BAND_ORDER = ["<50%", "50-80%", ">=80%"]
BAND_LABEL = {"<50%": "<50%", "50-80%": "50–80%", ">=80%": "≥80%"}
MODEL_ORDER = [
    "availability_only_oof",
    "dcabp_load_frozen",
    "dcabp_plus_availability_oof",
]
MODEL_LABEL = {
    "availability_only_oof": "Availability only",
    "dcabp_load_frozen": "DCABP load (frozen cut)",
    "dcabp_plus_availability_oof": "DCABP + availability",
}
THR = 1.0596384140662845
N_BOOT = 1000
BOOT_SEED = 20261116
# practiced_sport is almost always ≥80% here — not useful to stratify
SKIP_VAR_BAND = {"practiced_sport"}

# Register SVG namespace used by Word dual PNG+SVG pictures
nsmap["asvg"] = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"


def _band_of(cov: float) -> str:
    if not np.isfinite(cov):
        return "NA"
    if cov < 0.50:
        return "<50%"
    if cov < 0.80:
        return "50-80%"
    return ">=80%"


def compute_variable_band_perf(window_days: int = 90) -> pd.DataFrame:
    """Frozen-alarm AUROC by SAP band of each variable's own coverage (pre–VQ-VAE)."""
    from sklearn.metrics import roc_auc_score

    cov_df = pd.read_csv(COV / "coverage_by_decision.csv")
    sub = cov_df.loc[(cov_df.window_days == window_days) & (cov_df.emitted == 1)].copy()
    var_cols = [c for c in sub.columns if c.startswith("cov_")]
    variables = [c.replace("cov_", "", 1) for c in var_cols if c.replace("cov_", "", 1) not in SKIP_VAR_BAND]
    for var in variables:
        sub[f"band_{var}"] = sub[f"cov_{var}"].map(_band_of)

    # Point estimates
    point = {}
    for var in variables:
        col = f"cov_{var}"
        bcol = f"band_{var}"
        for band in BAND_ORDER:
            part = sub.loc[sub[bcol] == band]
            key = (var, band)
            n_dec = int(len(part))
            n_pat = int(part["id"].nunique()) if n_dec else 0
            n_pos = int(part["y"].sum()) if n_dec else 0
            mean_cov = float(part[col].mean()) if n_dec else np.nan
            if n_dec < 20 or part["y"].nunique() < 2:
                point[key] = dict(
                    n_patients=n_pat,
                    n_decisions=n_dec,
                    n_positive=n_pos,
                    mean_coverage=mean_cov,
                    AUROC=np.nan,
                    sensitivity=np.nan,
                    specificity=np.nan,
                )
                continue
            y = part["y"].to_numpy(int)
            s = part["load"].to_numpy(float)
            pred = s >= THR
            tp = int(((pred) & (y == 1)).sum())
            fn = int(((~pred) & (y == 1)).sum())
            fp = int(((pred) & (y == 0)).sum())
            tn = int(((~pred) & (y == 0)).sum())
            point[key] = dict(
                n_patients=n_pat,
                n_decisions=n_dec,
                n_positive=n_pos,
                mean_coverage=mean_cov,
                AUROC=float(roc_auc_score(y, s)),
                sensitivity=tp / (tp + fn) if (tp + fn) else np.nan,
                specificity=tn / (tn + fp) if (tn + fp) else np.nan,
            )

    # One patient bootstrap stream shared across all variable×band cells
    patients = sub["id"].drop_duplicates().to_numpy()
    by = {pid: g for pid, g in sub.groupby("id", sort=False)}
    rng = np.random.default_rng(BOOT_SEED + window_days + 17)
    boot_store = {key: [] for key in point}
    for _ in range(N_BOOT):
        draw = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([by[pid] for pid in draw], ignore_index=True)
        for var in variables:
            bcol = f"band_{var}"
            for band in BAND_ORDER:
                bb = boot.loc[boot[bcol] == band]
                if len(bb) < 10 or bb["y"].nunique() < 2:
                    continue
                boot_store[(var, band)].append(float(roc_auc_score(bb["y"], bb["load"])))

    rows = []
    for var in variables:
        for band in BAND_ORDER:
            key = (var, band)
            p = point[key]
            arr = np.asarray(boot_store[key], float)
            lo = float(np.quantile(arr, 0.025)) if len(arr) >= 20 and np.isfinite(p["AUROC"]) else np.nan
            hi = float(np.quantile(arr, 0.975)) if len(arr) >= 20 and np.isfinite(p["AUROC"]) else np.nan
            rows.append(
                {
                    "window_days": window_days,
                    "variable": var,
                    "band": band,
                    **p,
                    "AUROC_lo": lo,
                    "AUROC_hi": hi,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(COV / "performance_by_variable_band.csv", index=False)
    return out


def fmt(x: float, d: int = 3) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


def band_ci(point: float, lo: float, hi: float, d: int = 3) -> str:
    if not np.isfinite(point):
        return "—"
    if not np.isfinite(lo) or not np.isfinite(hi):
        return fmt(point, d)
    return f"{fmt(point, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def shade(cell, color: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    for old in props.findall(qn("w:shd")):
        props.remove(old)
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:color"), "auto")
    fill.set(qn("w:fill"), color)
    props.append(fill)


def set_cell_width(cell, dxa: int) -> None:
    props = cell._tc.get_or_add_tcPr()
    for old in props.findall(qn("w:tcW")):
        props.remove(old)
    tcw = OxmlElement("w:tcW")
    tcw.set(qn("w:w"), str(int(dxa)))
    tcw.set(qn("w:type"), "dxa")
    props.append(tcw)


def _calibri(run, *, size_pt: float | None = None, bold: bool | None = None, color: str | None = None) -> None:
    run.font.name = "Calibri"
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), "Calibri")
    fonts.set(qn("w:hAnsi"), "Calibri")
    fonts.set(qn("w:eastAsia"), "Calibri")
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    style = "Heading 1" if level == 1 else "Heading 2"
    p = doc.add_paragraph(style=style)
    pPr = p._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:spacing")):
        pPr.remove(old)
    spacing = OxmlElement("w:spacing")
    if level == 1:
        spacing.set(qn("w:before"), "0")
        spacing.set(qn("w:after"), "120")
    else:
        spacing.set(qn("w:before"), "280")
        spacing.set(qn("w:after"), "120")
    pPr.append(spacing)
    run = p.add_run(text)
    _calibri(run, size_pt=16.0 if level == 1 else None, color=INK)


def add_text(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="Normal")
    run = p.add_run(text)
    _calibri(run)


def _relate_svg(doc: Document, svg: Path) -> str:
    """Attach an SVG part the way Word dual-format pictures do (python-docx cannot decode SVG)."""
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.opc.part import Part
    from docx.opc.packuri import PackURI

    part = doc.part
    package = part.package
    existing = {str(p.partname) for p in package.iter_parts()}
    n = 1
    while True:
        name = PackURI(f"/word/media/image{n}.svg")
        if str(name) not in existing:
            break
        n += 1
    svg_part = Part(name, "image/svg+xml", svg.read_bytes(), package)
    return part.relate_to(svg_part, RT.IMAGE)


def add_figure(doc: Document, png: Path, svg: Path | None = None) -> None:
    """Embed PNG (+ SVG twin if present), matching the sensitivity informe layout."""
    svg = svg if svg is not None else png.with_suffix(".svg")
    p = doc.add_paragraph(style="Normal")
    pPr = p._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:spacing")):
        pPr.remove(old)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "80")
    spacing.set(qn("w:after"), "40")
    pPr.append(spacing)

    run = p.add_run()
    inline = run.add_picture(str(png), width=Inches(FIG_WIDTH_IN))
    if svg.exists():
        svg_rid = _relate_svg(doc, svg)
        blip = inline._inline.graphic.graphicData.pic.blipFill.blip
        ext_lst = blip.find(qn("a:extLst"))
        if ext_lst is None:
            ext_lst = OxmlElement("a:extLst")
            blip.append(ext_lst)
        ext = OxmlElement("a:ext")
        ext.set("uri", "{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
        ext.append(
            parse_xml(
                '<asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
                f'r:embed="{svg_rid}"/>'
            )
        )
        ext_lst.append(ext)

    cap = doc.add_paragraph(style="Normal")
    run = cap.add_run(f"Figure: {svg.name if svg.exists() else png.name}.")
    _calibri(run, size_pt=10.0)


def add_table(doc: Document, header: list[str], rows: list[list[str]], weight: list[float] | None = None, size: int = 9) -> None:
    """TableGrid + forest header + cream zebra — same tokens as the sensitivity informe."""
    n = len(header)
    if weight is None:
        weight = [1.0] * n
    wsum = float(sum(weight))
    widths = [int(round(TABLE_WIDTH_DXA * w / wsum)) for w in weight]
    widths[-1] += TABLE_WIDTH_DXA - sum(widths)

    table = doc.add_table(rows=1 + len(rows), cols=n, style="Table Grid")
    table.autofit = False
    tbl = table._tbl
    tbl_pr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    if tbl.tblPr is None:
        tbl.insert(0, tbl_pr)

    for tag in ("w:tblW", "w:tblLayout", "w:tblLook"):
        for old in tbl_pr.findall(qn(tag)):
            tbl_pr.remove(old)
    tbl_w = OxmlElement("w:tblW")
    tbl_w.set(qn("w:w"), str(TABLE_WIDTH_DXA))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_w)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    look = OxmlElement("w:tblLook")
    look.set(qn("w:val"), "04A0")
    look.set(qn("w:firstRow"), "1")
    look.set(qn("w:lastRow"), "0")
    look.set(qn("w:firstColumn"), "1")
    look.set(qn("w:lastColumn"), "0")
    look.set(qn("w:noHBand"), "0")
    look.set(qn("w:noVBand"), "1")
    tbl_pr.append(look)

    for existing in tbl.findall(qn("w:tblGrid")):
        tbl.remove(existing)
    grid = OxmlElement("w:tblGrid")
    for dxa in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(dxa))
        grid.append(col)
    tbl.insert(1, grid)

    def write_cell(cell, text: str, *, header: bool = False) -> None:
        cell.text = ""
        para = cell.paragraphs[0]
        # drop the empty run left by clearing text
        for child in list(para._p):
            if child.tag == qn("w:r"):
                para._p.remove(child)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(text)
        if header:
            _calibri(run, size_pt=float(size), bold=True, color="FFFFFF")
        else:
            _calibri(run, size_pt=float(size))

    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]
        set_cell_width(cell, widths[j])
        write_cell(cell, h, header=True)
        shade(cell, HEADER_FILL)

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            set_cell_width(cell, widths[j])
            write_cell(cell, str(val))
            if i % 2 == 1:
                shade(cell, ZEBRA)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures() -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    # 1) AUROC by coverage band (90 d)
    perf = pd.read_csv(COV / "performance_by_band.csv")
    p90 = perf[(perf.window_days == 90) & (perf.band.isin(BAND_ORDER))].copy()
    p90["band"] = pd.Categorical(p90["band"], BAND_ORDER, ordered=True)
    p90 = p90.sort_values("band")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(p90))
    y = p90["AUROC"].to_numpy()
    yerr = np.vstack([y - p90["AUROC_lo"].to_numpy(), p90["AUROC_hi"].to_numpy() - y])
    ax.bar(x, y, width=0.62, color=f"#{HEADER_FILL}", edgecolor="none")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#1C1B19", capsize=4, linewidth=1.2)
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_LABEL[b] for b in p90["band"]])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Coverage band (90-day window)")
    ax.set_title("Alarm AUROC by real-coverage band (frozen cut)")
    style_axes(ax)
    fig.tight_layout()
    paths["auroc_band"] = OUT / "fig_auroc_by_band_90d.png"
    fig.savefig(paths["auroc_band"], dpi=200)
    fig.savefig(paths["auroc_band"].with_suffix(".svg"))
    plt.close(fig)

    # 2) Decisions per band + mean coverage (90 d)
    counts = pd.read_csv(COV / "band_counts.csv")
    c90 = counts[(counts.window_days == 90) & (counts.band.isin(BAND_ORDER))].copy()
    c90["band"] = pd.Categorical(c90["band"], BAND_ORDER, ordered=True)
    c90 = c90.sort_values("band")
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(c90))
    bars = ax1.bar(x, c90["n_decisions"], width=0.62, color="#7A9E90", edgecolor="none", label="Decisions")
    ax1.set_ylabel("Decisions")
    ax1.set_xticks(x)
    ax1.set_xticklabels([BAND_LABEL[b] for b in c90["band"]])
    ax1.set_xlabel("Coverage band (90-day window)")
    ax2 = ax1.twinx()
    ax2.plot(x, c90["mean_coverage"], color=f"#{HEADER_FILL}", marker="o", linewidth=1.8, label="Mean coverage")
    ax2.set_ylabel("Mean coverage")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("How many decisions fall in each coverage band")
    style_axes(ax1)
    ax2.spines["top"].set_visible(False)
    for bar, n in zip(bars, c90["n_decisions"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, str(int(n)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    paths["band_counts"] = OUT / "fig_band_counts_90d.png"
    fig.savefig(paths["band_counts"], dpi=200)
    fig.savefig(paths["band_counts"].with_suffix(".svg"))
    plt.close(fig)

    # 3) Availability by variable (no wearable/smartphone split in the report for now)
    av = pd.read_csv(COV / "availability_by_variable.csv").sort_values("mean_coverage")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(av))
    ax.barh(y, av["mean_coverage"], color=f"#{HEADER_FILL}", edgecolor="none", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(av["variable"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean coverage (90 d)")
    ax.set_title("Availability by input variable")
    style_axes(ax)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    paths["by_var"] = OUT / "fig_coverage_by_variable.png"
    fig.savefig(paths["by_var"], dpi=200)
    fig.savefig(paths["by_var"].with_suffix(".svg"))
    plt.close(fig)

    # 4) Comparator AUROC
    models = pd.read_csv(CMP / "performance_models.csv")
    m = models[models["model"].isin(MODEL_ORDER)].copy()
    m["model"] = pd.Categorical(m["model"], MODEL_ORDER, ordered=True)
    m = m.sort_values("model")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(m))
    y = m["AUROC"].to_numpy()
    yerr = np.vstack([y - m["AUROC_lo"].to_numpy(), m["AUROC_hi"].to_numpy() - y])
    colors = ["#8A7355", f"#{HEADER_FILL}", "#5B7C8A"]
    ax.bar(x, y, width=0.62, color=colors, edgecolor="none")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#1C1B19", capsize=4, linewidth=1.2)
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[k] for k in m["model"]], rotation=12, ha="right")
    ax.set_ylim(0.25, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Missingness comparator on the same 1013 landmarks")
    style_axes(ax)
    fig.tight_layout()
    paths["comparator"] = OUT / "fig_comparator_auroc.png"
    fig.savefig(paths["comparator"], dpi=200)
    fig.savefig(paths["comparator"].with_suffix(".svg"))
    plt.close(fig)

    # 5) AUROC heatmap by per-variable coverage band (90 d), annotated with n / positives
    vb_path = COV / "performance_by_variable_band.csv"
    if vb_path.exists():
        vb = pd.read_csv(vb_path)
        vb = vb[vb.window_days == 90].copy()
        pivot = vb.pivot(index="variable", columns="band", values="AUROC").reindex(columns=BAND_ORDER)
        n_piv = vb.pivot(index="variable", columns="band", values="n_decisions").reindex(columns=BAND_ORDER)
        pos_piv = vb.pivot(index="variable", columns="band", values="n_positive").reindex(columns=BAND_ORDER)
        av = pd.read_csv(COV / "availability_by_variable.csv")
        order = [v for v in av.sort_values("mean_coverage")["variable"] if v in pivot.index]
        pivot, n_piv, pos_piv = pivot.reindex(order), n_piv.reindex(order), pos_piv.reindex(order)
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        data = pivot.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=0.85)
        ax.set_xticks(np.arange(len(BAND_ORDER)))
        ax.set_xticklabels([BAND_LABEL[b] for b in BAND_ORDER])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_xlabel("Coverage band of that variable (90 d)")
        ax.set_title("Frozen-alarm AUROC by per-variable coverage\n(cell = AUROC; below = +positives / −negatives)")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                n = n_piv.to_numpy()[i, j]
                pos = pos_piv.to_numpy()[i, j]
                if np.isfinite(val) and np.isfinite(n) and np.isfinite(pos):
                    neg = int(n) - int(pos)
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}\n+{int(pos)} / −{neg}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#1C1B19",
                        linespacing=1.2,
                    )
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#666666")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("AUROC")
        fig.tight_layout()
        paths["var_band"] = OUT / "fig_auroc_by_variable_band_90d.png"
        fig.savefig(paths["var_band"], dpi=200)
        fig.savefig(paths["var_band"].with_suffix(".svg"))
        plt.close(fig)

    return paths


def build_docx(figs: dict[str, Path], var_band: pd.DataFrame | None = None) -> Path:
    band_counts = pd.read_csv(COV / "band_counts.csv")
    perf = pd.read_csv(COV / "performance_by_band.csv")
    by_var = pd.read_csv(COV / "availability_by_variable.csv")
    by_ctr = pd.read_csv(COV / "availability_by_center.csv")
    by_tum = pd.read_csv(COV / "availability_by_tumor.csv")
    cont = pd.read_csv(COV / "continuous_coverage_summary.csv")
    overlap = pd.read_csv(CMP / "emission_overlap.csv").iloc[0]
    models = pd.read_csv(CMP / "performance_models.csv")
    if var_band is None and (COV / "performance_by_variable_band.csv").exists():
        var_band = pd.read_csv(COV / "performance_by_variable_band.csv")

    doc = Document()
    section = doc.sections[0]
    for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, side, Cm(1.5))

    add_heading(doc, "Real coverage and informative missingness", level=1)
    add_text(
        doc,
        "DB1 alarm landmarks only: 1013 decisions (114 patients, 155 positive / 858 negative), "
        "120-day horizon. This is point 2 of the missingness block. Artificial masking (point 1) asks "
        "how tolerant the algorithm is to deleted information. Here the question is whether the alarm "
        "depends on monitoring density, or whether it is enough to observe that a patient stops contributing data.",
    )
    add_text(
        doc,
        "Uncertainty note. In point 1, seeds 101–105 create the artificial masks; brackets are the CI of "
        "that seed-mean. Here there is no deletion experiment, so those seeds are not used. Brackets are "
        "patient-clustered bootstrap intervals (1000 resamples) on the fixed 1013 landmarks.",
    )

    add_heading(doc, "What is in this report", level=2)
    add_text(
        doc,
        "A. Real coverage: for each decision, coverage on the previous 90 days (alarm history) and the last "
        "30 days (recent coverage), as the share of variable–day cells really observed for the nine input "
        "variables before imputation, on the expected calendar (up to 810 and 270 opportunities). SAP bands "
        "fixed a priori: below 50%, 50% to below 80%, and at least 80%. Same frozen alarm and threshold "
        "(1.0596…). Breakdowns by variable, centre and tumour. Extra (before B): same frozen alarm inside "
        "SAP bands defined by each variable’s own coverage, to see whether the type of missingness matters.",
    )
    add_text(
        doc,
        "B. Missingness-only comparator: three scores on the same 1013 points — (1) availability alone "
        "(logistic on the four indicators), (2) frozen DCABP load with the alarm cut, (3) frozen load plus "
        "those indicators in one logistic. The DCABP load itself is not refit. Extra models fit in DB1 only, "
        "with patient-grouped cross-validation; no DB2/DB3 tuning.",
    )

    # ----- A -----
    add_heading(doc, "A. Counts by coverage band", level=2)
    rows = []
    for w in (30, 90):
        sub = band_counts[(band_counts.window_days == w) & (band_counts.band.isin(BAND_ORDER))].copy()
        sub["band"] = pd.Categorical(sub["band"], BAND_ORDER, ordered=True)
        for r in sub.sort_values("band").itertuples(index=False):
            rows.append(
                [
                    str(int(r.window_days)),
                    BAND_LABEL[r.band],
                    str(int(r.n_decisions)),
                    str(int(r.n_patients)),
                    fmt(r.mean_coverage, 3),
                ]
            )
    add_table(doc, ["Window (d)", "Band", "Decisions", "Patients", "Mean coverage"], rows, [1.2, 1.2, 1.3, 1.3, 1.6])
    add_text(
        doc,
        "Coverage is the fraction of expected variable–day cells with a real observation (observed zeros count; "
        "imputed values do not). The expected calendar is the dense 90- or 30-day window ending at the decision "
        "date; holes are kept and are not trimmed to the last received record.",
    )
    add_figure(doc, figs["band_counts"])

    add_heading(doc, "A. Alarm performance by band", level=2)
    add_text(
        doc,
        "The same alarm and cut are applied in every band. Brackets are 95% patient-clustered bootstrap "
        "intervals (1000 resamples). Every decision keeps a score after VQ-VAE, so no band is empty for performance.",
    )
    rows = []
    for w in (90, 30):
        sub = perf[perf.window_days == w].copy()
        order = BAND_ORDER + ["all"]
        sub["band"] = pd.Categorical(sub["band"], order, ordered=True)
        for r in sub.sort_values("band").itertuples(index=False):
            label = BAND_LABEL.get(r.band, "all")
            rows.append(
                [
                    str(int(r.window_days)),
                    label,
                    str(int(r.n_patients)),
                    str(int(r.n_decisions)),
                    str(int(r.n_positive)),
                    band_ci(r.AUROC, r.AUROC_lo, r.AUROC_hi),
                    band_ci(r.sensitivity, r.sensitivity_lo, r.sensitivity_hi),
                    band_ci(r.specificity, r.specificity_lo, r.specificity_hi),
                ]
            )
    add_table(
        doc,
        ["Window", "Band", "Patients", "Decisions", "Positive", "AUROC", "Sensitivity", "Specificity"],
        rows,
        [0.9, 1.0, 1.0, 1.1, 1.0, 2.0, 2.0, 2.0],
        size=8,
    )
    add_figure(doc, figs["auroc_band"])

    add_heading(doc, "A. Availability by variable (90 d)", level=2)
    rows = [
        [
            r.variable,
            fmt(r.mean_coverage, 3),
            fmt(r.median_coverage, 3),
            fmt(r.p25, 3),
            fmt(r.p75, 3),
        ]
        for r in by_var.itertuples(index=False)
    ]
    add_table(doc, ["Variable", "Mean", "Median", "P25", "P75"], rows, [2.6, 1.2, 1.2, 1.2, 1.2])
    add_figure(doc, figs["by_var"])

    add_heading(doc, "A. Availability by centre and tumour (90 d)", level=2)
    rows = [
        [
            r.center.replace("PMP_", ""),
            str(int(r.n_patients)),
            str(int(r.n_decisions)),
            fmt(r.mean_coverage_90, 3),
            fmt(r.mean_pct_days_with_data, 3),
        ]
        for r in by_ctr.itertuples(index=False)
    ]
    add_table(doc, ["Centre", "Patients", "Decisions", "Coverage", "Days with data"], rows, [2.0, 1.2, 1.2, 1.3, 1.5])
    rows = [
        [
            r.tumor,
            str(int(r.n_patients)),
            str(int(r.n_decisions)),
            fmt(r.mean_coverage_90, 3),
            fmt(r.mean_pct_days_with_data, 3),
        ]
        for r in by_tum.itertuples(index=False)
    ]
    add_table(doc, ["Tumour", "Patients", "Decisions", "Coverage", "Days with data"], rows, [1.8, 1.2, 1.2, 1.3, 1.5])

    add_heading(doc, "A. Continuous coverage and interaction with the score", level=2)
    add_text(
        doc,
        "Besides the SAP bands, coverage is kept as a continuous value. Associations with the load and with the "
        "120-day outcome, AUROC of coverage alone, AUROC of the frozen load, and an exploratory "
        "load + load×coverage score are reported. That interaction score is ranked for AUROC only "
        "(no new classification threshold). All brackets are patient-clustered bootstrap intervals (1000 resamples).",
    )
    add_text(
        doc,
        "Spearman ρ is used instead of Pearson because coverage is bounded in [0, 1] and the load–coverage "
        "relationship need not be linear: Spearman ranks both variables and measures monotone association "
        "(−1 to +1). Near 0 means little monotone link; the bootstrap interval crossing 0 means that association "
        "is compatible with none in these data. The same reading applies to ρ(coverage, 120-day outcome).",
    )
    rows = []
    for r in cont.itertuples(index=False):
        rows.append(
            [
                str(int(r.window_days)),
                fmt(r.mean_coverage, 3),
                band_ci(r.spearman_load_coverage, r.spearman_load_coverage_lo, r.spearman_load_coverage_hi),
                band_ci(r.spearman_coverage_outcome, r.spearman_coverage_outcome_lo, r.spearman_coverage_outcome_hi),
                band_ci(r.AUROC_coverage_alone, r.AUROC_coverage_alone_lo, r.AUROC_coverage_alone_hi),
                band_ci(r.AUROC_load, getattr(r, "AUROC_load_lo", float("nan")), getattr(r, "AUROC_load_hi", float("nan"))),
                band_ci(r.AUROC_load_plus_loadxcov, r.AUROC_load_plus_loadxcov_lo, r.AUROC_load_plus_loadxcov_hi),
            ]
        )
    add_table(
        doc,
        ["Window", "Mean cov.", "ρ load–cov", "ρ cov–outcome", "AUROC cov.", "AUROC load", "AUROC load+load×cov"],
        rows,
        [0.9, 1.0, 1.7, 1.7, 1.6, 1.1, 2.0],
        size=8,
    )

    # ----- A extra: per-variable coverage bands -----
    add_heading(doc, "A. Extra — alarm by per-variable coverage band", level=2)
    add_text(
        doc,
        "Exploratory addition (not a SAP primary table). Instead of the joint coverage of all nine variables, "
        "each decision is placed in the SAP bands using only that variable’s own 90-day coverage (still measured "
        "before VQ-VAE). The frozen load and cut are applied inside each cell. practiced_sport is omitted: almost "
        "all decisions sit in ≥80%. Brackets are patient-clustered bootstrap intervals on AUROC.",
    )
    add_text(
        doc,
        "Each metric row carries the class counts of that cell: positives, negatives and prevalence "
        "(positives / decisions). AUROC, sensitivity and specificity are computed only on those counts; "
        "the bootstrap interval widens when either class is thin. Do not read a high point AUROC in a small "
        "band as an effect of missingness — check +/− and the interval in the same row. The score is still the "
        "post–VQ-VAE load; the band only labels pre-imputation completeness of that variable.",
    )
    if var_band is not None and len(var_band):
        vb = var_band[var_band.window_days == 90].copy()
        vb["n_negative"] = vb["n_decisions"] - vb["n_positive"]
        vb["prevalence"] = vb["n_positive"] / vb["n_decisions"].replace(0, np.nan)
        vb["AUROC_ci_width"] = vb["AUROC_hi"] - vb["AUROC_lo"]
        mean_cov = by_var.set_index("variable")["mean_coverage"]
        vb["_ord"] = vb["variable"].map(lambda v: float(mean_cov.get(v, 1.0)))
        vb["band"] = pd.Categorical(vb["band"], BAND_ORDER, ordered=True)
        vb = vb.sort_values(["_ord", "band"])
        long_rows = []
        for r in vb.itertuples(index=False):
            long_rows.append(
                [
                    r.variable,
                    BAND_LABEL[r.band],
                    str(int(r.n_positive)),
                    str(int(r.n_negative)),
                    fmt(r.prevalence, 3) if np.isfinite(r.prevalence) else "—",
                    band_ci(r.AUROC, r.AUROC_lo, r.AUROC_hi),
                    fmt(r.AUROC_ci_width, 3) if np.isfinite(r.AUROC_ci_width) else "—",
                    fmt(r.sensitivity, 3) if np.isfinite(r.sensitivity) else "—",
                    fmt(r.specificity, 3) if np.isfinite(r.specificity) else "—",
                ]
            )
        add_table(
            doc,
            ["Variable", "Band", "Pos+", "Neg−", "Prev.", "AUROC", "CI width", "Sens.", "Spec."],
            long_rows,
            [1.9, 1.1, 0.9, 0.9, 0.9, 2.3, 1.0, 0.9, 0.9],
            size=7,
        )
        if "var_band" in figs:
            add_figure(doc, figs["var_band"])
        add_text(
            doc,
            "CI width is the bootstrap interval length (AUROC_hi − AUROC_lo): larger width means less precise "
            "discrimination in that band, usually from few positives or few negatives. "
            "Example: sleep_start <50% is +16 / −134 (prev. 0.107), AUROC 0.80 with CI width ~0.29; "
            "≥80% is +121 / −633 (prev. 0.160), AUROC 0.69 with narrower CI ~0.18 — the intervals still overlap. "
            "Thin middle bands (e.g. app_usage 50–80%: +14 / −85) show the widest CIs.",
        )

    # ----- B -----
    add_heading(doc, "B. Availability indicators", level=2)
    add_text(
        doc,
        "For each decision, on the past-only 90-day window before imputation: (1) proportion of absent days; "
        "(2) number of gaps of at least 48 hours; (3) days since the last valid record; "
        "(4) monitoring interrupted now (current gap ≥ 2 days).",
    )

    add_heading(doc, "B. Where each method can emit a prediction", level=2)
    add_text(
        doc,
        "On these 1013 landmarks both the DCABP load and the availability features are defined for every "
        "decision, so the three scores are compared on the full set.",
    )
    add_table(
        doc,
        ["Decisions", "Patients", "DCABP", "Availability", "Both", "DCABP only", "Avail. only", "Neither"],
        [
            [
                str(int(overlap.n_decisions)),
                str(int(overlap.n_patients)),
                str(int(overlap.n_dcabp)),
                str(int(overlap.n_availability)),
                str(int(overlap.n_both)),
                str(int(overlap.n_dcabp_only)),
                str(int(overlap.n_availability_only)),
                str(int(overlap.n_neither)),
            ]
        ],
        [1.2, 1.1, 1.0, 1.2, 0.9, 1.1, 1.1, 1.0],
    )

    add_heading(doc, "B. Comparator performance", level=2)
    add_text(
        doc,
        "How the three scores are built. The SAP (§17) asks for a missingness-only comparator on the same "
        "horizon as the alarm, but does not prescribe the formula that combines the availability indicators. "
        "Section 15 uses a penalised logistic when two digital signals are combined; we follow that practice "
        "for the missingness arm only. (1) Availability only: logistic on the four indicators, patient-grouped "
        "5-fold OOF probabilities, Youden cut on the training fold. (2) DCABP: frozen three-block load and "
        "alarm cut 1.0596… (not replaced by a logistic). (3) DCABP + availability: frozen load plus the four "
        "indicators in one logistic (same OOF scheme). Brackets are patient-clustered bootstrap intervals on "
        "those scores — not deletion-seed CIs.",
    )
    show = models[models["model"].isin(MODEL_ORDER)].copy()
    show["model"] = pd.Categorical(show["model"], MODEL_ORDER, ordered=True)
    rows = []
    for r in show.sort_values("model").itertuples(index=False):
        rows.append(
            [
                MODEL_LABEL[r.model],
                str(int(r.n_patients)),
                str(int(r.n_scored)),
                str(int(r.n_positive)),
                band_ci(r.AUROC, r.AUROC_lo, r.AUROC_hi),
                band_ci(r.sensitivity, r.sensitivity_lo, r.sensitivity_hi),
                band_ci(r.specificity, r.specificity_lo, r.specificity_hi),
                band_ci(r.PPV, r.PPV_lo, r.PPV_hi),
                band_ci(r.NPV, r.NPV_lo, r.NPV_hi),
            ]
        )
    add_table(
        doc,
        ["Model", "Patients", "N", "Pos.", "AUROC", "Sens.", "Spec.", "PPV", "NPV"],
        rows,
        [2.0, 0.9, 0.7, 0.7, 1.7, 1.5, 1.5, 1.5, 1.5],
        size=8,
    )
    add_figure(doc, figs["comparator"])
    add_text(
        doc,
        "Availability alone does not match the DCABP load (AUROC 0.402 vs 0.703). Adding the availability "
        "indicators to the load does not improve discrimination over the frozen load alone.",
    )

    add_heading(doc, "Files", level=2)
    add_text(
        doc,
        "Coverage and comparator outputs: results/missingness/Real_coverage/ (comparator CSVs under availability_comparator/). Scripts: run_coverage_real_objetivo.py, "
        "run_missingness_comparator_objetivo.py, build_punto2_informe.py. Figures in this folder: "
        "fig_band_counts_90d, fig_auroc_by_band_90d, fig_coverage_by_variable, "
        "fig_auroc_by_variable_band_90d, fig_comparator_auroc (.png / .svg).",
    )

    dest = OUT / "report.docx"
    try:
        doc.save(dest)
    except PermissionError:
        dest = OUT / "report_with_figures.docx"
        doc.save(dest)
    return dest


def main() -> None:
    vb_path = COV / "performance_by_variable_band.csv"
    if vb_path.exists():
        print(f"reusing {vb_path.relative_to(ROOT)}")
        var_band = pd.read_csv(vb_path)
    else:
        print("computing per-variable coverage bands…")
        var_band = compute_variable_band_perf(90)
    figs = make_figures()
    dest = build_docx(figs, var_band=var_band)
    (OUT / "report.md").write_text(
        "# Real coverage and informative missingness\n\n"
        "Word report with tables and figures:\n\n"
        f"`{dest.relative_to(ROOT)}`\n\n"
        "- Coverage CSVs and figures in this folder\n"
        "- Comparator CSVs: `availability_comparator/`\n"
        "- Extra A: `performance_by_variable_band.csv`\n"
    )
    print(f"→ figures: {len(figs)}")
    for k, p in figs.items():
        print(f"  {k}: {p.relative_to(ROOT)}")
    print(f"→ {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

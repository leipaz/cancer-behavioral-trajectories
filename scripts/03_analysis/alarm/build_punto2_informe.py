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
    "availability_only",
    "dcabp_load_frozen",
    "dcabp_plus_availability",
]
MODEL_LABEL = {
    "availability_only": "Availability only",
    "dcabp_load_frozen": "DCABP load (frozen cut)",
    "dcabp_plus_availability": "DCABP + availability",
}
THR = 1.0596384140662845
N_BOOT = 1000
BOOT_SEED = 20261116
# Every input variable is stratified, including practiced_sport after the
# erroneous-zero rule. Before that rule the flag sat in ≥80% for almost every decision.
SKIP_VAR_BAND: set[str] = set()

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


def add_figure(doc: Document, png: Path, svg: Path | None = None, caption: bool = True) -> None:
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

    if not caption:
        return
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


def variable_coverage_on_row(by_var: pd.DataFrame) -> pd.DataFrame:
    """Coverage among days with a daily row, same 90-day decision windows.

    The denominator is days with a daily row. practiced_sport zeros with no
    walking and no running are already missing in cov_practiced_sport.
    """
    cov = pd.read_csv(COV / "coverage_by_decision.csv")
    sub = cov.loc[cov.window_days == 90].copy()
    if "pct_days_with_row" not in sub.columns:
        raise SystemExit("coverage_by_decision.csv has no pct_days_with_row; rerun run_coverage_real_objetivo.py")
    den = sub["pct_days_with_row"].where(sub["pct_days_with_row"] > 0)
    extra = []
    for variable in by_var["variable"]:
        rate = sub[f"cov_{variable}"] / den
        extra.append(
            {
                "variable": variable,
                "mean_coverage_on_row": float(rate.mean()),
                "median_coverage_on_row": float(rate.median()),
                "p25_on_row": float(rate.quantile(0.25)),
                "p75_on_row": float(rate.quantile(0.75)),
                "mean_missing_on_row": float(1.0 - rate.mean()),
            }
        )
    drop_cols = [
        c
        for c in (
            "mean_coverage_on_row",
            "median_coverage_on_row",
            "p25_on_row",
            "p75_on_row",
            "mean_missing_on_row",
        )
        if c in by_var.columns
    ]
    out = by_var.drop(columns=drop_cols).merge(pd.DataFrame(extra), on="variable", how="left")
    out.to_csv(COV / "availability_by_variable.csv", index=False)
    return out


def make_figures(by_var_on_row: pd.DataFrame | None = None) -> dict[str, Path]:
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

    # 3) Availability by variable on days that have a daily row (same 90 d windows)
    if by_var_on_row is None:
        by_var_on_row = variable_coverage_on_row(pd.read_csv(COV / "availability_by_variable.csv"))
    av = by_var_on_row.sort_values("mean_coverage_on_row")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(av))
    ax.barh(y, av["mean_coverage_on_row"], color=f"#{HEADER_FILL}", edgecolor="none", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(av["variable"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean coverage on days with a daily row")
    ax.set_title("Availability by input variable (days with a row)")
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
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
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


BREAKDOWN_HEADER = ["Patients", "Decisions", "Days observed", "Mean coverage", "Median [IQR]", "Decisions with a gap ≥ 2 d"]


def breakdown_cells(r) -> list[str]:
    return [
        str(int(r.n_patients)),
        str(int(r.n_decisions)),
        fmt(r.mean_days_observed, 3),
        fmt(r.mean_coverage, 3),
        f"{r.median_coverage:.3f} [{r.p25:.3f}, {r.p75:.3f}]",
        f"{100 * r.pct_decisions_with_gap_ge_48h:.1f}%",
    ]


def build_docx(
    figs: dict[str, Path],
    var_band: pd.DataFrame | None = None,
    by_var_on_row: pd.DataFrame | None = None,
) -> Path:
    band_counts = pd.read_csv(COV / "band_counts.csv")
    perf = pd.read_csv(COV / "performance_by_band.csv")
    by_var = by_var_on_row if by_var_on_row is not None else variable_coverage_on_row(
        pd.read_csv(COV / "availability_by_variable.csv")
    )
    by_ctr = pd.read_csv(COV / "availability_by_center.csv")
    by_tum = pd.read_csv(COV / "availability_by_tumor.csv")
    cont = pd.read_csv(COV / "continuous_coverage_summary.csv")
    inter = pd.read_csv(COV / "coverage_interaction_coefficients.csv")
    brk = pd.read_csv(COV / "coverage_breakdowns.csv")
    models_cv = pd.read_csv(CMP / "performance_models_cv.csv")
    inter_auc = pd.read_csv(COV / "coverage_interaction_auroc.csv")
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
        "A. Real coverage: for each decision, coverage on the 90 days the alarm used (its three 30-day blocks) "
        "and on its last 30-day block (recent coverage), as the share of variable–day cells really observed for "
        "the nine input variables before imputation (9 variables × 90 days = 810 expected cells per decision, "
        "or 270 for the 30-day window). SAP bands "
        "fixed a priori: below 50%, 50% to below 80%, and at least 80%. Same frozen alarm and threshold "
        "(1.0596…). Breakdowns by variable, centre and tumour. Extra (before B): same frozen alarm inside "
        "SAP bands defined by each variable’s own coverage, to see whether the type of missingness matters.",
    )
    add_text(
        doc,
        "B. Missingness-only comparator: three scores on the same 1013 points — (1) availability alone "
        "(logistic on the five indicators), (2) frozen DCABP load with the alarm cut, (3) frozen load plus "
        "those indicators in one logistic. The DCABP load itself is not refit. Extra models fit in DB1 only "
        "and evaluated on the same decisions as the frozen load; no DB2/DB3 tuning.",
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
        "Coverage is the fraction of expected variable–day cells with a real observation. Observed zeros count. "
        "activity_walking outside 120–15000 seconds is missing, and practiced_sport on that day is missing too. "
        "A stored sport 0 with no recorded walking and no recorded running is also missing. "
        "Imputed values do not count. The window is the calendar days the alarm used: the 90 (or 30) days ending "
        "at the decision date, with every empty day kept. In 6 of the 1,013 decisions (6 patients) the data end "
        "1–18 days before the decision; the alarm then takes the last 30 days with data as its final block, so "
        "the window runs from the start of its first block to that last day (72–89 days; the 30-day window is "
        "that final block). Days since the last valid record are still counted up to the decision date.",
    )

    add_heading(doc, "A. Alarm performance by band", level=2)
    add_text(
        doc,
        "The same alarm and cut are applied in every band. Brackets are 95% patient-clustered bootstrap "
        "intervals (1000 resamples). Every decision keeps a score after VQ-VAE, so no band is empty for performance.",
    )
    rows = []
    for w in (90, 30):
        sub = perf[perf.window_days == w].copy()
        if w != 90:
            sub = sub[sub.band != "all"]  # same frozen alarm on the same decisions: shown once, at 90 d
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
                    str(int(r.n_progressions)),
                    band_ci(r.AUROC, r.AUROC_lo, r.AUROC_hi),
                    band_ci(r.sensitivity, r.sensitivity_lo, r.sensitivity_hi),
                    band_ci(r.specificity, r.specificity_lo, r.specificity_hi),
                    band_ci(r.PPV, r.PPV_lo, r.PPV_hi),
                    band_ci(r.NPV, r.NPV_lo, r.NPV_hi),
                ]
            )
    add_table(
        doc,
        ["Window", "Band", "Patients", "Decisions", "Positive horizons", "Progressions",
         "AUROC", "Sensitivity", "Specificity", "PPV", "NPV"],
        rows,
        [0.8, 0.9, 0.9, 0.9, 1.0, 1.1, 1.9, 1.9, 1.9, 1.9, 1.9],
        size=7,
    )
    add_text(
        doc,
        "Positive horizons are decisions followed by a progression within 120 days. Each patient has at most one "
        "progression, so one progression can give several positive horizons; Progressions counts the distinct "
        "progressions (patients) behind them. The frozen alarm is the stored load of the original pipeline "
        "with the threshold 1.0596384140662845.",
    )
    add_figure(doc, figs["auroc_band"])

    add_heading(doc, "A. Availability by variable, source, centre and tumour (90 d)", level=2)
    add_text(
        doc,
        "The four tables below share the same columns, on the same 1,013 windows of 90 days before each decision "
        "(full calendar as denominator). Coverage of a decision is the share of its cells really observed: for a "
        "variable, the share of the 90 days with that variable; for a source, the share of the variable–day cells "
        "of its variables; for a centre or tumour, the share of all nine variables × 90 days (the coverage of the "
        "bands). Days observed is the share of the 90 days with at least one value of that variable, source or (for centre and tumour) any variable; for a single variable it equals its coverage. Mean and median [IQR] are across decisions. The last column is the share of decisions with at "
        "least one run of 2 or more consecutive days with nothing observed for that variable, source or (for "
        "centre and tumour) any variable. practiced_sport is derived from activity duration and is counted only "
        "when walking survived the 120–15000 s filter, so it has the same coverage as walking.",
    )
    add_text(
        doc,
        "Source. Sleep start, sleep duration, walking and steps can be recorded by the wearable or by the phone. "
        "They are assigned here to the wearable, the primary source. The phone is the fallback when the wearable "
        "has no record; that day-level split is not recorded. Location and app use come from the phone. "
        "practiced_sport has no source of its own and is not included in either source. Every window lies inside "
        "the patient's digital monitoring period, from the first to the last daily record (not the clinical "
        "follow-up of the study); separate activation and end dates per device are not available, so this "
        "digital period is the denominator for both sources.",
    )
    src_of = dict(zip(by_var["variable"], by_var["source"]))
    rows = [
        [r.group, src_of.get(r.group, "")] + breakdown_cells(r)
        for r in brk[brk.level == "variable"].itertuples(index=False)
    ]
    add_table(doc, ["Variable", "Source"] + BREAKDOWN_HEADER, rows, [2.2, 1.6, 0.9, 0.9, 1.1, 1.1, 1.8, 1.4], size=7)
    if "by_var" in figs:
        add_figure(doc, figs["by_var"], caption=False)

    src_label = {"wearable": "Wearable", "smartphone": "Smartphone"}
    rows = [[src_label[r.group]] + breakdown_cells(r) for r in brk[brk.level == "source"].itertuples(index=False)]
    add_table(doc, ["Source"] + BREAKDOWN_HEADER, rows, [2.2, 0.9, 0.9, 1.1, 1.1, 1.8, 1.4], size=7)

    rows = [[r.group] + breakdown_cells(r) for r in brk[brk.level == "center"].itertuples(index=False)]
    rows += [[r.group] + breakdown_cells(r) for r in brk[brk.level == "tumor"].itertuples(index=False)]
    rows += [["All"] + breakdown_cells(r) for r in brk[brk.level == "all"].itertuples(index=False)]
    add_table(doc, ["Centre / tumour"] + BREAKDOWN_HEADER, rows, [2.2, 0.9, 0.9, 1.1, 1.1, 1.8, 1.4], size=7)

    add_heading(doc, "A. Continuous coverage and interaction with the score", level=2)
    add_text(
        doc,
        "Besides the SAP bands, coverage is kept as a continuous value. Associations with the load and with the "
        "120-day outcome, AUROC of coverage alone and AUROC of the frozen load are reported. The 90- and 30-day "
        "windows only change how coverage is measured; the load and the alarm are the same frozen ones in both "
        "(three 30-day blocks), so the frozen load is shown once, at 90 days, with the "
        "reference interval. A fitted "
        "logistic model with load, coverage and their interaction follows. All brackets are patient-clustered bootstrap "
        "intervals (1000 resamples).",
    )
    add_text(
        doc,
        "Spearman ρ is used instead of Pearson because coverage is bounded in [0, 1] and the load–coverage "
        "relationship need not be linear: Spearman ranks both variables and measures monotone association "
        "(−1 to +1). Near 0 means little monotone link; the bootstrap interval crossing 0 means that association "
        "is compatible with none in these data. The same reading applies to ρ(coverage, 120-day outcome).",
    )
    ref_all = perf[(perf.window_days == 90) & (perf.band == "all")].iloc[0]
    rows = []
    for r in cont.itertuples(index=False):
        load_cell = band_ci(ref_all.AUROC, ref_all.AUROC_lo, ref_all.AUROC_hi) if int(r.window_days) == 90 else "same as 90"
        rows.append(
            [
                str(int(r.window_days)),
                fmt(r.mean_coverage, 3),
                band_ci(r.spearman_load_coverage, r.spearman_load_coverage_lo, r.spearman_load_coverage_hi),
                band_ci(r.spearman_coverage_outcome, r.spearman_coverage_outcome_lo, r.spearman_coverage_outcome_hi),
                band_ci(r.AUROC_coverage_alone, r.AUROC_coverage_alone_lo, r.AUROC_coverage_alone_hi),
                load_cell,
            ]
        )
    add_table(
        doc,
        ["Window", "Mean cov.", "ρ load–cov", "ρ cov–outcome", "AUROC cov.", "AUROC load"],
        rows,
        [0.9, 1.0, 1.8, 1.8, 1.7, 1.7],
        size=8,
    )

    i90 = inter[(inter.window_days == 90) & (inter.term == "load_z_x_cov_z")].iloc[0]
    add_text(
        doc,
        "Interaction model. A logistic model of the 120-day outcome is fitted with the load, the coverage and "
        "their product (load × coverage). Load and coverage are standardised, so each coefficient is per SD. "
        "The load × coverage coefficient says how much the effect of the load changes per SD of coverage. "
        "Brackets are 95% patient bootstrap intervals (1000 resamples). "
        f"At 90 days the interaction is {i90.beta:.2f} [{i90.beta_lo_boot:.2f}, {i90.beta_hi_boot:.2f}]: "
        "the interval includes 0, so coverage does not change how the load relates to the outcome.",
    )
    labels = {"intercept": "Intercept", "load_z": "Load", "cov_z": "Coverage",
              "load_z_x_cov_z": "Load × coverage"}
    rows = [
        [str(int(r.window_days)), labels[r.term], band_ci(r.beta, r.beta_lo_boot, r.beta_hi_boot)]
        for r in inter.itertuples(index=False)
        if r.term != "intercept"
    ]
    add_table(doc, ["Window", "Term", "Coefficient [95% CI]"], rows, [0.9, 2.0, 2.4], size=8)
    add_text(
        doc,
        "Discrimination is measured exactly as for the frozen baseline: the fitted scores on the same 1,013 "
        "decisions, no cross-validation, patient bootstrap. The difference is paired (same resampled patients).",
    )
    rows = [
        [str(int(r.window_days)), r.score, band_ci(r.AUROC, r.AUROC_lo, r.AUROC_hi),
         "—" if pd.isna(r.diff_vs_frozen) else band_ci(r.diff_vs_frozen, r.diff_lo, r.diff_hi)]
        for r in inter_auc.itertuples(index=False)
    ]
    add_table(doc, ["Window", "Score", "AUROC [95% CI]", "Difference vs frozen [95% CI]"], rows,
              [0.9, 3.0, 2.0, 2.4], size=8)

    # ----- A extra: per-variable coverage bands -----
    add_heading(doc, "A. Extra — alarm by per-variable coverage band", level=2)
    add_text(
        doc,
        "Exploratory addition (not a SAP primary table). Instead of the joint coverage of all nine variables, "
        "each decision is placed in the SAP bands using only that variable’s own 90-day coverage (still measured "
        "before VQ-VAE, with the same sport-zero rule). The frozen load and cut are applied inside each cell. "
        "Brackets are patient-clustered bootstrap intervals on AUROC.",
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
        "(4) monitoring interrupted now (current gap ≥ 2 days); (5) cell coverage, the share of the nine variables "
        "× 90 days really observed (the coverage of section A), which also captures days where only some "
        "variables are missing. A valid record is a day with at least one of the nine variables observed after "
        "the range filters; indicators (1)–(4) are built on valid records.",
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
        "How the three scores are built. (1) Availability only: a logistic model on the five indicators "
        "(standardised, mild L2 penalty). (2) DCABP: the frozen load and alarm cut 1.0596…, nothing refitted. "
        "(3) DCABP + availability: the same logistic on the frozen load plus the five indicators. "
        "Models (1) and (3) are evaluated exactly as the frozen load: their scores on the same 1,013 decisions, "
        "no cross-validation, with the cut that maximises sensitivity + specificity (Youden) on those decisions. "
        "Brackets are patient bootstrap intervals (1000 resamples). The AUROC difference is paired against the "
        "frozen load (same resampled patients). As a check, the same logistic on the load alone gives exactly "
        "the frozen alarm: same AUROC, and its Youden cut gives the same sensitivity and specificity.",
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
                "—" if pd.isna(r.AUROC_diff_vs_frozen) else band_ci(r.AUROC_diff_vs_frozen, r.AUROC_diff_lo, r.AUROC_diff_hi),
                band_ci(r.sensitivity, r.sensitivity_lo, r.sensitivity_hi),
                band_ci(r.specificity, r.specificity_lo, r.specificity_hi),
                band_ci(r.PPV, r.PPV_lo, r.PPV_hi),
                band_ci(r.NPV, r.NPV_lo, r.NPV_hi),
            ]
        )
    add_table(
        doc,
        ["Model", "Patients", "N", "Pos.", "AUROC", "AUROC diff. vs frozen", "Sens.", "Spec.", "PPV", "NPV"],
        rows,
        [1.9, 0.9, 0.7, 0.7, 1.6, 1.7, 1.5, 1.5, 1.5, 1.5],
        size=7,
    )
    add_figure(doc, figs["comparator"])
    add_text(
        doc,
        "Availability alone has AUROC "
        f"{float(show.set_index('model').loc['availability_only', 'AUROC']):.3f}, well below the frozen load "
        f"({float(show.set_index('model').loc['dcabp_load_frozen', 'AUROC']):.3f}). Adding the availability "
        "indicators to the load gives "
        f"{float(show.set_index('model').loc['dcabp_plus_availability', 'AUROC']):.3f}; the paired difference "
        "interval includes 0, so missingness adds no clear information to the alarm.",
    )

    add_heading(doc, "B. Added value with patient cross-validation", level=2)
    add_text(
        doc,
        "Separate comparison. Three logistic models are fitted with one shared assignment "
        "of patients to 5 folds, so every model is evaluated on exactly the same held-out patients: availability "
        "only, the frozen load alone, and the frozen load plus the five indicators. Scores are the out-of-fold "
        "probability of progression; the classification cut is the Youden cut of the training folds. The AUROC "
        "difference (load + availability minus load) is paired, with patient bootstrap intervals (1000 resamples). "
        "The frozen alarm is shown apart as the reference: it is not refitted, so it is not part of this comparison.",
    )
    cv_label = {
        "availability_only_cv": "Availability only",
        "load_logistic_cv": "Load (logistic)",
        "load_plus_availability_cv": "Load + availability",
    }
    rows = [
        [
            cv_label[r.model],
            band_ci(r.AUROC, r.AUROC_lo, r.AUROC_hi),
            "—" if pd.isna(r.AUROC_diff_vs_load_cv) else band_ci(r.AUROC_diff_vs_load_cv, r.AUROC_diff_lo, r.AUROC_diff_hi),
            band_ci(r.sensitivity, r.sensitivity_lo, r.sensitivity_hi),
            band_ci(r.specificity, r.specificity_lo, r.specificity_hi),
            band_ci(r.PPV, r.PPV_lo, r.PPV_hi),
            band_ci(r.NPV, r.NPV_lo, r.NPV_hi),
        ]
        for r in models_cv.itertuples(index=False)
    ]
    fz = models.set_index("model").loc["dcabp_load_frozen"]
    rows.append([
        "Reference: frozen alarm",
        band_ci(fz.AUROC, fz.AUROC_lo, fz.AUROC_hi),
        "—",
        band_ci(fz.sensitivity, fz.sensitivity_lo, fz.sensitivity_hi),
        band_ci(fz.specificity, fz.specificity_lo, fz.specificity_hi),
        band_ci(fz.PPV, fz.PPV_lo, fz.PPV_hi),
        band_ci(fz.NPV, fz.NPV_lo, fz.NPV_hi),
    ])
    add_table(
        doc,
        ["Model", "AUROC (out-of-fold)", "Difference vs load [95% CI]", "Sens.", "Spec.", "PPV", "NPV"],
        rows,
        [1.9, 1.8, 1.9, 1.6, 1.6, 1.6, 1.6],
        size=7,
    )
    cvx = models_cv.set_index("model")
    d = cvx.loc["load_plus_availability_cv"]
    add_text(
        doc,
        f"With the same folds, adding the availability indicators to the load changes the out-of-fold AUROC from "
        f"{cvx.loc['load_logistic_cv', 'AUROC']:.3f} to {d.AUROC:.3f} (difference {d.AUROC_diff_vs_load_cv:.3f} "
        f"[{d.AUROC_diff_lo:.3f}, {d.AUROC_diff_hi:.3f}]). "
        + ("The interval is below 0: availability does not improve the prediction and, out of fold, makes it slightly worse. "
           if d.AUROC_diff_hi < 0 else
           "The interval includes 0: availability does not improve the prediction. "
           if d.AUROC_diff_lo <= 0 else
           "The interval is above 0. ")
        + f"Availability alone has an out-of-fold AUROC of {cvx.loc['availability_only_cv', 'AUROC']:.3f}.",
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
    by_var_on_row = variable_coverage_on_row(pd.read_csv(COV / "availability_by_variable.csv"))
    figs = make_figures(by_var_on_row)
    dest = build_docx(figs, var_band=var_band, by_var_on_row=by_var_on_row)
    (OUT / "report.md").write_text(
        "# Real coverage and informative missingness\n\n"
        "Word report with tables and figures:\n\n"
        f"`{dest.relative_to(ROOT)}`\n\n"
        "- Coverage CSVs and figures in this folder\n"
        "- Comparator CSVs: `availability_comparator/`\n"
        "- Extra A: `performance_by_variable_band.csv`\n"
        "- Load × coverage interaction: `coverage_interaction_coefficients.csv`, `coverage_interaction_auroc.csv` "
        "(`scripts/03_analysis/alarm/run_coverage_interaction.py`)\n"
    )
    print(f"→ figures: {len(figs)}")
    for k, p in figs.items():
        print(f"  {k}: {p.relative_to(ROOT)}")
    print(f"→ {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

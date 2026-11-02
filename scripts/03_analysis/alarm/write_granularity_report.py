#!/usr/bin/env python3
"""Write alarm_granularity_report.{md,html,pdf} in the **main-branch** layout.

Same section structure / wording as ``main:results/alarm/alarm_granularity_report.md``,
filled with the Evento_DB1 clock run (not Subjects Obs_time).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from clock_schema import default_clock_path, load_clock, project_root_from

GRAN_LABEL = {
    "daily": "daily (sampled)",
    "weekly": "weekly sampled",
    "biweekly": "biweekly sampled",
    "monthly_sample": "monthly sampled",
    "weekly_collapsed": "weekly collapsed",
    "biweekly_collapsed": "biweekly collapsed",
    "monthly_collapsed": "monthly collapsed (30 d)",
}


def _fmt3(x: float) -> str:
    return f"{float(x):.3f}"


def _ci(lo: float, hi: float) -> str:
    return f"{float(lo):.3f}–{float(hi):.3f}"


def _wh_triplet(sweep: pd.DataFrame, gran: str, h: int) -> str:
    parts = []
    for w in (60, 90, 120):
        r = sweep[
            (sweep.granularity == gran)
            & (sweep.w_load_days == w)
            & (sweep.h_horizon_days == h)
        ]
        parts.append(_fmt3(r.roc_auc.iloc[0]) if len(r) else "—")
    return " / ".join(parts)


def write_report(root: Path, run_id: str = "Evento_DB1_clock") -> Path:
    alarm = root / "results/alarm"
    run = alarm / "runs" / run_id
    paper = pd.read_csv(run / "paper_W3_H4" / "alarm_metrics_W3_H4.csv").iloc[0]
    sweep = pd.read_csv(run / "granularity" / "granularity_sweep_summary.csv")
    g90 = pd.read_csv(alarm / "analysis_figures" / "metrics_table_W90_H120.csv")
    ci = pd.read_csv(alarm / "analysis_figures" / "auc_ci_W90_H120.csv")
    g90 = g90.merge(ci[["granularity", "ci_low", "ci_high"]], on="granularity", how="left")

    clock = load_clock(default_clock_path(root))
    ex = clock.set_index("id")

    n_lm = int(paper.n_landmarks)
    n_pos = int(paper.n_positive_targets)
    n_pat = int(paper.n_patients)
    auroc = float(paper.AUROC)
    paper_theta = float(paper.paper_theta)
    youden_theta = float(paper.youden_theta)

    # §5 rows: Youden per construction; monthly_collapsed uses manuscript θ (like main §4/§5 align)
    rows_md = []
    for _, r in g90.iterrows():
        gran = str(r["granularity"])
        n = int(r["n_decision_points"])
        auc_v = float(r["roc_auc"])
        sens = float(r["sensitivity"])
        spec = float(r["specificity"])
        ppv = float(r["PPV"])
        npv = float(r["NPV"])
        lo, hi = float(r["ci_low"]), float(r["ci_high"])
        if gran == "monthly_collapsed":
            n = n_lm
            auc_v = auroc
            sens = float(paper.paper_sens)
            spec = float(paper.paper_spec)
            ppv = float(paper.paper_ppv)
            npv = float(paper.paper_npv)
            bold_n = f"**{n}**"
            bold_auc = f"**{_fmt3(auc_v)}**"
        else:
            bold_n = str(n)
            bold_auc = _fmt3(auc_v)
        rows_md.append(
            f"| {GRAN_LABEL.get(gran, gran)} | {bold_n} | {bold_auc} | {_ci(lo, hi)} | "
            f"{_fmt3(sens)} | {_fmt3(spec)} | {_fmt3(ppv)} | {_fmt3(npv)} |"
        )

    sampled = g90[g90["granularity"].isin(["daily", "weekly", "biweekly", "monthly_sample"])]
    collapsed = g90[g90["granularity"].str.endswith("collapsed")]
    auc_s = float(sampled["roc_auc"].mean())
    auc_c = float(collapsed["roc_auc"].mean())
    best = g90.loc[g90["roc_auc"].idxmax()]
    ppv_lo = float(g90["PPV"].min())
    ppv_hi = float(g90["PPV"].max())

    # Patient notes vs operating θ (same as baseline metrics)
    traj_theta = float(paper_theta)
    notes = {}
    for pid in (62004, 23003, 31002, 41006):
        ev = int(ex.loc[pid, "evento"])
        te = int(ex.loc[pid, "t_evento"])
        profile = f"PD (day {te})" if ev == 1 else "No PD"
        ser_path = alarm / "tables" / f"trajectory_series_{pid}_W3.csv"
        note = "see trajectory"
        if ser_path.exists():
            s = pd.read_csv(ser_path)
            if "AUC" in s.columns:
                hit = s[s["AUC"] >= traj_theta]
                if len(hit):
                    note = f"1st alarm ~month {int(hit['month'].iloc[0])}"
                    if ev == 1:
                        note += " (lead before PD)" if int(hit["month"].iloc[0]) * 30 < te else ""
                    elif ev == 0:
                        note += " with no event in follow-up"
                elif ev == 1:
                    note = "peaks near end; no sustained cross of θ"
                else:
                    note = "Below threshold; no alarm"
        notes[pid] = (profile, note)

    today = date.today().isoformat()
    md = f"""# Rolling-burden alarm and temporal granularity

## 1. What the alarm is

The alarm is a **prospective rolling-burden** rule on behavioural topic probabilities from LDA / DCABP.

At each decision time \(t\):

1. Build **pUF** = P(topic 3) + P(topic 4) + P(topic 5) (unfavourable day-types).
2. Aggregate pUF over a **lookback window W** into a **burden** score.
3. Raise an alarm if burden ≥ threshold \(\\theta\).
4. Label the decision as positive if progression (PD) occurs within a future **horizon H** after \(t\).

Only decisions with enough history (full W) and enough follow-up to score the label (or a known PD time) are kept. Those eligible \((patient, t)\) pairs are the **decision points**.

---

## 2. Parameters that can be varied

| Parameter | Role | Typical choices |
|-----------|------|-----------------|
| **W (lookback)** | How much past burden is summarised at \(t\) | 60 / 90 / 120 days (≈ 2 / 3 / 4 months) |
| **H (horizon)** | How far ahead PD counts as a hit | 90 / 120 / 180 days (≈ 3 / 4 / 6 months) |
| **Granularity** | How often decisions are made and how LDA is built | daily, weekly, biweekly, monthly |
| **Score family** | How pUF / burden are constructed | **sampled** vs **collapsed** (below) |
| **θ (threshold)** | Alarm cutoff on the burden scale | Fixed operating point (θ = 1.60 on monthly *sum* scale), or Youden for other constructions |
| **Event / censor time** | Clock used to stop landmarks and label PD | user clock BBDD (`fecha_fin − fecha_inicio`) |

### Sampled vs collapsed

| Family | Idea |
|--------|------|
| **Sampled** | Daily sliding LDA (30-day window) → daily pUF; weekly / biweekly / monthly only change *when* you decide (same underlying daily signal). Burden = rolling mean over the last **W days**. |
| **Collapsed** | One LDA per non-overlapping block (7 / 14 / 30 d) → block pUF; burden = rolling mean over the last `round(W/block)` blocks. |

Collapsed block pUF is **not** the mean of daily pUF inside the block: each family is its own LDA construction.

### Monthly collapsed landmarks

On the monthly scale, month \(m\) is placed at day \(t = m \\times 30\). A landmark is kept only if \(t\) is strictly before the event/censor time. Burden is the **sum** of pUF over the last W months (so thresholds are on a sum scale, θ = 1.60, not the mean scale ~0.53 used elsewhere).

---

## 3. Experiment in this report

We fix a main operating region and then sweep the knobs above:

1. **Monthly collapsed baseline** — W = 3 months, H = 4 months, user clock BBDD (`t_evento = fecha_fin − fecha_inicio`) as event/censor clock, fixed θ = 1.60.  
2. **Granularity comparison** — W = 90 d, H = 120 d; sampled (daily→monthly) vs collapsed (weekly / biweekly / monthly).  
3. **W × H grid** — W ∈ {{60, 90, 120}} d × H ∈ {{90, 120, 180}} d.  
4. **Individual trajectories** — example patients with and without PD.

Outputs live under `results/alarm/` (`tables/`, `figures/`, `analysis_figures/`, `runs/{run_id}/`). Primary monthly landmarks: `runs/{run_id}/paper_W3_H4/decision_points_W3_H4.csv` (**clock BBDD**, n={n_lm}).

**Delivery summary figures** (high-res, clock-aligned, AUROC with patient-bootstrap CIs): [`analysis_figures/`](analysis_figures/).

---

## 4. Results — monthly collapsed baseline (W = 3 mo, H = 4 mo)

Score = **sum** of pUF over W months; cutoff: `month × 30 < t_evento` (clock BBDD).  
n = **{n_lm}** decision points · **{n_pat}** patients · **{n_pos}** positives.

| Metric | Value |
|--------|-------|
| ROC-AUC | **{_fmt3(auroc)}** |
| Sens / Spec (θ≈{paper_theta:.2f}, sum scale) | {_fmt3(paper.paper_sens)} / {_fmt3(paper.paper_spec)} |
| PPV / NPV | {_fmt3(paper.paper_ppv)} / {_fmt3(paper.paper_npv)} |
| TP / FN / FP / TN | {int(paper.paper_TP)} / {int(paper.paper_FN)} / {int(paper.paper_FP)} / {int(paper.paper_TN)} |

**Threshold scale (same rule, two writings):**

| Scale | θ | Relation |
|-------|---:|----------|
| Sum (as used here) | **{paper_theta:.2f}** | `sum` of last W=3 monthly pUF |
| Mean (equivalent) | **≈ {_fmt3(paper_theta / 3)}** | `{paper_theta:.2f} / 3` |

Operating point used throughout (baseline metrics + trajectories): **θ = {paper_theta:.2f}** (sum scale).  
AUROC describes the whole ROC and does **not** pick a threshold. A pure Youden cut on this clock (~1.06) over-alarms (sens≈0.88 / spec≈0.45) and is **not** used. Final clinical θ to be confirmed.

![ROC monthly collapsed W=3 H=4](figures/roc_W3_H4.png)

---

## 5. Results — granularity (W = 90 d, H = 120 d)

Monthly collapsed row: manuscript θ={paper_theta:.2f} (same as §4). Other rows: construction-wise Youden on their own score scale (for a single comparable operating point per row — not the clinical cut-off of the paper).

**Clinical clock:** all rows use **`clock_bbdd.xlsx`** (`t_evento = fecha_fin − fecha_inicio`).  
**AUROC uncertainty:** 95% CIs from **patient-clustered bootstrap** (2000 reps). See `analysis_figures/auc_ci_W90_H120.csv`.

| Granularity | n | AUC | 95% CI | Sens | Spec | PPV | NPV |
|-------------|--:|----:|--------|-----:|-----:|----:|----:|
{chr(10).join(rows_md)}

**Short read**

- **Sampled** (daily→monthly): AUC nearly flat (~{_fmt3(auc_s)}). Subsampling does not improve discrimination; it only reduces n.
- **Collapsed** improves somewhat vs sampled (~{_fmt3(auc_c)}). Highest point estimate here: **{GRAN_LABEL.get(str(best.granularity), best.granularity)} (AUC {_fmt3(best.roc_auc)})**; CIs overlap across collapsed configs.
- PPV remains low (~{_fmt3(ppv_lo)}–{_fmt3(ppv_hi)}); NPV high.
High-resolution delivery panel (metrics table + AUROC points with 95% CI + ROC; no winner star):

![Alarm performance across temporal granularities](analysis_figures/summary_granularities_W90_H120.png)

---

## 6. Results — W × H grid

Grid: **W ∈ {{60, 90, 120}} d** × **H ∈ {{90, 120, 180}} d**.

### AUC (selected)

| | H=90 | H=120 | H=180 |
|--|-----:|------:|------:|
| **Daily** W=60/90/120 | {_wh_triplet(sweep, 'daily', 90)} | {_wh_triplet(sweep, 'daily', 120)} | {_wh_triplet(sweep, 'daily', 180)} |
| **Weekly collapsed** | {_wh_triplet(sweep, 'weekly_collapsed', 90)} | {_wh_triplet(sweep, 'weekly_collapsed', 120)} | {_wh_triplet(sweep, 'weekly_collapsed', 180)} |
| **Monthly collapsed** | {_wh_triplet(sweep, 'monthly_collapsed', 90)} | {_wh_triplet(sweep, 'monthly_collapsed', 120)} | {_wh_triplet(sweep, 'monthly_collapsed', 180)} |

**Read**

- **↑H** → AUC rises monotonically.
- **↑W** → AUC rises moderately (more historical load).
- Ranking **collapsed ≳ sampled** holds across almost the whole grid.
- W≈90 / H≈120 sits in the mid region of the map.

![AUC heatmaps W×H](figures/wh_auc_heatmaps.png)

![AUC vs horizon H](figures/wh_auc_vs_horizon.png)

Table: `runs/{run_id}/granularity/granularity_sweep_summary.csv`

---

## 7. Results — individual trajectories

Patients: **62004**, **23003**, **31002**, **41006**.

| ID | Profile | Notes |
|----|---------|-------|
| 62004 | {notes[62004][0]} | {notes[62004][1]} |
| 23003 | {notes[23003][0]} | {notes[23003][1]} |
| 31002 | {notes[31002][0]} | {notes[31002][1]} |
| 41006 | {notes[41006][0]} | {notes[41006][1]} |

### 7.1 Monthly collapsed (pUF + rolling burden)

W=3 months, H=4 months, θ={paper_theta:.2f} (sum scale; mean-scale θ/W≈{_fmt3(paper_theta / 3)}).

![Monthly panel](figures/trajectories_panel_paper_W3_H4.png)

### 7.2 Sampled vs collapsed (W = 90 d)

Simplified high-res delivery plots (daily/sampled burden vs monthly collapsed; clock BBDD; manuscript-aligned θ labelled):

![62004 simplified](analysis_figures/trajectory_patient_62004_W90_simplified.png)

![41006 simplified](analysis_figures/trajectory_patient_41006_W90_simplified.png)

Full multi-panel trajectories:

![Granularity panel](figures/trajectories_panel_W90.png)

- **Blues:** daily sliding pUF + daily/sampled burden  
- **Reds:** monthly collapsed pUF + monthly collapsed burden  
- Steps: weekly / biweekly collapsed burden  
- Markers: sampled evaluation (7 / 14 / 30 d) on the daily burden  

![62004 W90](figures/trajectory_patient_62004_W90.png)

![23003 W90](figures/trajectory_patient_23003_W90.png)

![31002 W90](figures/trajectory_patient_31002_W90.png)

![41006 W90](figures/trajectory_patient_41006_W90.png)

---

## 8. How to reproduce

```bash
# Paper baseline + granularity (user clock BBDD)
python scripts/03_analysis/alarm/run_alarm_pipeline.py \\
  --clock data/revision/clock_bbdd.xlsx --run-id {run_id} \\
  --with-granularity --skip-lda

# Monthly / granularity trajectories (θ={paper_theta:.2f})
python scripts/03_analysis/alarm/plot_patient_trajectories.py \\
  --patients 62004,23003,31002,41006 --clock data/revision/clock_bbdd.xlsx
python scripts/03_analysis/alarm/plot_granularity_trajectories.py \\
  --patients 62004,23003,31002,41006 --W 90 --clock data/revision/clock_bbdd.xlsx

# High-res delivery figures (clock-aligned; AUROC patient-bootstrap CIs)
python scripts/03_analysis/alarm/plot_analysis_figures.py \\
  --clock data/revision/clock_bbdd.xlsx --n-boot 2000

# This report (main-branch markdown layout)
python scripts/03_analysis/alarm/write_granularity_report.py --run-id {run_id}
```

File map: [`README.md`](README.md). Delivery panel: [`analysis_figures/`](analysis_figures/).

---

## 9. Conclusions

1. **Subsampling** the same daily signal (sampled) **does not improve** discrimination: AUC stays ~**{_fmt3(auc_s)}** from daily→monthly.  
2. **Collapsing** LDA into non-overlapping blocks (7/14/30 d) **does improve** somewhat: AUC ~**{_fmt3(auc_c)}** at W=90 d, H=120 d.  
3. The **highest point estimate** in that setting is **{GRAN_LABEL.get(str(best.granularity), best.granularity)}** (AUC **{_fmt3(best.roc_auc)}**); patient-bootstrap 95% CIs overlap across collapsed configs.  
4. **↑W** and **↑H** raise AUC in an ordered way; ranking collapsed ≳ sampled holds on the grid.  
5. **PPV** stays low (~**{_fmt3(g90.PPV.mean())}**): the alarm discriminates but yields many false positives at useful thresholds.  
6. Individually, the score can anticipate PD (lead time), mark persistently high burden, or fire late alarms with no event — useful for interpretation beyond aggregate metrics.  
7. The delivery summary figure uses the **same clinical follow-up field (clock BBDD)** for all granularities as the monthly baseline (n={n_lm}), with **AUROC 95% CIs from patient-level bootstrap**.

---

*Generated {today} · run `{run_id}` · clock `data/revision/clock_bbdd.xlsx`*
"""

    out_md = alarm / "alarm_granularity_report.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_md}")

    # HTML from markdown (same path as main README)
    try:
        import markdown as md_lib
    except ImportError:
        subprocess.check_call(
            [__import__("sys").executable, "-m", "pip", "install", "markdown", "-q"]
        )
        import markdown as md_lib

    body = md_lib.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    css = """
    body { font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
           color: #1a1a1a; max-width: 920px; margin: 24px auto; padding: 0 20px 48px; }
    h1 { font-size: 26px; margin-bottom: 8px; }
    h2 { font-size: 18px; margin-top: 28px; border-top: 1px solid #ddd; padding-top: 16px; }
    h3 { font-size: 15px; margin-top: 18px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0; }
    th, td { border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }
    th { color: #555; font-weight: 600; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; margin: 8px 0; }
    code { font-size: 12px; background: #f5f5f5; padding: 1px 4px; }
    pre { background: #f5f5f5; padding: 12px; overflow-x: auto; font-size: 12px; }
    """
    html = (
        "<!DOCTYPE html><html><head><meta charset=utf-8>"
        f"<title>Alarm report</title><style>{css}</style></head><body>"
        f"{body}<p><em>Generated {today}</em></p></body></html>"
    )
    out_html = alarm / "alarm_granularity_report.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {out_html}")

    # PDF via Chrome/Chromium headless if available
    out_pdf = alarm / "alarm_granularity_report.pdf"
    chrome_candidates = [
        "google-chrome",
        "chromium",
        "chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    chrome = next((c for c in chrome_candidates if shutil.which(c) or Path(c).exists()), None)
    if chrome:
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}",
                f"file://{out_html.resolve()}",
            ],
            check=False,
            capture_output=True,
        )
        if out_pdf.exists():
            print(f"Wrote {out_pdf}")
        else:
            print("WARNING: Chrome PDF export failed; HTML/MD written.")
    else:
        # fallback: keep previous weasy/pdfkit path if write used before
        try:
            from weasyprint import HTML as WeasyHTML  # type: ignore

            WeasyHTML(filename=str(out_html)).write_pdf(str(out_pdf))
            print(f"Wrote {out_pdf} (weasyprint)")
        except Exception as e:
            print(f"WARNING: no Chrome/weasyprint for PDF ({e}). Open the HTML instead.")

    return out_md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--run-id", type=str, default="Evento_DB1_clock")
    args = p.parse_args()
    write_report(project_root_from(args.root), run_id=args.run_id)


if __name__ == "__main__":
    main()

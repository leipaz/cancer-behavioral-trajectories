# Rolling-burden alarm and temporal granularity

## 1. What the alarm is

The alarm is a **prospective rolling-burden** rule on behavioural topic probabilities from LDA / DCABP.

At each decision time \(t\):

1. Build **pUF** = P(topic 3) + P(topic 4) + P(topic 5) (unfavourable day-types).
2. Aggregate pUF over a **lookback window W** into a **burden** score.
3. Raise an alarm if burden ≥ threshold \(\theta\).
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

On the monthly scale, month \(m\) is placed at day \(t = m \times 30\). A landmark is kept only if \(t\) is strictly before the event/censor time. Burden is the **sum** of pUF over the last W months (so thresholds are on a sum scale, θ = 1.60, not the mean scale ~0.53 used elsewhere).

---

## 3. Experiment in this report

We fix a main operating region and then sweep the knobs above:

1. **Monthly collapsed baseline** — W = 3 months, H = 4 months, user clock BBDD (`t_evento = fecha_fin − fecha_inicio`) as event/censor clock, fixed θ = 1.60.  
2. **Granularity comparison** — W = 90 d, H = 120 d; sampled (daily→monthly) vs collapsed (weekly / biweekly / monthly).  
3. **W × H grid** — W ∈ {60, 90, 120} d × H ∈ {90, 120, 180} d.  
4. **Individual trajectories** — example patients with and without PD.

Outputs live under `results/alarm/` (`tables/`, `figures/`, `analysis_figures/`, `runs/Evento_DB1_clock/`). Primary monthly landmarks: `runs/Evento_DB1_clock/paper_W3_H4/decision_points_W3_H4.csv` (**clock BBDD**, n=1016).

**Delivery summary figures** (high-res, clock-aligned, AUROC with patient-bootstrap CIs): [`analysis_figures/`](analysis_figures/).

---

## 4. Results — monthly collapsed baseline (W = 3 mo, H = 4 mo)

Score = **sum** of pUF over W months; cutoff: `month × 30 < t_evento` (clock BBDD).  
n = **1016** decision points · **116** patients · **156** positives.

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.700** |
| Sens / Spec (θ≈1.60, sum scale) | 0.673 / 0.616 |
| PPV / NPV | 0.241 / 0.912 |
| TP / FN / FP / TN | 105 / 51 / 330 / 530 |

**Threshold scale (same rule, two writings):**

| Scale | θ | Relation |
|-------|---:|----------|
| Sum (as used here) | **1.60** | `sum` of last W=3 monthly pUF |
| Mean (equivalent) | **≈ 0.533** | `1.60 / 3` |

Operating point used throughout (baseline metrics + trajectories): **θ = 1.60** (sum scale).  
AUROC describes the whole ROC and does **not** pick a threshold. A pure Youden cut on this clock (~1.06) over-alarms (sens≈0.88 / spec≈0.45) and is **not** used. Final clinical θ to be confirmed.

![ROC monthly collapsed W=3 H=4](figures/roc_W3_H4.png)

---

## 5. Results — granularity (W = 90 d, H = 120 d)

Monthly collapsed row: manuscript θ=1.60 (same as §4). Other rows: construction-wise Youden on their own score scale (for a single comparable operating point per row — not the clinical cut-off of the paper).

**Clinical clock:** all rows use **`clock_bbdd.xlsx`** (`t_evento = fecha_fin − fecha_inicio`).  
**AUROC uncertainty:** 95% CIs from **patient-clustered bootstrap** (2000 reps). See `analysis_figures/auc_ci_W90_H120.csv`.

| Granularity | n | AUC | 95% CI | Sens | Spec | PPV | NPV |
|-------------|--:|----:|--------|-----:|-----:|----:|----:|
| daily (sampled) | 36495 | 0.683 | 0.610–0.749 | 0.845 | 0.437 | 0.207 | 0.942 |
| weekly sampled | 5262 | 0.682 | 0.613–0.753 | 0.845 | 0.437 | 0.206 | 0.942 |
| biweekly sampled | 2664 | 0.678 | 0.608–0.749 | 0.824 | 0.456 | 0.205 | 0.939 |
| monthly sampled | 1283 | 0.679 | 0.612–0.749 | 0.816 | 0.475 | 0.208 | 0.939 |
| weekly collapsed | 4137 | 0.700 | 0.620–0.773 | 0.734 | 0.590 | 0.242 | 0.925 |
| biweekly collapsed | 2130 | 0.697 | 0.620–0.769 | 0.781 | 0.537 | 0.245 | 0.928 |
| monthly collapsed (30 d) | **1016** | **0.700** | 0.623–0.770 | 0.673 | 0.616 | 0.241 | 0.912 |

**Short read**

- **Sampled** (daily→monthly): AUC nearly flat (~0.680). Subsampling does not improve discrimination; it only reduces n.
- **Collapsed** improves somewhat vs sampled (~0.699). Highest point estimate here: **weekly collapsed (AUC 0.700)**; CIs overlap across collapsed configs.
- PPV remains low (~0.205–0.245); NPV high.
High-resolution delivery panel (metrics table + AUROC points with 95% CI + ROC; no winner star):

![Alarm performance across temporal granularities](analysis_figures/summary_granularities_W90_H120.png)

---

## 6. Results — W × H grid

Grid: **W ∈ {60, 90, 120} d** × **H ∈ {90, 120, 180} d**.

### AUC (selected)

| | H=90 | H=120 | H=180 |
|--|-----:|------:|------:|
| **Daily** W=60/90/120 | 0.664 / 0.672 / 0.680 | 0.673 / 0.683 / 0.691 | 0.704 / 0.711 / 0.715 |
| **Weekly collapsed** | 0.676 / 0.693 / 0.702 | 0.685 / 0.700 / 0.711 | 0.711 / 0.725 / 0.732 |
| **Monthly collapsed** | 0.673 / 0.692 / 0.699 | 0.680 / 0.700 / 0.710 | 0.706 / 0.728 / 0.736 |

**Read**

- **↑H** → AUC rises monotonically.
- **↑W** → AUC rises moderately (more historical load).
- Ranking **collapsed ≳ sampled** holds across almost the whole grid.
- W≈90 / H≈120 sits in the mid region of the map.

![AUC heatmaps W×H](figures/wh_auc_heatmaps.png)

![AUC vs horizon H](figures/wh_auc_vs_horizon.png)

Table: `runs/Evento_DB1_clock/granularity/granularity_sweep_summary.csv`

---

## 7. Results — individual trajectories

Patients: **62004**, **23003**, **31002**, **41006**.

| ID | Profile | Notes |
|----|---------|-------|
| 62004 | PD (day 317) | 1st alarm ~month 9 (lead before PD) |
| 23003 | PD (day 209) | 1st alarm ~month 3 (lead before PD) |
| 31002 | No PD | 1st alarm ~month 21 with no event in follow-up |
| 41006 | No PD | Below threshold; no alarm |

### 7.1 Monthly collapsed (pUF + rolling burden)

W=3 months, H=4 months, θ=1.60 (sum scale; mean-scale θ/W≈0.533).

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
python scripts/03_analysis/alarm/run_alarm_pipeline.py \
  --clock data/revision/clock_bbdd.xlsx --run-id Evento_DB1_clock \
  --with-granularity --skip-lda

# Monthly / granularity trajectories (θ=1.60)
python scripts/03_analysis/alarm/plot_patient_trajectories.py \
  --patients 62004,23003,31002,41006 --clock data/revision/clock_bbdd.xlsx
python scripts/03_analysis/alarm/plot_granularity_trajectories.py \
  --patients 62004,23003,31002,41006 --W 90 --clock data/revision/clock_bbdd.xlsx

# High-res delivery figures (clock-aligned; AUROC patient-bootstrap CIs)
python scripts/03_analysis/alarm/plot_analysis_figures.py \
  --clock data/revision/clock_bbdd.xlsx --n-boot 2000

# This report (main-branch markdown layout)
python scripts/03_analysis/alarm/write_granularity_report.py --run-id Evento_DB1_clock
```

File map: [`README.md`](README.md). Delivery panel: [`analysis_figures/`](analysis_figures/).

---

## 9. Conclusions

1. **Subsampling** the same daily signal (sampled) **does not improve** discrimination: AUC stays ~**0.680** from daily→monthly.  
2. **Collapsing** LDA into non-overlapping blocks (7/14/30 d) **does improve** somewhat: AUC ~**0.699** at W=90 d, H=120 d.  
3. The **highest point estimate** in that setting is **weekly collapsed** (AUC **0.700**); patient-bootstrap 95% CIs overlap across collapsed configs.  
4. **↑W** and **↑H** raise AUC in an ordered way; ranking collapsed ≳ sampled holds on the grid.  
5. **PPV** stays low (~**0.222**): the alarm discriminates but yields many false positives at useful thresholds.  
6. Individually, the score can anticipate PD (lead time), mark persistently high burden, or fire late alarms with no event — useful for interpretation beyond aggregate metrics.  
7. The delivery summary figure uses the **same clinical follow-up field (clock BBDD)** for all granularities as the monthly baseline (n=1016), with **AUROC 95% CIs from patient-level bootstrap**.

---

*Generated 2026-10-21 · run `Evento_DB1_clock` · clock `data/revision/clock_bbdd.xlsx`*

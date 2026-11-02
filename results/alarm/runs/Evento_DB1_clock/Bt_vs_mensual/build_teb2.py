"""Repeat the DB1 granularity alarm with progression time from the first daily day.

t_evento_eb2 = (fecha_fin - fecha_inicio_daily).days
This matches Tiempo_obs minus the initial-start gap only.
The event's study day is t_evento_eb2 + 1. The horizon uses that study day:
event inside if t_evento_eb2 + 1 <= T + 120.

Patients whose two start dates match are unchanged. The original report is not overwritten.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.oxml.ns import qn

ROOT = Path("/export/gts_usuarios/lparbaiza/cancer-behavioral-trajectories")
BT = ROOT / "results/alarm/runs/Evento_DB1_clock/Bt_vs_mensual"
N_BOOT, SEED, H = 2000, 20261020, 120
STEPS = [("Daily", 1), ("Weekly", 7), ("Every 14 days", 14), ("Every 30 days", 30)]


def youden(y, x):
    best_j, best_thr = -1.0, None
    best = None
    for thr in np.unique(x):
        pred = x >= thr
        tp = int(((pred) & (y == 1)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        tn = int(((~pred) & (y == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thr = float(thr)
            best = dict(threshold=best_thr, J=float(j), TP=tp, FN=fn, FP=fp, TN=tn,
                        sensitivity=float(sens), specificity=float(spec),
                        PPV=float(tp / (tp + fp)) if tp + fp else np.nan,
                        NPV=float(tn / (tn + fn)) if tn + fn else np.nan)
    return best


def auroc(y, x, w=None):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pos, neg = w * y, w * (1 - y)
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    _, ix = np.unique(x, return_inverse=True)
    pg = np.bincount(ix, weights=pos)
    ng = np.bincount(ix, weights=neg)
    return float(np.sum(pg * (np.cumsum(ng) - ng / 2)) / (pos.sum() * neg.sum()))


def clasif(y, x, thr, w=None):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pred = x >= thr
    pos, neg = w * y, w * (1 - y)
    tp, fn = float(pos[pred].sum()), float(pos[~pred].sum())
    fp, tn = float(neg[pred].sum()), float(neg[~pred].sum())

    def ratio(a, b):
        return a / b if b else np.nan

    return dict(AUROC=auroc(y, x, w), sensitivity=ratio(tp, tp + fn), specificity=ratio(tn, tn + fp),
                PPV=ratio(tp, tp + fp), NPV=ratio(tn, tn + fn),
                TP=tp, FN=fn, FP=fp, TN=tn)


def label(day, tev, event):
    # tev is the date difference. The event's study day is tev + 1.
    if day >= tev + 1:
        return "drop"
    if event == 1 and tev + 1 <= day + H:
        return "1"
    if tev + 1 >= day + H:
        return "0"
    return "ne"


def ci(a):
    v = np.asarray(a, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def fmt3(x):
    return f"{x:.3f}"


def fmt_num(v):
    if abs(v - round(v)) < 0.05:
        return str(int(round(v)))
    return f"{v:.1f}"


def iqr_cell(series):
    s = pd.Series(series).dropna().to_numpy(float)
    if len(s) == 0:
        return "—", 0
    med = float(np.median(s))
    lo, hi = np.percentile(s, [25, 75])
    return f"{fmt_num(med)} ({fmt_num(lo)}–{fmt_num(hi)}), n={len(s)}", len(s)


def main():
    clock = pd.read_excel(ROOT / "data/revision/clock_bbdd.xlsx")
    rang = pd.read_csv(ROOT / "results/by_db/DB1/topics_meses_rangos_fechas.csv")
    clock["id"] = clock["id"].astype(int)
    rang["id"] = rang["id"].astype(int)
    off = clock.merge(rang[["id", "fecha_inicio_daily"]], on="id", how="left")
    off["fecha_inicio"] = pd.to_datetime(off["fecha_inicio"])
    off["fecha_fin"] = pd.to_datetime(off["fecha_fin"])
    off["fecha_inicio_daily"] = pd.to_datetime(off["fecha_inicio_daily"])
    off["offset_days"] = (off["fecha_inicio_daily"] - off["fecha_inicio"]).dt.days
    off["t_evento_eb2"] = off["t_evento"] - off["offset_days"]
    off["evento"] = off["evento"].astype(int)
    omap = off.set_index("id").to_dict("index")

    shifted = off.loc[off["offset_days"].fillna(0) > 0,
                      ["id", "fecha_inicio", "fecha_inicio_daily", "fecha_fin", "evento",
                       "t_evento", "offset_days", "t_evento_eb2"]].sort_values("id")
    shifted.to_csv(BT / "pacientes_offset_teb2.csv", index=False)

    daily = pd.read_csv(BT / "daily_sliding_puf_clock_cierre_mes_alineado.csv")
    daily["id"] = daily["id"].astype(int)
    qmap = {int(pid): dict(zip(g.study_day.astype(int), g.pUF.to_numpy(float)))
            for pid, g in daily.groupby("id")}

    def meta(pid):
        r = omap[int(pid)]
        return int(r["evento"]), float(r["t_evento_eb2"]), r

    # --- decision points and alerts ---
    point_rows = []
    alert_rows = []
    per = {n: {} for n, _ in STEPS}
    ant_rows = []
    for name, step in STEPS:
        for pid, q in qmap.items():
            ev, tev, r = meta(pid)
            opened = tp = fp = ne = holes = detected = 0
            t = 90
            na = nh = 90
            raw_any = raw_tp = blk_any = blk_tp = np.nan
            while t < tev + 1:
                ok = (t - 60) in q and (t - 30) in q and t in q
                b = (q[t - 60] + q[t - 30] + q[t]) / 3 if ok else np.nan
                if not ok:
                    status_pt = "missing_q"
                    y = ""
                else:
                    y = label(t, tev, ev)
                    status_pt = {"1": "positive", "0": "negative", "ne": "unresolved", "drop": "drop"}[y]
                point_rows.append(dict(
                    cadence=name, step_days=step, id=pid, study_day=t,
                    fecha_decision=(r["fecha_inicio_daily"] + pd.Timedelta(days=t - 1)).strftime("%Y-%m-%d")
                    if pd.notna(r["fecha_inicio_daily"]) else "",
                    fecha_inicio=r["fecha_inicio"].strftime("%Y-%m-%d"),
                    fecha_inicio_daily=r["fecha_inicio_daily"].strftime("%Y-%m-%d")
                    if pd.notna(r["fecha_inicio_daily"]) else "",
                    fecha_fin=r["fecha_fin"].strftime("%Y-%m-%d"),
                    t_evento_eb2=int(tev),
                    event=ev, score_B=b, label=y, status=status_pt, horizon_days=H,
                ))
                if ok:
                    if t >= nh:
                        holes += 1
                        nh = t + 30
                    # anticipation and alerts use the cut filled later; store B now
                t += step
            per[name][pid] = dict(holes=holes, q=q, ev=ev, tev=tev, meta=r)
        # second pass after cut is known happens below
    # The loop above stored points but alert/anticipation need the cut.
    # Rebuild alerts after Youden. Points are complete.

    pts = pd.DataFrame(point_rows)

    # evaluable complete months for the granularity cut: every-30 rows with label 0/1
    # Use monthly_mean from the 1,009 file, relabelled, so the cut is the same score the original Youden used.
    m1009 = pd.read_csv(BT / "puntos_meses_completos_1009.csv")
    m1009["id"] = m1009["id"].astype(int)
    m1009 = m1009.merge(off[["id", "t_evento", "t_evento_eb2", "evento", "offset_days",
                             "fecha_inicio", "fecha_inicio_daily", "fecha_fin"]], on="id", how="left")
    m1009["y"] = [label(d, tv, ev) for d, tv, ev in zip(m1009.day, m1009.t_evento_eb2, m1009.evento)]
    fit = m1009[m1009.y.isin(["0", "1"])].copy()
    fit["y"] = fit.y.astype(int)
    cut_mean = youden(fit.y.to_numpy(), fit.monthly_mean.to_numpy(float))
    TH = cut_mean["threshold"]
    cut_mean["N_points"] = int(len(fit))
    cut_mean["N_patients"] = int(fit.id.nunique())
    cut_mean["positive_points"] = int(fit.y.sum())
    cut_mean["negative_points"] = int((1 - fit.y).sum())
    cut_mean["unique_events"] = int(fit.loc[fit.y == 1, "id"].nunique())
    cut_mean["AUROC"] = auroc(fit.y, fit.monthly_mean)
    # apply this cut to B for the confusion of the same points
    at_b = clasif(fit.y.to_numpy(), fit.B.to_numpy(float), TH)
    cut_mean["at_B"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in at_b.items()}

    # section 2, the 1,016
    d1016 = pd.read_csv(ROOT / "results/alarm/runs/Evento_DB1_clock/paper_W3_H4/decision_points_W3_H4.csv")
    d1016["id"] = d1016["id"].astype(int)
    d1016 = d1016.merge(off[["id", "t_evento", "t_evento_eb2", "evento", "offset_days",
                             "fecha_inicio", "fecha_inicio_daily", "fecha_fin"]], on="id", how="left")
    d1016["y"] = [label(d, tv, ev) for d, tv, ev in zip(d1016.decision_day, d1016.t_evento_eb2, d1016.evento)]
    fit16 = d1016[d1016.y.isin(["0", "1"])].copy()
    fit16["y"] = fit16.y.astype(int)
    cut16 = youden(fit16.y.to_numpy(), fit16.AUC_W.to_numpy(float))
    cut16["N_points"] = int(len(fit16))
    cut16["N_patients"] = int(fit16.id.nunique())
    cut16["positive_points"] = int(fit16.y.sum())
    cut16["negative_points"] = int((1 - fit16.y).sum())
    cut16["unique_events"] = int(fit16.loc[fit16.y == 1, "id"].nunique())
    cut16["AUROC"] = auroc(fit16.y, fit16.AUC_W)
    op16 = clasif(fit16.y.to_numpy(), fit16.AUC_W.to_numpy(float), cut16["threshold"])
    cut16.update({k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in op16.items()})

    # section 2 bootstrap
    rng16 = np.random.default_rng(SEED)
    ids16 = np.array(sorted(fit16.id.unique()))
    pos16 = {pid: i for i, pid in enumerate(ids16)}
    ix16 = fit16.id.map(pos16).to_numpy()
    y16 = fit16.y.to_numpy(float)
    x16 = fit16.AUC_W.to_numpy(float)
    draws16 = rng16.integers(0, len(ids16), size=(N_BOOT, len(ids16)))
    boot16 = np.empty((N_BOOT, 5))
    for b in range(N_BOOT):
        c = np.bincount(draws16[b], minlength=len(ids16)).astype(float)
        m = clasif(y16, x16, cut16["threshold"], c[ix16])
        boot16[b] = [m["AUROC"], m["sensitivity"], m["specificity"], m["PPV"], m["NPV"]]
    ci16 = {k: ci(boot16[:, i]) for i, k in enumerate(["AUROC", "sensitivity", "specificity", "PPV", "NPV"])}

    # alerts, anticipation, per-patient holes already counted; recount alerts at TH
    per_alert = {n: {} for n, _ in STEPS}
    ant = []
    alerts = []
    for name, step in STEPS:
        for pid, q in qmap.items():
            ev, tev, r = meta(pid)
            opened = tp = fp = ne = holes = detected = 0
            t = 90
            na = nh = 90
            raw_any = raw_tp = blk_any = blk_tp = np.nan
            while t < tev + 1:
                if (t - 60) in q and (t - 30) in q and t in q:
                    b = (q[t - 60] + q[t - 30] + q[t]) / 3
                    if t >= nh:
                        holes += 1
                        nh = t + 30
                    if b >= TH:
                        if np.isnan(raw_any):
                            raw_any = t
                        if np.isnan(raw_tp) and ev == 1 and tev + 1 <= t + H:
                            raw_tp = t
                        if t >= na:
                            na = t + 30
                            opened += 1
                            if ev == 1 and tev + 1 <= t + H:
                                status, tp = "TP", tp + 1
                                detected = 1
                                if np.isnan(blk_tp):
                                    blk_tp = t
                            elif tev + 1 >= t + H:
                                status, fp = "FP", fp + 1
                            else:
                                status, ne = "NE", ne + 1
                            if np.isnan(blk_any):
                                blk_any = t
                            alerts.append(dict(
                                cadence=name, id=int(pid), study_day=int(t),
                                fecha_alert=(r["fecha_inicio_daily"] + pd.Timedelta(days=t - 1)).strftime("%Y-%m-%d"),
                                fecha_inicio=r["fecha_inicio"].strftime("%Y-%m-%d"),
                                fecha_inicio_daily=r["fecha_inicio_daily"].strftime("%Y-%m-%d"),
                                fecha_fin=r["fecha_fin"].strftime("%Y-%m-%d"),
                                t_evento_eb2=int(tev),
                                event=ev, score_B=b, label=status, horizon_days=H,
                            ))
                t += step
            per_alert[name][pid] = dict(opened=opened, tp=tp, fp=fp, ne=ne, holes=holes, detected=detected)
            if ev == 1 and tev + 1 > 90 and max(q) >= 90:
                def fdate(day):
                    if np.isnan(day) or pd.isna(r["fecha_inicio_daily"]):
                        return ""
                    return (r["fecha_inicio_daily"] + pd.Timedelta(days=int(day) - 1)).strftime("%Y-%m-%d")

                def lead(day):
                    return float((tev + 1) - day) if not np.isnan(day) else np.nan

                ant.append(dict(
                    cadence=name, step_days=step, id=int(pid),
                    t_evento_eb2=int(tev),
                    fecha_inicio=r["fecha_inicio"].strftime("%Y-%m-%d"),
                    fecha_inicio_daily=r["fecha_inicio_daily"].strftime("%Y-%m-%d"),
                    fecha_fin=r["fecha_fin"].strftime("%Y-%m-%d"),
                    first_alarm_day=raw_any, first_alarm_date=fdate(raw_any), first_alarm_lead=lead(raw_any),
                    first_tp_day=raw_tp, first_tp_date=fdate(raw_tp), first_tp_lead=lead(raw_tp),
                    first_issued_day=blk_any, first_issued_date=fdate(blk_any), first_issued_lead=lead(blk_any),
                    first_issued_tp_day=blk_tp, first_issued_tp_date=fdate(blk_tp), first_issued_tp_lead=lead(blk_tp),
                    denominator_events=None,
                ))
    ant = pd.DataFrame(ant)
    n48 = int(ant.loc[ant.cadence == "Daily", "id"].nunique())
    ant["denominator_events"] = n48
    alerts = pd.DataFrame(alerts)

    # mark decision points included, attach cadence denominators
    evset = {}
    for name, _ in STEPS:
        sub = pts[(pts.cadence == name) & (pts.label.isin(["0", "1"]))]
        evset[name] = sub
    for name, _ in STEPS:
        sub = evset[name]
        n_pts = int(len(sub))
        n_pat = int(sub.id.nunique())
        n_ev = int(sub.loc[sub.label == "1", "id"].nunique())
        mask = pts.cadence == name
        pts.loc[mask, "included_in_metrics"] = pts.loc[mask, "label"].isin(["0", "1"]).astype(int)
        pts.loc[mask, "denom_evaluable_points"] = n_pts
        pts.loc[mask, "denom_patients"] = n_pat
        pts.loc[mask, "denom_events"] = n_ev
        pts.loc[mask, "score_cut"] = TH
    amap = {name: alerts[alerts.cadence == name] for name, _ in STEPS}
    for name, _ in STEPS:
        a = amap[name]
        judged = int(a.label.isin(["TP", "FP"]).sum())
        openings = int(sum(v["holes"] for v in per_alert[name].values()))
        mask = alerts.cadence == name
        alerts.loc[mask, "denom_openings"] = openings
        alerts.loc[mask, "denom_alerts_judged"] = judged
        alerts.loc[mask, "denom_not_evaluable"] = int((a.label == "NE").sum())
        alerts.loc[mask, "denom_events"] = n48
        alerts.loc[mask, "score_cut"] = TH

    # cadence metrics + bootstrap at fixed TH, score B, evaluable only
    packed = {}
    for name, _ in STEPS:
        sub = evset[name]
        packed[name] = (sub.id.to_numpy(), sub.label.astype(int).to_numpy(float), sub.score_B.to_numpy(float))
    names = [n for n, _ in STEPS]
    disc = sorted(set.intersection(*[set(packed[n][0]) for n in names]))
    ev_ids = sorted(ant.loc[ant.cadence == "Daily", "id"].unique())
    hole_ids = sorted(set.union(*[{pid for pid, v in per_alert[n].items() if v["holes"] > 0} for n in names]))
    pos = {pid: i for i, pid in enumerate(disc)}
    rng = np.random.default_rng(SEED)
    D, E, Hh = len(disc), len(ev_ids), len(hole_ids)
    dd = rng.integers(0, D, size=(N_BOOT, D))
    de = rng.integers(0, E, size=(N_BOOT, E))
    dh = rng.integers(0, Hh, size=(N_BOOT, Hh))
    est, boots = {}, {}
    for n in names:
        ids, y, x = packed[n]
        est[n] = clasif(y, x, TH)
        ix = np.array([pos[i] for i in ids])
        boots[n] = np.empty((N_BOOT, 5))
        packed[n] = (y, x, ix)
    for b in range(N_BOOT):
        c = np.bincount(dd[b], minlength=D).astype(float)
        for n in names:
            y, x, ix = packed[n]
            m = clasif(y, x, TH, c[ix])
            boots[n][b] = [m["AUROC"], m["sensitivity"], m["specificity"], m["PPV"], m["NPV"]]
    det = np.zeros((4, E))
    for i, n in enumerate(names):
        detected = {pid: per_alert[n][pid]["detected"] for pid in per_alert[n]}
        for j, pid in enumerate(ev_ids):
            det[i, j] = detected.get(pid, 0)
    eb = np.empty((4, N_BOOT))
    for b in range(N_BOOT):
        c = np.bincount(de[b], minlength=E).astype(float)
        eb[:, b] = (det * c).sum(1) / c.sum()
    jud = np.zeros((4, Hh))
    ho = np.zeros((4, Hh))
    for i, n in enumerate(names):
        for j, pid in enumerate(hole_ids):
            v = per_alert[n].get(pid)
            if v:
                jud[i, j] = v["tp"] + v["fp"]
                ho[i, j] = v["holes"]
    bb = np.empty((4, N_BOOT))
    for b in range(N_BOOT):
        c = np.bincount(dh[b], minlength=Hh).astype(float)
        num = (jud * c).sum(1)
        den = (ho * c).sum(1)
        bb[:, b] = np.where(den > 0, 100 * num / den, np.nan)

    metric_rows = []
    for i, n in enumerate(names):
        y, x, _ = packed[n]
        pred = x >= TH
        rec = dict(
            cadence=n, n=int(len(y)), patients=int(len(set(evset[n].id))),
            events=int(evset[n].loc[evset[n].label == "1", "id"].nunique()),
            TP=int(((pred) & (y == 1)).sum()), FN=int(((~pred) & (y == 1)).sum()),
            FP=int(((pred) & (y == 0)).sum()), TN=int(((~pred) & (y == 0)).sum()),
            alerts=int(sum(v["opened"] for v in per_alert[n].values())),
            alerts_judged=int(jud[i].sum()), holes=int(ho[i].sum()),
            ne=int(sum(v["ne"] for v in per_alert[n].values())),
            burden=float(100 * jud[i].sum() / ho[i].sum()),
            detected=int(det[i].sum()), detected_p=float(det[i].mean()), detected_denom=n48,
        )
        for k, mname in enumerate(["AUROC", "sensitivity", "specificity", "PPV", "NPV"]):
            lo, hi = ci(boots[n][:, k])
            rec[mname] = est[n][mname]
            rec[mname + "_lo"] = lo
            rec[mname + "_hi"] = hi
        lo, hi = ci(eb[i])
        rec["detected_lo"], rec["detected_hi"] = lo, hi
        lo, hi = ci(bb[i])
        rec["burden_lo"], rec["burden_hi"] = lo, hi
        # funnel
        sub = pts[pts.cadence == n]
        rec["calendar"] = int(len(sub))
        rec["missing_q"] = int((sub.status == "missing_q").sum())
        rec["with_B"] = int(sub.status.isin(["positive", "negative", "unresolved"]).sum())
        rec["unresolved"] = int((sub.status == "unresolved").sum())
        metric_rows.append(rec)
    metrics = pd.DataFrame(metric_rows)

    diffs = []
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        rec = {"a": names[i], "b": names[j]}
        for k, mname in enumerate(["AUROC", "sensitivity", "specificity", "PPV", "NPV"]):
            delta = boots[names[i]][:, k] - boots[names[j]][:, k]
            lo, hi = ci(delta)
            rec[mname] = est[names[i]][mname] - est[names[j]][mname]
            rec[mname + "_lo"] = lo
            rec[mname + "_hi"] = hi
        delta = eb[i] - eb[j]
        lo, hi = ci(delta)
        rec["detected"] = float(det[i].mean() - det[j].mean())
        rec["detected_lo"], rec["detected_hi"] = lo, hi
        delta = bb[i] - bb[j]
        lo, hi = ci(delta)
        rec["burden"] = float(metrics.iloc[i].burden - metrics.iloc[j].burden)
        rec["burden_lo"], rec["burden_hi"] = lo, hi
        diffs.append(rec)
    diffs = pd.DataFrame(diffs)

    # anticipation table + paired bootstrap
    ant_summary = {}
    for n in names:
        sub = ant[ant.cadence == n]
        ant_summary[n] = {
            "any": iqr_cell(sub.first_alarm_lead)[0],
            "tp": iqr_cell(sub.first_tp_lead)[0],
            "issued_tp": iqr_cell(sub.first_issued_tp_lead)[0],
        }
    rng_a = np.random.default_rng(SEED)
    pair_rows = []
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        a = ant[ant.cadence == names[i]].set_index("id")
        b = ant[ant.cadence == names[j]].set_index("id")
        det_a = set(a.index[~a.first_issued_tp_day.isna()])
        det_b = set(b.index[~b.first_issued_tp_day.isna()])
        both = sorted(det_a & det_b)
        only_a = sorted(det_a - det_b)
        only_b = sorted(det_b - det_a)
        def leads(frame, ids, col):
            return frame.loc[ids, col].to_numpy(float)
        raw = leads(a, both, "first_tp_lead") - leads(b, both, "first_tp_lead")
        issued = leads(a, both, "first_issued_tp_lead") - leads(b, both, "first_issued_tp_lead")
        draws = rng_a.integers(0, len(both), size=(N_BOOT, len(both))) if both else np.zeros((N_BOOT, 0), int)
        def boot_med(values):
            if len(values) == 0:
                return np.nan, [np.nan, np.nan]
            out = np.empty(N_BOOT)
            for bidx in range(N_BOOT):
                out[bidx] = np.median(values[draws[bidx]])
            return float(np.median(values)), ci(out)
        raw_med, raw_ci = boot_med(raw)
        iss_med, iss_ci = boot_med(issued)
        pair_rows.append(dict(
            a=names[i], b=names[j], both=len(both),
            only_a=", ".join(str(x) for x in only_a) or "—",
            only_b=", ".join(str(x) for x in only_b) or "—",
            raw_tp=raw_med, raw_tp_lo=raw_ci[0], raw_tp_hi=raw_ci[1],
            issued_tp=iss_med, issued_tp_lo=iss_ci[0], issued_tp_hi=iss_ci[1],
        ))
    pairs = pd.DataFrame(pair_rows)

    # alternative metrics on corrected 1,016 labels
    alt = pd.read_csv(BT / "metricas_alternativas_1016.csv")
    alt["id"] = alt["id"].astype(int)
    alt = alt.merge(off[["id", "t_evento_eb2", "evento"]], on="id", how="left")
    alt["y"] = [label(d, tv, ev) for d, tv, ev in zip(alt.day, alt.t_evento_eb2, alt.evento)]
    alt_ev = alt[alt.y.isin(["0", "1"])].copy()
    alt_ev["y"] = alt_ev.y.astype(int)
    alt_rows = []
    for col in ["auc", "std", "slope", "n_changepoints", "n_threshold_crossings_0.5",
                "n_threshold_crossings_0.6", "n_threshold_crossings_0.7", "n_threshold_crossings_0.8"]:
        ev = alt_ev.loc[alt_ev.y == 1, col]
        nev = alt_ev.loc[alt_ev.y == 0, col]
        alt_rows.append(dict(
            metric=col,
            mean_event=float(ev.mean()), sd_event=float(ev.std(ddof=1)),
            mean_no_event=float(nev.mean()), sd_no_event=float(nev.std(ddof=1)),
            AUROC=auroc(alt_ev.y, alt_ev[col]),
            n=int(len(alt_ev)), n_event_points=int(alt_ev.y.sum()),
            n_no_event_points=int((1 - alt_ev.y).sum()),
        ))
    alt_sum = pd.DataFrame(alt_rows)

    # 23003 lead check
    p23003 = ant[(ant.id == 23003)][["cadence", "t_evento_eb2", "first_issued_tp_day", "first_issued_tp_lead",
                                    "first_tp_day", "first_tp_lead"]].to_dict("records")

    # save
    pts.to_csv(BT / "puntos_decision_teb2.csv", index=False)
    alerts.to_csv(BT / "avisos_teb2.csv", index=False)
    ant.to_csv(BT / "anticipacion_por_paciente_teb2.csv", index=False)
    pairs.to_csv(BT / "anticipacion_pares_teb2.csv", index=False)
    metrics.to_csv(BT / "metricas_cadencia_bootstrap_teb2.csv", index=False)
    diffs.to_csv(BT / "metricas_cadencia_diferencias_bootstrap_teb2.csv", index=False)
    pd.DataFrame([cut_mean]).to_csv(BT / "youden_meses_teb2.csv", index=False)
    pd.DataFrame([{**cut16, **{k + "_lo": ci16[k][0] for k in ci16}, **{k + "_hi": ci16[k][1] for k in ci16}}]).to_csv(
        BT / "youden_1016_teb2.csv", index=False)
    alt_sum.to_csv(BT / "metricas_alternativas_resumen_teb2.csv", index=False)
    alt.to_csv(BT / "metricas_alternativas_1016_teb2.csv", index=False)
    den = metrics[["cadence", "calendar", "missing_q", "with_B", "unresolved", "n", "patients", "events",
                   "alerts", "TP", "FN", "FP", "TN", "ne", "alerts_judged", "holes", "burden",
                   "detected", "detected_denom"]].copy()
    den.to_csv(BT / "denominadores_teb2.csv", index=False)

    summary = dict(
        TH=TH, cut_mean=cut_mean, cut16=cut16, ci16=ci16,
        metrics=metrics.to_dict("records"), diffs=diffs.to_dict("records"),
        ant=ant_summary, pairs=pairs.to_dict("records"),
        alt=alt_sum.to_dict("records"), p23003=p23003,
        n_disc=len(disc), n48=n48, n_hole=len(hole_ids),
        n_shifted=int(len(shifted)),
        fit_drop=int((m1009.y == "drop").sum()), fit_ne=int((m1009.y == "ne").sum()),
        fit16_drop=int((d1016.y == "drop").sum()), fit16_ne=int((d1016.y == "ne").sum()),
    )
    (BT / "resumen_teb2.json").write_text(json.dumps(summary, default=float, indent=2))
    print("TH", TH)
    print("1016", cut16["N_points"], cut16["N_patients"], cut16["positive_points"], cut16["threshold"], cut16["TP"], cut16["FN"], cut16["FP"], cut16["TN"])
    print(metrics[["cadence", "n", "patients", "events", "TP", "FN", "FP", "TN", "burden", "detected", "alerts", "ne", "holes"]].to_string(index=False))
    print(pairs.to_string(index=False))
    print("23003", p23003)
    print("alt auc", alt_sum.iloc[0].to_dict())
    return summary


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Point 2A — fitted load × coverage interaction on the 1013 objetivo decisions.

Logistic model of the 120-day outcome on the frozen load, the real coverage and
their product: logit P(y) = b0 + b1 load_z + b2 cov_z + b3 load_z·cov_z.
Load and coverage are z-scored on the 1013 decisions of each window, so b1 and
b2 are per SD and b3 is the change in the load log-odds per SD of coverage.
No penalty. Uncertainty is per patient: a patient bootstrap (percentile) and
cluster-robust (sandwich) standard errors with the patient as cluster.
Discrimination is measured as for the frozen baseline: fixed scores on the same decisions, patient bootstrap.

Reads results/missingness/Real_coverage/coverage_by_decision.csv and writes
coverage_interaction_coefficients.csv and coverage_interaction_auroc.csv there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/missingness/Real_coverage"
SRC = OUT / "coverage_by_decision.csv"
WINDOWS = (90, 30)
TERMS = ("intercept", "load_z", "cov_z", "load_z_x_cov_z")
N_BOOT = 1000
BOOT_SEED = 20261119


def design(frame: pd.DataFrame, mu_l: float, sd_l: float, mu_c: float, sd_c: float) -> np.ndarray:
    lz = (frame["load"].to_numpy(float) - mu_l) / sd_l
    cz = (frame["coverage"].to_numpy(float) - mu_c) / sd_c
    return np.column_stack([lz, cz, lz * cz])


def fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-10).fit(x, y)
    return np.concatenate([m.intercept_, m.coef_.ravel()])


def cluster_se(x: np.ndarray, y: np.ndarray, beta: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Sandwich SE with the patient as cluster, small-sample factor G/(G-1)."""
    xd = np.column_stack([np.ones(len(x)), x])
    p = 1.0 / (1.0 + np.exp(-xd @ beta))
    bread = np.linalg.inv(xd.T @ (xd * (p * (1 - p))[:, None]))
    score = xd * (y - p)[:, None]
    ids = pd.unique(groups)
    meat = np.zeros((xd.shape[1], xd.shape[1]))
    for g in ids:
        s = score[groups == g].sum(axis=0)
        meat += np.outer(s, s)
    k = len(ids)
    cov = bread @ meat @ bread * k / (k - 1)
    return np.sqrt(np.diag(cov))


def main() -> None:
    cov = pd.read_csv(SRC)
    coef_rows, auc_rows = [], []
    for w in WINDOWS:
        sub = cov.loc[(cov.window_days == w) & (cov.emitted == 1)].reset_index(drop=True)
        if len(sub) != 1013 or sub["id"].nunique() != 114:
            raise SystemExit(f"window {w}: {len(sub)} decisions / {sub.id.nunique()} patients")
        y = sub["y"].to_numpy(int)
        groups = sub["id"].to_numpy()
        st = (sub.load.mean(), sub.load.std(ddof=0), sub.coverage.mean(), sub.coverage.std(ddof=0))
        x = design(sub, *st)
        beta = fit(x, y)
        se = cluster_se(x, y, beta, groups)

        rng = np.random.default_rng(BOOT_SEED + w)
        by = {pid: idx for pid, idx in sub.groupby("id", sort=False).indices.items()}
        ids = np.array(list(by))
        boots = []
        for _ in range(N_BOOT):
            idx = np.concatenate([by[i] for i in rng.choice(ids, len(ids), replace=True)])
            if len(np.unique(y[idx])) < 2:
                continue
            boots.append(fit(x[idx], y[idx]))
        boots = np.array(boots)

        z = stats.norm.ppf(0.975)
        for j, term in enumerate(TERMS):
            lo_b, hi_b = np.quantile(boots[:, j], [0.025, 0.975])
            coef_rows.append({
                "window_days": w,
                "term": term,
                "beta": beta[j],
                "OR": float(np.exp(beta[j])),
                "se_cluster": se[j],
                "beta_lo_cluster": beta[j] - z * se[j],
                "beta_hi_cluster": beta[j] + z * se[j],
                "p_cluster": float(2 * stats.norm.sf(abs(beta[j] / se[j]))),
                "beta_lo_boot": float(lo_b),
                "beta_hi_boot": float(hi_b),
                "OR_lo_boot": float(np.exp(lo_b)),
                "OR_hi_boot": float(np.exp(hi_b)),
                "n_boot": int(len(boots)),
                "n_decisions": int(len(sub)),
                "n_patients": int(sub["id"].nunique()),
                "load_mean": st[0], "load_sd": st[1], "coverage_mean": st[2], "coverage_sd": st[3],
            })

        # Discrimination is measured exactly as for the frozen baseline: fixed scores on the same
        # 1013 decisions, no cross-validation, patient bootstrap. The fitted model's score is its
        # linear predictor; a load-only logit would rank exactly as the frozen load (AUROC 0.703).
        lin = np.column_stack([np.ones(len(x)), x]) @ beta
        sub = sub.assign(fitted=lin)
        auc_fit = float(roc_auc_score(y, lin))
        auc_load = float(roc_auc_score(y, sub["load"]))
        rng = np.random.default_rng(BOOT_SEED + 1000 + w)
        bf, bd = [], []
        for _ in range(N_BOOT):
            idx = np.concatenate([by[i] for i in rng.choice(ids, len(ids), replace=True)])
            if len(np.unique(y[idx])) < 2:
                continue
            af = roc_auc_score(y[idx], lin[idx])
            bf.append(af)
            bd.append(af - roc_auc_score(y[idx], sub["load"].to_numpy()[idx]))
        if w == 90:
            ref = pd.read_csv(OUT / "performance_by_band.csv")
            ref = ref[(ref.window_days == 90) & (ref.band == "all")].iloc[0]
            if round(auc_load, 6) != round(float(ref.AUROC), 6):
                raise SystemExit(f"frozen load AUROC {auc_load} is not the reference {ref.AUROC}")
            auc_rows.append({"window_days": w, "score": "frozen load (reference)", "AUROC": auc_load,
                             "AUROC_lo": float(ref.AUROC_lo), "AUROC_hi": float(ref.AUROC_hi),
                             "diff_vs_frozen": np.nan, "diff_lo": np.nan, "diff_hi": np.nan})
        auc_rows.append({"window_days": w, "score": "load + coverage + load×coverage",
                         "AUROC": auc_fit,
                         "AUROC_lo": float(np.quantile(bf, 0.025)), "AUROC_hi": float(np.quantile(bf, 0.975)),
                         "diff_vs_frozen": auc_fit - auc_load,
                         "diff_lo": float(np.quantile(bd, 0.025)), "diff_hi": float(np.quantile(bd, 0.975))})

    coefs = pd.DataFrame(coef_rows)
    aucs = pd.DataFrame(auc_rows)
    coefs.to_csv(OUT / "coverage_interaction_coefficients.csv", index=False)
    aucs.to_csv(OUT / "coverage_interaction_auroc.csv", index=False)
    prov = OUT / "PROVENANCE.json"
    meta = json.loads(prov.read_text(encoding="utf-8")) if prov.exists() else {}
    meta["coverage_interaction"] = {
        "script": "scripts/03_analysis/alarm/run_coverage_interaction.py",
        "model": "logit P(y120) = b0 + b1 load_z + b2 cov_z + b3 load_z*cov_z, no penalty, z-scored on the 1013 decisions per window",
        "uncertainty": f"patient bootstrap percentile ({N_BOOT}, seed {BOOT_SEED}+window) and cluster-robust sandwich SE by patient",
        "discrimination": "same as the frozen baseline: fixed scores (linear predictor) on the 1013 decisions, no cross-validation; patient bootstrap CI and paired difference vs the frozen load",
        "replaces": "the earlier fixed score load_z + load_z*cov_z (all weights 1, no coverage main effect, nothing fitted)",
    }
    prov.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(coefs[["window_days", "term", "beta", "OR", "beta_lo_boot", "beta_hi_boot", "beta_lo_cluster", "beta_hi_cluster", "p_cluster"]].round(3).to_string(index=False))
    print(aucs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

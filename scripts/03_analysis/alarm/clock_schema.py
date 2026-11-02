"""Universal clinical clock for alarm landmarks.

Single source of truth: a user-supplied BBDD with
``id``, ``fecha_inicio``, ``fecha_fin``, ``evento``.

Derived (never taken from Subjects / leftover master fields):
  t_evento = (fecha_fin - fecha_inicio).days
  Evento PD = evento
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_COLS = ("id", "fecha_inicio", "fecha_fin", "evento")

# Known aliases → canonical names (first match wins when normalizing)
_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "s_record_id", "S_RECORD_id"),
    "fecha_inicio": (
        "fecha_inicio",
        "Fecha_inicio_actualizada",
        "fecha_inicio_actualizada",
        "Inicio_datos_DB1",
        "inicio",
    ),
    "fecha_fin": (
        "fecha_fin",
        "Fecha_Fin_DB1_Actualizado",
        "fecha_fin_db1_actualizado",
        "Fin_datos_DB1",
        "fin",
    ),
    "evento": (
        "evento",
        "Evento_DB1",
        "evento_db1",
        "PD_event",
        "Evento PD",
        "event",
    ),
}


def project_root_from(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return Path(__file__).resolve().parents[3]


def default_clock_path(root: Path | None = None) -> Path:
    root = project_root_from(root)
    return root / "data/revision/clock_bbdd.xlsx"


def default_audit_db1_path(root: Path | None = None) -> Path:
    root = project_root_from(root)
    return root / "data/revision/178 audit oct 2 2026 copia.xlsx"


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clock / audit file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported clock file type: {path.suffix}")


def _find_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    cols = {str(c).strip(): c for c in df.columns}
    lower = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def normalize_clock_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical columns; leave other cols intact."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename: dict[str, str] = {}
    for canon, aliases in _ALIASES.items():
        if canon in out.columns:
            continue
        found = _find_col(out, aliases)
        if found is not None and found != canon:
            rename[found] = canon
    return out.rename(columns=rename)


def validate_clock(
    clock: pd.DataFrame,
    *,
    keep_t_evento: bool = False,
) -> pd.DataFrame:
    """Validate and return a clean clock frame with ``t_evento``.

    If ``keep_t_evento`` and a ``t_evento`` column is present (e.g. already capped
    to complete months), that value is kept. Otherwise
    ``t_evento = (fecha_fin − fecha_inicio).days``.
    """
    missing = [c for c in CANONICAL_COLS if c not in clock.columns]
    if missing:
        raise ValueError(
            f"Clock BBDD missing columns {missing}. "
            f"Need: {list(CANONICAL_COLS)}. Got: {list(clock.columns)}"
        )
    extras = [
        c
        for c in (
            "t_evento",
            "t_evento_audit",
            "n_days",
            "n_complete_months",
            "t_complete_months",
            "days_capped",
        )
        if c in clock.columns
    ]
    out = clock[list(CANONICAL_COLS) + extras].copy()
    out = out[pd.to_numeric(out["id"], errors="coerce").notna()].copy()
    out["id"] = out["id"].astype(int)
    out["evento"] = pd.to_numeric(out["evento"], errors="coerce")
    out = out[out["evento"].isin([0, 1])].copy()
    out["evento"] = out["evento"].astype(int)
    out["fecha_inicio"] = pd.to_datetime(out["fecha_inicio"], errors="coerce")
    out["fecha_fin"] = pd.to_datetime(out["fecha_fin"], errors="coerce")
    bad = out["fecha_inicio"].isna() | out["fecha_fin"].isna()
    if bad.any():
        bad_ids = out.loc[bad, "id"].tolist()[:10]
        raise ValueError(f"Clock has missing dates for ids (sample): {bad_ids}")
    audit_days = (out["fecha_fin"] - out["fecha_inicio"]).dt.days.astype(float)
    if keep_t_evento and "t_evento" in out.columns and out["t_evento"].notna().any():
        out["t_evento"] = pd.to_numeric(out["t_evento"], errors="coerce")
        out["t_evento"] = out["t_evento"].fillna(audit_days)
        if "t_evento_audit" not in out.columns:
            out["t_evento_audit"] = audit_days
    else:
        out["t_evento"] = audit_days
        out["t_evento_audit"] = audit_days
    if (out["t_evento"] < 0).any():
        neg = out.loc[out["t_evento"] < 0, "id"].tolist()[:10]
        raise ValueError(f"Clock has fecha_fin < fecha_inicio for ids (sample): {neg}")
    out = out.drop_duplicates("id", keep="last").sort_values("id").reset_index(drop=True)
    return out


def load_clock(path: Path | str) -> pd.DataFrame:
    """Load and validate a universal clock BBDD (xlsx/csv).

    Preserves a stored ``t_evento`` when present (complete-month capped clocks).
    """
    raw = normalize_clock_columns(_read_table(Path(path)))
    keep = "t_evento" in raw.columns
    return validate_clock(raw, keep_t_evento=keep)

def adapt_audit_db1_to_clock(audit_path: Path | str | None = None, root: Path | None = None) -> pd.DataFrame:
    """Map audit DB1 columns → canonical clock schema."""
    path = Path(audit_path) if audit_path else default_audit_db1_path(root)
    raw = normalize_clock_columns(_read_table(path))
    return validate_clock(raw)


def default_day_table_path(root: Path | None = None) -> Path:
    root = project_root_from(root)
    return root / "data/revision/PD_cutoff_topics_revision_a0.csv"


def sequence_lengths_from_day_table(day_csv: Path | str) -> pd.DataFrame:
    """Per-id sequence length (days) from the revision day / embeddings table."""
    import ast

    df = pd.read_csv(day_csv, usecols=lambda c: c in {"id", "n_days", "embedding_ids"})
    df["id"] = df["id"].astype(int)
    if "n_days" in df.columns and df["n_days"].notna().all():
        out = df[["id", "n_days"]].drop_duplicates("id")
        out["n_days"] = out["n_days"].astype(int)
        return out
    def _len(v) -> int:
        if isinstance(v, str):
            return len(ast.literal_eval(v))
        return len(v)

    out = df[["id", "embedding_ids"]].drop_duplicates("id").copy()
    out["n_days"] = out["embedding_ids"].map(_len).astype(int)
    return out[["id", "n_days"]]


def cap_clock_to_complete_months(
    clock: pd.DataFrame,
    day_csv: Path | str | None = None,
    *,
    root: Path | None = None,
    window_size: int = 30,
) -> pd.DataFrame:
    """Same rule as monthly topics: do not count an incomplete final month.

    Keeps audit ``fecha_inicio`` / ``fecha_fin`` / ``evento`` as recorded, but sets
    analysis ``t_evento`` to::

        min(t_evento_audit, (n_days // window_size) * window_size)

    so the clinical end date does not extend past the last **complete** block of
    wearable/embedding days (e.g. 95 calendar days → 90 for month counting).
    """
    c = validate_clock(clock) if "t_evento" not in clock.columns else clock.copy()
    c = c.copy()
    c["t_evento_audit"] = c["t_evento"].astype(float)

    day_path = Path(day_csv) if day_csv else default_day_table_path(root)
    if not day_path.exists():
        raise FileNotFoundError(
            f"Day table required to cap incomplete months: {day_path}"
        )
    lengths = sequence_lengths_from_day_table(day_path)
    c = c.merge(lengths, on="id", how="left")
    missing = c["n_days"].isna()
    if missing.any():
        # patients in audit without day sequence: keep audit t_evento
        c.loc[missing, "n_days"] = c.loc[missing, "t_evento_audit"]
    c["n_days"] = c["n_days"].astype(int)
    c["n_complete_months"] = (c["n_days"] // window_size).astype(int)
    c["t_complete_months"] = (c["n_complete_months"] * window_size).astype(float)
    c["t_evento"] = c[["t_evento_audit", "t_complete_months"]].min(axis=1)
    c["days_capped"] = (c["t_evento_audit"] - c["t_evento"]).clip(lower=0)
    return c


def write_clock_bbdd(
    clock: pd.DataFrame,
    out_path: Path | str | None = None,
    root: Path | None = None,
) -> Path:
    """Persist canonical clock (with ``t_evento``) to xlsx."""
    out = Path(out_path) if out_path else default_clock_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = [
        c
        for c in (
            "t_evento",
            "t_evento_audit",
            "n_days",
            "n_complete_months",
            "t_complete_months",
            "days_capped",
        )
        if c in clock.columns
    ]
    # de-dupe while keeping order
    cols = list(dict.fromkeys(list(CANONICAL_COLS) + extra))
    clock[cols].to_excel(out, index=False)
    return out


def clock_to_clinical(clock: pd.DataFrame) -> pd.DataFrame:
    """Shape clock as clinical-like frame for extract_monthly_puf_from_master."""
    c = validate_clock(clock) if "t_evento" not in clock.columns else clock.copy()
    return pd.DataFrame(
        {
            "id": c["id"].astype(int),
            "PD_event": c["evento"].astype(int),
            "Obs_time": c["t_evento"].astype(float),
        }
    )


def apply_clock(df: pd.DataFrame, clock: pd.DataFrame) -> pd.DataFrame:
    """Overwrite Evento PD / t_evento_eb2 / follow_up_days from clock BBDD only.

    Rows whose ``id`` is not in the clock are dropped.
    """
    c = validate_clock(clock) if "t_evento" not in clock.columns else clock.copy()
    amap = c.set_index("id")
    if "id" not in df.columns:
        raise ValueError("pUF / landmark frame must have an 'id' column")
    out = df.copy()
    out["id"] = out["id"].astype(int)
    out = out[out["id"].isin(amap.index)].copy()
    out["Evento PD"] = out["id"].map(amap["evento"]).astype(int)
    out["t_evento_eb2"] = out["id"].map(amap["t_evento"]).astype(float)
    out["follow_up_days"] = out["id"].map(amap["t_evento"]).astype(float)
    out["clock_source"] = "user_bbdd"
    return out


def strip_master_clock_fields(master: pd.DataFrame) -> pd.DataFrame:
    """Drop leftover event/FU columns so they cannot override the user clock."""
    drop = []
    for c in master.columns:
        cl = str(c).strip().lower()
        if cl in {"pd_event", "obs_time", "t_evento_eb2", "evento pd", "event eb2"}:
            drop.append(c)
        elif cl.startswith("event"):
            drop.append(c)
    return master.drop(columns=drop, errors="ignore")


def write_provenance(
    out_dir: Path,
    *,
    clock_path: Path,
    clock: pd.DataFrame,
    run_kind: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``clock_used.csv`` + ``PROVENANCE.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = validate_clock(clock) if "t_evento" not in clock.columns else clock.copy()
    clock_csv = out_dir / "clock_used.csv"
    c.to_csv(clock_csv, index=False)

    has_cap = "t_evento_audit" in c.columns
    lines = [
        f"# Alarm provenance — {run_kind}",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Clock source: **user BBDD only** (`{Path(clock_path).resolve()}`)",
        "- Audit dates: `fecha_inicio` / `fecha_fin` / `evento`",
        "- Rule: `t_evento = (fecha_fin − fecha_inicio).days` (paper-style; uncapped)",
        "- If `evento=1`, `fecha_fin` is progression date; if `0`, censor at `fecha_fin`",
        "",
        "## Clock summary",
        "",
        f"- n patients: {len(c)}",
        f"- n events (evento=1): {int((c['evento'] == 1).sum())}",
        f"- n censored (evento=0): {int((c['evento'] == 0).sum())}",
        f"- t_evento days: min={c['t_evento'].min():.0f}, "
        f"median={c['t_evento'].median():.0f}, max={c['t_evento'].max():.0f}",
        "",
        "## Columns",
        "",
        "- `id`, `fecha_inicio`, `fecha_fin`, `evento` → `t_evento`",
        "",
    ]
    if has_cap and "t_evento_audit" in c.columns and (c["t_evento"] < c["t_evento_audit"]).any():
        n_cap = int((c["t_evento"] < c["t_evento_audit"]).sum())
        lines.insert(
            -3,
            f"- Note: t_evento was capped to complete months for {n_cap} patients",
        )
    if extra:
        lines.append("## Run parameters")
        lines.append("")
        for k, v in extra.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    md = out_dir / "PROVENANCE.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def build_default_clock_from_audit(
    root: Path | None = None,
    out_path: Path | None = None,
    *,
    day_csv: Path | None = None,
    cap_incomplete_months: bool = False,
) -> Path:
    """One-shot: audit DB1 → ``data/revision/clock_bbdd.xlsx``.

    Default: paper-style clock ``t_evento = fecha_fin − fecha_inicio`` (no cap).
    Pass ``cap_incomplete_months=True`` to floor analysis time to complete
    30-day months of the day sequence.
    """
    clock = adapt_audit_db1_to_clock(root=root)
    if cap_incomplete_months:
        clock = cap_clock_to_complete_months(clock, day_csv=day_csv, root=root)
    return write_clock_bbdd(clock, out_path=out_path, root=root)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build / validate universal alarm clock BBDD")
    p.add_argument(
        "--from-audit-db1",
        action="store_true",
        help="Build clock_bbdd.xlsx from audit Evento_DB1 dates",
    )
    p.add_argument("--audit", type=Path, default=None, help="Override audit xlsx path")
    p.add_argument("--out", type=Path, default=None, help="Output clock path")
    p.add_argument(
        "--day-csv",
        type=Path,
        default=None,
        help="Day/embeddings table for complete-month capping",
    )
    p.add_argument(
        "--cap-incomplete-months",
        action="store_true",
        help="Floor analysis t_evento to complete months of day data (not paper default)",
    )
    p.add_argument("--validate", type=Path, default=None, help="Validate an existing clock file")
    args = p.parse_args()

    root = project_root_from()
    if args.validate:
        c = load_clock(args.validate)
        print(f"OK: {args.validate} — n={len(c)}, events={(c['evento']==1).sum()}")
    elif args.from_audit_db1:
        clock = adapt_audit_db1_to_clock(args.audit, root=root)
        if args.cap_incomplete_months:
            clock = cap_clock_to_complete_months(clock, day_csv=args.day_csv, root=root)
            n_cap = int((clock["t_evento"] < clock["t_evento_audit"]).sum())
            print(
                f"Capped incomplete months: {n_cap}/{len(clock)} patients "
                f"(t_evento = min(audit, complete_months*30))"
            )
        else:
            print("Paper-style clock: t_evento = fecha_fin − fecha_inicio (uncapped)")
        out = write_clock_bbdd(clock, out_path=args.out, root=root)
        print(f"Wrote {out} (n={len(clock)}, events={(clock['evento']==1).sum()})")
    else:
        p.error("Use --from-audit-db1 or --validate PATH")

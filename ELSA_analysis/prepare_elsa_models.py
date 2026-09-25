"""Prepare preprocessed ELSA data for the models in elsa_model_definitions.py.
 
Expected input
--------------
`file_elsa` is the path to the already-preprocessed ELSA table produced by the
R pipeline discussed in this project. In particular, it should contain:
 
    idauniq, yob, entry_age, obs_start, obs_end, exit_dis, exit_death,
    age_cvd, age_stroke, age_hypertension, age_diabetes,
    age_asthma_respir, age_cancer, age_arthritis, age_osteoporosis,
    age_hip_replacement, age_mental_health, age_neuro, age_cataract,
    age_died,
    sex, edu, wealth, bmi (or a compatible BMI column),
    r2mbmi / r4mbmi / r6mbmi / r8mbmi and date1..dateW  (optional, for
    time-varying BMI)
 
The module creates one canonical wide table and adapters for:
  1. separate Cox PH models;
  2. MultiCoxNN;
  3. simple discrete-time multi-binary model;
  4. causal Transformer time-series model.
 
Important conventions
---------------------
* Disease outcomes are first-occurrence outcomes.
* A condition diagnosed before/at study entry is PREVALENT: it is not counted
  as an incident outcome and its disease-specific risk mask is zero from entry.
* osteo_hip = first of osteoporosis / hip replacement.
* death is an absorbing terminal outcome. No interval after death is at risk
  for any outcome.
* Disease follow-up ends at exit_dis. Death follow-up ends at exit_death, so
  death can be observed after the last disease-response wave.
* Time is years since entry_age. Discrete time defaults to 2-year bins.
* In the LONG table `age` and `bmi` are TIME-VARYING, matching the simulator,
  where both advance at every step. The wide (Cox) table keeps baseline values,
  which is where a baseline covariate belongs.
"""
 
from __future__ import annotations
 
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
 
import numpy as np
import pandas as pd
import torch
 
 
EVENT_TYPES: List[str] = [
    "cvd",
    "stroke",
    "hypertension",
    "diabetes",
    "asthma_respir",
    "cancer",
    "arthritis",
    "osteo_hip",
    "mental_health",
    "neuro",
    "cataract",
    "death",
]
 
DISEASE_EVENTS: List[str] = [e for e in EVENT_TYPES if e != "death"]
COVARIATE_COLS: List[str] = ["age", "bmi", "sex", "edu", "wealth"]
 
PREVALENCE_COLS: List[str] = [f"prevalent_{e}" for e in DISEASE_EVENTS]
BINARY_COVARIATE_COLS: List[str] = COVARIATE_COLS + PREVALENCE_COLS
HISTORY_COLS: List[str] = [f"prev_{e}" for e in DISEASE_EVENTS]
TRANSFORMER_FEATURE_COLS: List[str] = COVARIATE_COLS + HISTORY_COLS
 
# Waves carrying a nurse visit, i.e. a measured BMI.
BMI_WAVES: Tuple[int, ...] = (2, 4, 6, 8)
 
AGE_SOURCE: Dict[str, Union[str, Tuple[str, ...]]] = {
    "cvd": "age_cvd",
    "stroke": "age_stroke",
    "hypertension": "age_hypertension",
    "diabetes": "age_diabetes",
    "asthma_respir": "age_asthma_respir",
    "cancer": "age_cancer",
    "arthritis": "age_arthritis",
    "osteo_hip": ("age_osteoporosis", "age_hip_replacement"),
    "mental_health": "age_mental_health",
    "neuro": "age_neuro",
    "cataract": "age_cataract",
    "death": "age_died",
}
 
 
def _read_table(file_elsa: Union[str, Path]) -> pd.DataFrame:
    path = Path(file_elsa)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".feather", ".fst"}:
        return pd.read_feather(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported ELSA file type: {suffix}")
 
 
def _first_available(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns is present: {list(candidates)}")
 
 
def _row_min(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[existing].apply(pd.to_numeric, errors="coerce").min(axis=1, skipna=True)
 
 
def _event_age(df: pd.DataFrame, event: str) -> pd.Series:
    src = AGE_SOURCE[event]
    if isinstance(src, tuple):
        return _row_min(df, src)
    if src not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[src], errors="coerce")
 
 
def prepare_elsa_wide(
    file_elsa: Union[str, Path, pd.DataFrame],
    *,
    bmi_col: Optional[str] = None,
    drop_missing_covariates: bool = True,
    keep_bmi_waves: bool = True,
) -> pd.DataFrame:
    """Create the canonical one-row-per-person ELSA table.
 
    Output contains the five agreed covariates plus, for every outcome e:
        prevalent_e : present at/before entry
        event_e     : incident event observed during its follow-up window
        time_e      : years from entry to event or censoring
        age_e       : consolidated event age (useful for checking)
 
    For disease outcomes censoring is at exit_dis. For death it is exit_death.
 
    BMI column priority puts the EARLIEST nurse measurement first. `bmi_latest`
    is measured at the LAST nurse visit, so for most people it postdates their
    events: used as a baseline covariate it is a look-ahead leak that inflates
    discrimination. It is kept only as a last resort.
 
    With keep_bmi_waves=True the per-wave nurse BMIs are carried through as
    bmi_w{W} together with bmi_t{W} (years from entry to that measurement), so
    make_discrete_long can build a TIME-VARYING BMI.
    """
    raw = file_elsa.copy() if isinstance(file_elsa, pd.DataFrame) else _read_table(file_elsa)
 
    required = ["idauniq", "entry_age", "exit_dis", "exit_death", "sex", "edu", "wealth"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(f"ELSA input is missing required columns: {missing}")
 
    if bmi_col is None:
        # earliest-measured first; bmi_latest last (look-ahead leak)
        bmi_col = _first_available(
            raw, ["bmi", "r2mbmi", "r4mbmi", "r6mbmi", "r8mbmi", "bmi_latest"]
        )
        if bmi_col == "bmi_latest":
            print("WARNING: falling back to bmi_latest, measured at the LAST nurse "
                  "visit. For most people this POSTDATES their events and will "
                  "inflate discrimination. Prefer an earliest-visit BMI.")
 
    out = pd.DataFrame(index=raw.index)
    out["id"] = raw["idauniq"]
    out["age"] = pd.to_numeric(raw["entry_age"], errors="coerce")
    out["bmi"] = pd.to_numeric(raw[bmi_col], errors="coerce")
    out["sex"] = pd.to_numeric(raw["sex"], errors="coerce")
    out["edu"] = pd.to_numeric(raw["edu"], errors="coerce")
    out["wealth"] = pd.to_numeric(raw["wealth"], errors="coerce")
 
    out["exit_dis"] = pd.to_numeric(raw["exit_dis"], errors="coerce")
    out["exit_death"] = pd.to_numeric(raw["exit_death"], errors="coerce")
 
    entry = out["age"]
 
    # --- per-wave nurse BMI, timed in years since entry -------------------
    if keep_bmi_waves and "yob" in raw.columns:
        yob = pd.to_numeric(raw["yob"], errors="coerce")
        kept = []
        for w in BMI_WAVES:
            vcol, dcol = f"r{w}mbmi", f"date{w}"
            if vcol in raw.columns and dcol in raw.columns:
                out[f"bmi_w{w}"] = pd.to_numeric(raw[vcol], errors="coerce")
                out[f"bmi_t{w}"] = pd.to_numeric(raw[dcol], errors="coerce") - yob - entry
                kept.append(w)
        if kept:
            print(f"time-varying BMI available from waves {kept}")
 
    for event in EVENT_TYPES:
        age_evt = _event_age(raw, event)
        censor_age = out["exit_death"] if event == "death" else out["exit_dis"]
 
        # Event at/before entry = prevalent, not an incident event.
        prevalent = age_evt.notna() & entry.notna() & (age_evt <= entry)
        incident = (
            age_evt.notna()
            & entry.notna()
            & censor_age.notna()
            & (age_evt > entry)
            & (age_evt <= censor_age)
        )
 
        followup = (censor_age - entry).clip(lower=0)
        event_time = (age_evt - entry).clip(lower=0)
        time = followup.where(~incident, event_time)
 
        out[f"age_{event}"] = age_evt
        out[f"prevalent_{event}"] = prevalent.astype(np.int8)
        out[f"event_{event}"] = incident.astype(np.int8)
        out[f"time_{event}"] = time.astype(float)
 
    # A person dead at/before entry cannot contribute prospective follow-up.
    out = out.loc[out["prevalent_death"] == 0].copy()
 
    if drop_missing_covariates:
        out = out.dropna(subset=COVARIATE_COLS)
 
    # Cox code cannot use missing follow-up times. Disease and death follow-up
    # may differ, so require the relevant censoring ages to exist.
    out = out.dropna(subset=["age", "exit_dis", "exit_death"])
    out = out.reset_index(drop=True)
    return out
 
 
# ---------------------------------------------------------------------------
# 1) Separate Cox PH
# ---------------------------------------------------------------------------
def make_separate_coxph(
    wide: pd.DataFrame,
    *,
    covariates: Sequence[str] = COVARIATE_COLS,
    event_types: Sequence[str] = EVENT_TYPES,
    as_torch: bool = False,
):
    """Return one Cox-ready dataset per outcome.
 
    Prevalent cases of a disease are removed from that disease's risk set.
    Death has no prevalent cases after prepare_elsa_wide filtering.
 
    If as_torch=False:
        dict[event] -> DataFrame(id, covariates..., time, event)
    If as_torch=True:
        dict[event] -> (x, time, event).
    """
    ans = {}
    for e in event_types:
        d = wide.loc[wide[f"prevalent_{e}"] == 0, ["id", *covariates, f"time_{e}", f"event_{e}"]].copy()
        d = d.dropna(subset=[*covariates, f"time_{e}", f"event_{e}"])
        d = d.rename(columns={f"time_{e}": "time", f"event_{e}": "event"})
        if as_torch:
            ans[e] = (
                torch.tensor(d[list(covariates)].to_numpy(dtype=np.float32), dtype=torch.float32),
                torch.tensor(d["time"].to_numpy(dtype=np.float32), dtype=torch.float32),
                torch.tensor(d["event"].to_numpy(dtype=np.float32), dtype=torch.float32),
            )
        else:
            ans[e] = d.reset_index(drop=True)
    return ans
 
 
# ---------------------------------------------------------------------------
# 2) Multi Cox NN
# ---------------------------------------------------------------------------
def make_multicox_nn(
    wide: pd.DataFrame,
    *,
    covariates: Sequence[str] = COVARIATE_COLS,
    event_types: Sequence[str] = EVENT_TYPES,
    prevalent_policy: str = "keep_with_mask",
    as_torch: bool = True,
):
    """Prepare a rectangular table for MultiCoxNN.
 
    DEFAULT IS `keep_with_mask`, and it should stay that way.
 
    `drop_any` retains only people who are incident-risk for EVERY outcome. In
    ELSA hypertension is ~31% prevalent at entry and arthritis ~29%, so that
    rule discards the majority of the cohort and the remainder is heavily
    selected towards the healthy -- which biases every downstream comparison.
 
    It is not needed: cox_partial_loss_masked() in elsa_model_definitions.py
    takes an outcome-specific eligibility mask, so a person prevalent for
    arthritis still contributes to the diabetes risk set. fit_multicox_nn()
    builds that mask itself from the prevalent_* columns, so the simplest and
    safest call is to pass the full `wide` table straight to it.
    """
    event_types = list(event_types)
    prev_cols = [f"prevalent_{e}" for e in event_types]
 
    d = wide.copy()
    if prevalent_policy == "drop_any":
        n_before = len(d)
        d = d.loc[d[prev_cols].sum(axis=1) == 0].copy()
        print(f"WARNING: prevalent_policy='drop_any' kept {len(d)}/{n_before} "
              f"({100*len(d)/max(n_before,1):.0f}%) of the cohort, selected "
              f"towards people free of ALL {len(event_types)} conditions at "
              f"entry. Use 'keep_with_mask' unless you specifically want this.")
    elif prevalent_policy != "keep_with_mask":
        raise ValueError("prevalent_policy must be 'keep_with_mask' or 'drop_any'")
 
    time_cols = [f"time_{e}" for e in event_types]
    event_cols = [f"event_{e}" for e in event_types]
    d = d.dropna(subset=[*covariates, *time_cols, *event_cols]).reset_index(drop=True)
 
    if not as_torch:
        return d[["id", *covariates, *time_cols, *event_cols, *prev_cols]].copy()
 
    x = torch.tensor(d[list(covariates)].to_numpy(dtype=np.float32), dtype=torch.float32)
    times = torch.tensor(d[time_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    events = torch.tensor(d[event_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    risk_mask = torch.tensor((1 - d[prev_cols]).to_numpy(dtype=np.float32), dtype=torch.float32)
 
    if prevalent_policy == "keep_with_mask":
        return x, times, events, risk_mask, d["id"].to_numpy()
    return x, times, events
 
 
# ---------------------------------------------------------------------------
# Shared discrete-time long table
# ---------------------------------------------------------------------------
def _timevarying_bmi(wide: pd.DataFrame, start: float, backfill: bool = True):
    """Nurse-visit BMI carried forward to interval `start` (years since entry).
 
    The ELSA analogue of the simulator's drifting BMI, and the only genuinely
    time-varying covariate this dataset supports. Falls back to the baseline
    `bmi` column where no measurement is available.
    """
    waves = [w for w in BMI_WAVES
             if f"bmi_w{w}" in wide.columns and f"bmi_t{w}" in wide.columns]
    cur = wide["bmi"].to_numpy(dtype=float).copy()
    if not waves:
        return cur
 
    got = np.zeros(len(wide), dtype=bool)
    first_val = np.full(len(wide), np.nan)
    for w in sorted(waves):                       # ascending -> carry forward
        v = wide[f"bmi_w{w}"].to_numpy(dtype=float)
        t = wide[f"bmi_t{w}"].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(t) & (t <= start)
        cur[ok] = v[ok]
        got |= ok
        newly = np.isnan(first_val) & np.isfinite(v)
        first_val[newly] = v[newly]
 
    if backfill:                                  # before the first measurement
        use = (~got) & np.isfinite(first_val)
        cur[use] = first_val[use]
    return cur
 
 
def make_discrete_long(
    wide: pd.DataFrame,
    *,
    interval_years: float = 2.0,
    event_types: Sequence[str] = EVENT_TYPES,
    covariates: Sequence[str] = COVARIATE_COLS,
    time_varying: bool = True,
    bmi_backfill: bool = True,
) -> pd.DataFrame:
    """Create a rectangular person x interval table for binary/Transformer models.
 
    Rectangular sequences are deliberate: the Transformer requires every patient
    to have the same T. Rows after a person's censoring or death remain as
    padding-like rows but have all masks set to zero.
 
    `prev_e` is history available BEFORE the current interval. For a prevalent
    condition it is 1 from interval 0. For an incident condition it switches to
    1 only in the interval AFTER the first event.
 
    With time_varying=True, `age` and `bmi` ADVANCE with the interval -- age by
    `start` years, BMI by carrying forward the most recent nurse measurement.
    This matches the simulator, where both update at every step; freezing them
    at baseline would leave the sequence models with no moving covariate at all.
    Baseline values are retained as `age_baseline` / `bmi_baseline`.
    """
    if interval_years <= 0:
        raise ValueError("interval_years must be > 0")
 
    event_types = list(event_types)
    max_fu = np.nanmax((wide["exit_death"] - wide["age"]).to_numpy(dtype=float))
    if not np.isfinite(max_fu) or max_fu <= 0:
        raise ValueError("No positive follow-up found")
    T = int(np.ceil(max_fu / interval_years))
 
    bmi_cols = [c for c in wide.columns if c.startswith(("bmi_w", "bmi_t"))]
 
    blocks = []
    for t in range(T):
        start = t * interval_years
        end = (t + 1) * interval_years
        base_cols = list(dict.fromkeys(
            ["id", *covariates, "age", "bmi", "exit_dis", "exit_death", *bmi_cols]
        ))
        base_cols = [c for c in base_cols if c in wide.columns]
        b = wide[base_cols].copy()
        b["interval"] = t
        b["start"] = start
        b["end"] = end
 
        b["age_baseline"] = wide["age"].to_numpy()
        b["bmi_baseline"] = wide["bmi"].to_numpy()
        if time_varying:
            b["age"] = b["age_baseline"] + start
            b["bmi"] = _timevarying_bmi(wide, start, backfill=bmi_backfill)
        b["age_current"] = b["age_baseline"] + start      # kept for reference
 
        death_time = wide["time_death"].to_numpy(dtype=float)
        death_event = wide["event_death"].to_numpy(dtype=int)
        alive_at_start = (death_event == 0) | (death_time >= start)
 
        for e in event_types:
            evt = wide[f"event_{e}"].to_numpy(dtype=int)
            tm = wide[f"time_{e}"].to_numpy(dtype=float)
            prev0 = wide[f"prevalent_{e}"].to_numpy(dtype=int)
            censor_tm = (
                wide["exit_death"].to_numpy(dtype=float) - wide["age"].to_numpy(dtype=float)
                if e == "death"
                else wide["exit_dis"].to_numpy(dtype=float) - wide["age"].to_numpy(dtype=float)
            )
 
            # History strictly before this interval: no same-interval leakage.
            prior_incident = (evt == 1) & (tm < start)
            prev = ((prev0 == 1) | prior_incident).astype(np.int8)
 
            # Event belongs to [start, end).
            y = ((evt == 1) & (tm >= start) & (tm < end)).astype(np.int8)
 
            # At risk at interval start, with observable follow-up in this interval.
            mask = ((prev == 0) & (censor_tm > start) & alive_at_start).astype(np.int8)
 
            b[f"event_{e}"] = y
            b[f"prev_{e}"] = prev
            b[f"mask_{e}"] = mask
 
        blocks.append(b)
 
    long = pd.concat(blocks, ignore_index=True)
    long = long.sort_values(["id", "interval"]).reset_index(drop=True)
 
    # Enforce absorbing death after its event interval.
    if "death" in event_types:
        death_seen_before = long.groupby("id", sort=False)["event_death"].transform(
            lambda s: s.cumsum().shift(1, fill_value=0)
        ).astype(bool)
        for e in event_types:
            long.loc[death_seen_before, f"mask_{e}"] = 0
        long["prev_death"] = long.groupby("id", sort=False)["event_death"].transform(
            lambda s: s.cummax().shift(1, fill_value=0)
        ).astype(np.int8)
 
    # an event must never fire outside its own risk set
    for e in event_types:
        bad = int(((long[f"event_{e}"] == 1) & (long[f"mask_{e}"] == 0)).sum())
        if bad:
            print(f"WARNING: {e}: {bad} events fire outside the risk set")
 
    return long
 
 
# ---------------------------------------------------------------------------
# 3) Simple binary time series
# ---------------------------------------------------------------------------
def make_simple_binary_timeseries(
    wide: pd.DataFrame,
    *,
    interval_years: float = 2.0,
    covariates: Sequence[str] = BINARY_COVARIATE_COLS,
    event_types: Sequence[str] = EVENT_TYPES,
    include_history: bool = False,
    time_varying: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return df_long and the feature list for fit_simple_binary().
 
    By default this model receives baseline characteristics plus fixed
    prevalent-disease indicators. It does not receive evolving prev_* history
    unless include_history=True.
    """
    long = make_discrete_long(
        wide, interval_years=interval_years, event_types=event_types,
        covariates=covariates, time_varying=time_varying,
    )
    features = list(covariates)
    if include_history:
        # death is terminal, so prev_death has no unmasked future row to predict.
        features += [f"prev_{e}" for e in event_types if e != "death"]
    return long, features
 
 
# ---------------------------------------------------------------------------
# 4) Transformer
# ---------------------------------------------------------------------------
def make_transformer(
    wide: pd.DataFrame,
    *,
    interval_years: float = 2.0,
    covariates: Sequence[str] = COVARIATE_COLS,
    event_types: Sequence[str] = EVENT_TYPES,
    include_prev_features: bool = True,
    time_varying: bool = True,
    as_torch: bool = False,
):
    """Prepare the rectangular causal sequence used by the Transformer.
 
    The input at interval t contains covariates AT t (age and BMI advance) plus
    the accumulated disease history available strictly before t.
    """
    long = make_discrete_long(
        wide, interval_years=interval_years, event_types=event_types,
        covariates=covariates, time_varying=time_varying,
    )
    features = list(covariates)
    if include_prev_features:
        features += [f"prev_{e}" for e in event_types if e != "death"]
 
    if not as_torch:
        return long, features
 
    df = long.sort_values(["id", "interval"]).reset_index(drop=True)
    ids = df["id"].drop_duplicates().to_numpy()
    n_pat = len(ids)
    T = df["interval"].nunique()
    K = len(event_types)
 
    if len(df) != n_pat * T:
        raise ValueError("Transformer table is not rectangular")
 
    event_cols = [f"event_{e}" for e in event_types]
    mask_cols = [f"mask_{e}" for e in event_types]
    X = torch.tensor(df[features].to_numpy(dtype=np.float32), dtype=torch.float32).view(n_pat, T, len(features))
    events = torch.tensor(df[event_cols].to_numpy(dtype=np.float32), dtype=torch.float32).view(n_pat, T, K)
    masks = torch.tensor(df[mask_cols].to_numpy(dtype=np.float32), dtype=torch.float32).view(n_pat, T, K)
    return X, events, masks, ids, T, features
 
 
def prepare_all(file_elsa: Union[str, Path, pd.DataFrame], *, interval_years: float = 2.0):
    """Convenience wrapper returning all four model inputs."""
    wide = prepare_elsa_wide(file_elsa)
    return {
        "wide": wide,
        "separate_coxph": make_separate_coxph(wide),
        # keep_with_mask: fit_multicox_nn builds the outcome-specific mask itself
        "multicox_nn": make_multicox_nn(wide, prevalent_policy="keep_with_mask"),
        "simple_binary": make_simple_binary_timeseries(wide, interval_years=interval_years),
        "transformer": make_transformer(wide, interval_years=interval_years),
    }
 
 
if __name__ == "__main__":
    # Example:
    # prepared = prepare_all("elsa_data.csv", interval_years=2.0)
    # print(prepared["wide"].shape)
    pass
 


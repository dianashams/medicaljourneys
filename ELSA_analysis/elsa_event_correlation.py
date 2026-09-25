"""
ELSA multimorbidity / event-pair correlation utilities.

Designed for the model objects produced by:
    elsa_model_definitions.py
and the wide/long data produced by:
    prepare_elsa_models.py

The analysis mirrors the simulation multimorbidity figure:
  1) choose a landmark cohort that is free of the selected non-death outcomes;
  2) calculate observed event-pair phi correlations over a future horizon;
  3) simulate repeated joint trajectories from MultiCox, Simple Binary TS,
     and the causal Transformer;
  4) average the simulated population-level phi correlations;
  5) plot one heatmap per source.

Important:
- The Simple TS rollout keeps its baseline prevalent_* covariates fixed.
- The Transformer rollout updates prev_* after simulated events.
- Death is terminal: after a simulated death, later disease events are impossible.
- With 2-year intervals and a 5-year horizon, the last interval is only half used.
  For discrete-time models, its hazard is converted assuming constant hazard
  within that interval: h_partial = 1 - (1-h)**fraction.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


DEFAULT_CORR_EVENTS = [
    "death", "cancer", "cvd", "diabetes", "cataract", "mental_health"
]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def phi_binary(x, y):
    """Phi coefficient = Pearson correlation for two binary variables."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_phi(df, events=DEFAULT_CORR_EVENTS):
    rows = []
    for a, b in combinations(events, 2):
        rows.append({
            "pair": (a, b),
            "event_a": a,
            "event_b": b,
            "phi": phi_binary(df[a], df[b]),
            "n": len(df),
        })
    return pd.DataFrame(rows)


def _elapsed_exit(wide, col):
    """exit_* is stored as attained age; convert to years since entry."""
    return wide[col].to_numpy(float) - wide["age"].to_numpy(float)


def _landmark_cohort_ids(wide, events, horizon):
    """
    T0=0 cohort used for all panels.

    Mirrors the simulation restriction to people event-free for the selected
    non-death outcomes at the landmark. People who die within the horizon are
    retained (death is a valid outcome); non-death dropout before the horizon
    is excluded.
    """
    keep = np.ones(len(wide), dtype=bool)

    for e in events:
        if e != "death":
            keep &= wide[f"prevalent_{e}"].to_numpy(int) == 0

    # Disease observation must continue to horizon OR terminate because of death.
    dis_fu = _elapsed_exit(wide, "exit_dis")
    death_event = wide["event_death"].to_numpy(int) == 1
    death_time = wide["time_death"].to_numpy(float)
    death_within = death_event & np.isfinite(death_time) & (death_time <= horizon)

    complete_disease_followup = (dis_fu >= horizon) | (
        death_within & (dis_fu + 1e-8 >= death_time)
    )

    # Mortality itself must be observable through the horizon unless death occurs.
    death_fu = _elapsed_exit(wide, "exit_death")
    complete_death_followup = (death_fu >= horizon) | death_within

    keep &= complete_disease_followup & complete_death_followup
    return wide.loc[keep, "id"].to_numpy()


def observed_events_at_horizon(
    wide,
    events=DEFAULT_CORR_EVENTS,
    horizon=5.0,
    ids=None,
):
    """Observed 0/1 incidence by horizon in the common landmark cohort."""
    if ids is None:
        ids = _landmark_cohort_ids(wide, events, horizon)

    d = wide[wide["id"].isin(ids)].copy()
    out = pd.DataFrame({"id": d["id"].to_numpy()})

    for e in events:
        out[e] = (
            (d[f"event_{e}"].to_numpy(int) == 1)
            & (d[f"time_{e}"].to_numpy(float) <= horizon)
        ).astype(np.int8)

    return out


def _aggregate_simulated_phi(sim_df, events, source):
    pieces = []
    for s, d in sim_df.groupby("sim", sort=True):
        p = pairwise_phi(d, events)
        p["sim"] = s
        pieces.append(p)

    raw = pd.concat(pieces, ignore_index=True)
    ans = (
        raw.groupby(["event_a", "event_b"], as_index=False)
        .agg(
            phi=("phi", "mean"),
            phi_sd=("phi", "std"),
            n=("n", "first"),
        )
    )
    ans["pair"] = list(zip(ans["event_a"], ans["event_b"]))
    ans["source"] = source
    return ans, raw


def _partial_interval_hazard(h, fraction):
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0 - 1e-10)
    if fraction >= 1:
        return h
    if fraction <= 0:
        return np.zeros_like(h)
    return 1.0 - np.power(1.0 - h, fraction)


# ---------------------------------------------------------------------
# MultiCox: Breslow baseline + simulated event times
# ---------------------------------------------------------------------

def fit_breslow_baselines(
    multicox_model,
    multicox_scaler,
    train_wide,
    covariates,
    event_types,
):
    """
    Estimate one Breslow baseline cumulative hazard per MultiCox output.
    Prevalent cases are excluded outcome-by-outcome, matching model fitting.
    """
    multicox_model.eval()
    x = torch.tensor(
        multicox_scaler.transform(
            train_wide[list(covariates)].to_numpy(dtype=np.float32)
        ),
        dtype=torch.float32,
    )
    with torch.no_grad():
        eta = multicox_model(x).cpu().numpy()

    baselines = {}
    for k, e in enumerate(event_types):
        eligible = train_wide[f"prevalent_{e}"].to_numpy(int) == 0
        t = train_wide.loc[eligible, f"time_{e}"].to_numpy(float)
        y = train_wide.loc[eligible, f"event_{e}"].to_numpy(int)
        r = np.exp(np.clip(eta[eligible, k], -30, 30))

        event_times = np.sort(np.unique(t[y == 1]))
        increments = []
        for tj in event_times:
            d_j = np.sum((t == tj) & (y == 1))
            risk_sum = r[t >= tj].sum()
            increments.append(d_j / risk_sum if risk_sum > 0 else 0.0)

        baselines[e] = pd.DataFrame({
            "time": event_times,
            "dH0": np.asarray(increments, dtype=float),
            "H0": np.cumsum(increments),
        })

    return baselines


def _sample_cox_time(eta, baseline, rng):
    """
    Inverse-transform sample from S(t|x)=exp[-H0(t) exp(eta)].
    Returns inf if sampled cumulative hazard exceeds observed baseline support.
    """
    if baseline.empty:
        return np.inf

    target = -np.log(rng.random()) / np.exp(np.clip(eta, -30, 30))
    H0 = baseline["H0"].to_numpy(float)
    j = np.searchsorted(H0, target, side="left")
    if j >= len(H0):
        return np.inf
    return float(baseline["time"].iloc[j])


def simulate_multicox(
    multicox_model,
    multicox_scaler,
    baselines,
    wide,
    covariates,
    event_types,
    corr_events=DEFAULT_CORR_EVENTS,
    ids=None,
    horizon=5.0,
    n_sim=100,
    seed=42,
):
    """
    Simulate event times from fitted MultiCox marginal Cox models.

    Dependence can arise through shared patient covariates / shared NN
    representation. There is no disease-history updating. Death is imposed
    as terminal: a disease sampled after death is not counted.
    """
    if ids is None:
        ids = _landmark_cohort_ids(wide, corr_events, horizon)

    d = wide[wide["id"].isin(ids)].copy().reset_index(drop=True)

    x = torch.tensor(
        multicox_scaler.transform(d[list(covariates)].to_numpy(dtype=np.float32)),
        dtype=torch.float32,
    )
    multicox_model.eval()
    with torch.no_grad():
        eta = multicox_model(x).cpu().numpy()

    eidx = {e: k for k, e in enumerate(event_types)}
    rng = np.random.default_rng(seed)
    sims = []

    for s in range(n_sim):
        out = pd.DataFrame({"id": d["id"].to_numpy(), "sim": s})
        sampled = {}

        for e in corr_events:
            k = eidx[e]
            sampled[e] = np.array([
                _sample_cox_time(eta[i, k], baselines[e], rng)
                for i in range(len(d))
            ])

        death_t = sampled.get("death", np.full(len(d), np.inf))

        for e in corr_events:
            te = sampled[e]
            if e == "death":
                out[e] = (te <= horizon).astype(np.int8)
            else:
                out[e] = ((te <= horizon) & (te <= death_t)).astype(np.int8)

        sims.append(out)

    return pd.concat(sims, ignore_index=True)


# ---------------------------------------------------------------------
# Simple binary time-series rollout
# ---------------------------------------------------------------------

def simulate_simple_binary(
    binary_model,
    binary_scaler,
    long_df,
    features,
    event_types,
    corr_events=DEFAULT_CORR_EVENTS,
    ids=None,
    horizon=5.0,
    interval_years=2.0,
    n_sim=100,
    seed=42,
):
    """
    Monte-Carlo rollout for SimpleBinaryTimeSeries.

    Baseline covariates, including prevalent_* indicators, stay fixed.
    The model has no evolving prev_* history. Once an event occurs it is
    absorbing for that event; death stops later intervals.
    """
    df = long_df.sort_values(["id", "interval"]).copy()
    if ids is not None:
        df = df[df["id"].isin(ids)].copy()

    patient_ids = df["id"].drop_duplicates().to_numpy()
    base = df[df["interval"] == df["interval"].min()].set_index("id")
    X0 = binary_scaler.transform(
        base.loc[patient_ids, list(features)].to_numpy(dtype=np.float32)
    )
    X0 = torch.tensor(X0, dtype=torch.float32)

    eidx = {e: k for k, e in enumerate(event_types)}
    max_needed = int(np.ceil(horizon / interval_years))
    max_available = binary_model.n_intervals
    n_steps = min(max_needed, max_available)

    rng = np.random.default_rng(seed)
    sims = []

    binary_model.eval()
    with torch.no_grad():
        eta = binary_model.get_eta(X0).cpu().numpy()
        alpha = binary_model.alpha.detach().cpu().numpy()

    for s in range(n_sim):
        occurred = {e: np.zeros(len(patient_ids), dtype=bool) for e in corr_events}
        dead = np.zeros(len(patient_ids), dtype=bool)

        for t in range(n_steps):
            start = t * interval_years
            remaining = horizon - start
            if remaining <= 0:
                break
            frac = min(1.0, remaining / interval_years)

            logits = eta + alpha[:, t][None, :]
            hazards = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            hazards = _partial_interval_hazard(hazards, frac)

            # Sample all selected outcomes within this interval. Co-occurrence
            # with death in the same interval is allowed because ordering is
            # unresolved at the 2-year resolution; death blocks later intervals.
            newly = {}
            for e in corr_events:
                k = eidx[e]
                at_risk = (~occurred[e]) & (~dead)
                newly[e] = at_risk & (rng.random(len(patient_ids)) < hazards[:, k])

            for e in corr_events:
                occurred[e] |= newly[e]
            if "death" in corr_events:
                dead |= newly["death"]

        out = pd.DataFrame({"id": patient_ids, "sim": s})
        for e in corr_events:
            out[e] = occurred[e].astype(np.int8)
        sims.append(out)

    return pd.concat(sims, ignore_index=True)


# ---------------------------------------------------------------------
# Transformer sequential rollout
# ---------------------------------------------------------------------

def simulate_transformer(
    transformer_model,
    long_df,
    features,
    event_types,
    corr_events=DEFAULT_CORR_EVENTS,
    ids=None,
    horizon=5.0,
    interval_years=2.0,
    n_sim=100,
    seed=42,
):
    """
    Sequential Monte-Carlo rollout for CausalSurvivalTransformer.

    The future observed prev_* columns are NOT used. Starting from interval-0
    history, each simulated incident disease updates prev_<disease> for the
    next interval. This is the key difference from the simple binary model.
    """
    df = long_df.sort_values(["id", "interval"]).copy()
    if ids is not None:
        df = df[df["id"].isin(ids)].copy()

    patient_ids = df["id"].drop_duplicates().to_numpy()
    t0 = df[df["interval"] == df["interval"].min()].set_index("id")
    X0 = t0.loc[patient_ids, list(features)].to_numpy(dtype=np.float32)

    eidx = {e: k for k, e in enumerate(event_types)}
    fidx = {f: j for j, f in enumerate(features)}
    history_events = [
        e for e in event_types
        if e != "death" and f"prev_{e}" in fidx
    ]

    max_needed = int(np.ceil(horizon / interval_years))
    n_steps = min(max_needed, transformer_model.n_intervals)

    rng = np.random.default_rng(seed)
    sims = []
    transformer_model.eval()

    for s in range(n_sim):
        occurred = {e: np.zeros(len(patient_ids), dtype=bool) for e in corr_events}
        dead = np.zeros(len(patient_ids), dtype=bool)

        # Current disease state starts from baseline prev_*.
        state = {
            e: X0[:, fidx[f"prev_{e}"]].astype(bool).copy()
            for e in history_events
        }

        seq_rows = []

        for t in range(n_steps):
            start = t * interval_years
            remaining = horizon - start
            if remaining <= 0:
                break
            frac = min(1.0, remaining / interval_years)

            xt = X0.copy()

            # Replace prev_* with the simulated state available BEFORE interval t.
            for e in history_events:
                xt[:, fidx[f"prev_{e}"]] = state[e].astype(np.float32)

            seq_rows.append(torch.tensor(xt, dtype=torch.float32))
            Xseq = torch.stack(seq_rows, dim=1)

            with torch.no_grad():
                logits_t = transformer_model(Xseq)[:, -1, :]
                hazards = torch.sigmoid(logits_t).cpu().numpy()

            hazards = _partial_interval_hazard(hazards, frac)

            newly = {}
            for e in corr_events:
                k = eidx[e]
                at_risk = (~occurred[e]) & (~dead)
                newly[e] = at_risk & (rng.random(len(patient_ids)) < hazards[:, k])

            for e in corr_events:
                occurred[e] |= newly[e]

            # Update the complete disease-history state, including outcomes not
            # displayed in corr_events, because they can influence later hazards.
            # They must therefore also be sampled.
            for e in history_events:
                if e in newly:
                    inc = newly[e]
                else:
                    k = eidx[e]
                    at_risk = (~state[e]) & (~dead)
                    inc = at_risk & (rng.random(len(patient_ids)) < hazards[:, k])
                state[e] |= inc

            if "death" in corr_events:
                dead |= newly["death"]

        out = pd.DataFrame({"id": patient_ids, "sim": s})
        for e in corr_events:
            out[e] = occurred[e].astype(np.int8)
        sims.append(out)

    return pd.concat(sims, ignore_index=True)


# ---------------------------------------------------------------------
# One wrapper to compute everything
# ---------------------------------------------------------------------

def compute_elsa_correlation_summary(
    *,
    train_wide,
    test_wide,
    test_long_binary,
    test_long_transformer,
    multicox_model,
    multicox_scaler,
    binary_model,
    binary_scaler,
    transformer_model,
    multicox_covariates,
    binary_features,
    transformer_features,
    event_types,
    corr_events=DEFAULT_CORR_EVENTS,
    horizon=5.0,
    interval_years=2.0,
    n_sim=100,
    seed=42,
):
    """
    Return:
      summary : mean phi for the four panels
      details : useful intermediate data/simulations
    """
    corr_events = list(corr_events)
    missing = [e for e in corr_events if e not in event_types]
    if missing:
        raise ValueError(f"corr_events not in event_types: {missing}")

    ids = _landmark_cohort_ids(test_wide, corr_events, horizon)
    if len(ids) == 0:
        raise ValueError("No participants remain in the common landmark cohort.")

    obs = observed_events_at_horizon(
        test_wide, corr_events, horizon=horizon, ids=ids
    )
    obs_phi = pairwise_phi(obs, corr_events)
    obs_phi["phi_sd"] = np.nan
    obs_phi["source"] = "Observed ELSA"

    baselines = fit_breslow_baselines(
        multicox_model, multicox_scaler, train_wide,
        multicox_covariates, event_types
    )
    mc_sim = simulate_multicox(
        multicox_model, multicox_scaler, baselines, test_wide,
        multicox_covariates, event_types, corr_events, ids,
        horizon, n_sim, seed
    )
    mc_phi, mc_raw = _aggregate_simulated_phi(
        mc_sim, corr_events, "MultiCox DeepNN"
    )

    bin_sim = simulate_simple_binary(
        binary_model, binary_scaler, test_long_binary, binary_features,
        event_types, corr_events, ids, horizon, interval_years, n_sim, seed + 1
    )
    bin_phi, bin_raw = _aggregate_simulated_phi(
        bin_sim, corr_events, "Simple TS"
    )

    tr_sim = simulate_transformer(
        transformer_model, test_long_transformer, transformer_features,
        event_types, corr_events, ids, horizon, interval_years, n_sim, seed + 2
    )
    tr_phi, tr_raw = _aggregate_simulated_phi(
        tr_sim, corr_events, "Transformer"
    )

    summary = pd.concat(
        [obs_phi, mc_phi, bin_phi, tr_phi],
        ignore_index=True,
        sort=False,
    )

    details = {
        "landmark_ids": ids,
        "observed_events": obs,
        "multicox_baselines": baselines,
        "multicox_simulations": mc_sim,
        "binary_simulations": bin_sim,
        "transformer_simulations": tr_sim,
        "multicox_phi_replicates": mc_raw,
        "binary_phi_replicates": bin_raw,
        "transformer_phi_replicates": tr_raw,
    }
    return summary, details


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_elsa_correlation_heatmaps(
    summary,
    events=DEFAULT_CORR_EVENTS,
    sources=("Observed ELSA", "MultiCox DeepNN", "Simple TS", "Transformer"),
    horizon=5.0,
    figsize=None,
    cmap="RdBu_r",
):
    """
    Plot the ELSA analogue of the simulation event-pair correlation figure.
    Uses matplotlib only, so seaborn is not required.
    """
    events = list(events)
    sources = list(sources)

    vals = summary["phi"].to_numpy(float)
    finite = np.isfinite(vals)
    vmax = np.nanmax(np.abs(vals[finite])) if finite.any() else 1.0
    if vmax == 0:
        vmax = 1.0

    if figsize is None:
        figsize = (4.0 * len(sources), 4.8)

    fig, axes = plt.subplots(1, len(sources), figsize=figsize, squeeze=False)
    axes = axes.ravel()

    last_im = None
    for ax, src in zip(axes, sources):
        mat = pd.DataFrame(np.nan, index=events, columns=events, dtype=float)
        sub = summary[summary["source"] == src]

        for _, row in sub.iterrows():
            a, b = row["event_a"], row["event_b"]
            if a in mat.index and b in mat.columns:
                mat.loc[a, b] = mat.loc[b, a] = row["phi"]

        arr = mat.to_numpy(float)
        masked = np.ma.masked_invalid(arr)
        last_im = ax.imshow(
            masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal"
        )

        ax.set_xticks(range(len(events)), events, rotation=45, ha="right")
        ax.set_yticks(range(len(events)), events)
        ax.set_title(src, fontweight="bold")

        # White grid between cells
        ax.set_xticks(np.arange(-0.5, len(events), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(events), 1), minor=True)
        ax.grid(which="minor", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(len(events)):
            for j in range(len(events)):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

        n = int(sub["n"].iloc[0]) if len(sub) and np.isfinite(sub["n"].iloc[0]) else 0
        ax.set_xlabel(f"n = {n}")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.78, pad=0.02)
        cbar.set_label("φ")

    fig.suptitle(
        f"ELSA event-pair correlation over {horizon:g} years",
        fontsize=14,
        y=1.02,
    )
    fig.subplots_adjust(wspace=0.28, top=0.86, bottom=0.20)
    return fig, axes

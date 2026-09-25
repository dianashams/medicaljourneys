##############################################################################
## ELSA: the four simulation figures, reproduced on real data               ##
##############################################################################
#
#   1) calibration at a landmark, for a fixed horizon
#   2) progression of per-event discrimination with available history
#   3) pair-wise event correlations, observed vs model-simulated
#   4) delta log-odds of event k given event j present at T0  (transformer probe)
#
# Everything is landmarked: at T0 we keep people still at risk and observable,
# use only information available by T0, and evaluate over (T0, T0+H].
#
# Works with prepare_elsa_models.py / elsa_model_definitions.py.
#
# NOTE ON GROUND TRUTH. In the simulation the generative coefficients were
# known, so figure 4 could be validated. In ELSA there is no ground truth: the
# influence matrix is a PREDICTIVE TEMPORAL ASSOCIATION, adjusted for the
# measured covariates, not a causal effect. Face validity against known
# progressions (hypertension -> stroke, arthritis -> hip replacement) is the
# available check.
 
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
 
 
# ============================================================ landmark frame
 
def landmark_frame(wide, event, landmark, horizon):
    """
    Eligibility and observed outcome for one event at one landmark.
 
      elig : at risk for this event at T0 -- NOT prevalent at entry, and
             neither event nor censoring has occurred by T0
      occ  : event observed in (T0, T0+H]
      dur  : time from T0, administratively censored at H
      complete : follow-up reaches T0+H, or the event occurred
                 (used for the correlation figure, which needs a binary
                  indicator rather than a survival time)
    """
    t = wide[f"time_{event}"].to_numpy(float)
    e = wide[f"event_{event}"].to_numpy(int)
    prev = wide[f"prevalent_{event}"].to_numpy(int)
 
    elig = (prev == 0) & (t > landmark)
 
    rem = t - landmark
    occ = ((e == 1) & (rem <= horizon)).astype(int)
    dur = np.minimum(rem, horizon)
    complete = (occ == 1) | (rem >= horizon)
 
    return dict(elig=elig, occ=occ, dur=dur, complete=complete)
 
 
def _feat_index(features, names):
    return [features.index(n) for n in names if n in features]
 
 
# ================================================================== rollouts
#
# Both discrete-time models are rolled forward from the landmark interval by
# ancestral sampling: draw each event from its hazard, update the state, step.
#
# The transformer's prev_* feed back, so a simulated diagnosis changes later
# hazards. The simple binary model sees only FIXED baseline prevalence, so it
# has no feedback -- that asymmetry is the point of the comparison, and it is
# why only the sequential models can generate event-to-event dependence.
#
# BMI is held at its landmark value: future nurse visits are not known at
# prediction time. Age advances with the interval.
 
@torch.no_grad()
def rollout_transformer(model, long_tr, features, event_types, landmark_interval,
                        n_steps, interval_years=2.0, n_sims=100, ids=None,
                        death_event="death", seed=0):
    """
    Returns ever : (n, n_sims, K) -- 1 if event k occurred by the end of the
    horizon, simulating forward from `landmark_interval`.
 
    Death absorbs: once it fires the trajectory stops, so no further diagnoses
    are generated for someone the model has already killed. Without this every
    absolute risk at older ages is overstated.
    """
    g = torch.Generator().manual_seed(seed)
    model.eval()
 
    df = long_tr.sort_values(["id", "interval"]).reset_index(drop=True)
    all_ids = df["id"].drop_duplicates().to_numpy()
    T = df["interval"].nunique()
    p = len(features)
    X = torch.tensor(df[features].to_numpy(np.float32)).view(len(all_ids), T, p)
 
    if ids is not None:
        pos = {v: i for i, v in enumerate(all_ids)}
        X = X[[pos[v] for v in ids]]
        all_ids = np.asarray(ids)
 
    n = X.shape[0]
    K = len(event_types)
    kd = event_types.index(death_event) if death_event in event_types else None
 
    i_prev = {e: features.index(f"prev_{e}") for e in event_types
              if f"prev_{e}" in features}
    i_age = features.index("age") if "age" in features else None
 
    L = landmark_interval
    ever = torch.zeros(n, n_sims, K)
 
    for s in range(n_sims):
        x = X[:, :L + 1, :].clone()
        alive = torch.ones(n, dtype=torch.bool)
 
        for step in range(n_steps):
            if x.shape[1] > model.n_intervals:            # positional limit
                break
            haz = torch.sigmoid(model(x)[:, -1, :])        # (n, K)
            fired = (torch.rand(haz.shape, generator=g) < haz).float()
 
            # cannot re-acquire something already present
            have = torch.zeros(n, K)
            for k, e in enumerate(event_types):
                if e in i_prev:
                    have[:, k] = x[:, -1, i_prev[e]]
            fired = fired * (1.0 - have) * alive[:, None].float()
 
            ever[:, s] = torch.clamp(ever[:, s] + fired, max=1.0)
            if kd is not None:
                alive &= fired[:, kd] == 0
            if not alive.any():
                break
 
            nxt = x[:, -1:].clone()
            for k, e in enumerate(event_types):
                if e in i_prev:
                    nxt[:, 0, i_prev[e]] = torch.clamp(
                        x[:, -1, i_prev[e]] + fired[:, k], max=1.0)
            if i_age is not None:
                nxt[:, 0, i_age] = x[:, -1, i_age] + interval_years
            x = torch.cat([x, nxt], dim=1)
 
    return ever.numpy(), all_ids
 
 
@torch.no_grad()
def rollout_binary(model, scaler, long_bin, features, event_types,
                   landmark_interval, n_steps, interval_years=2.0,
                   n_sims=100, ids=None, death_event="death", seed=0):
    """
    Same interface for the simple binary model.
 
    Its inputs are baseline covariates plus FIXED prevalent_* indicators, so a
    simulated diagnosis does not change later hazards: there is no feedback,
    and any event-pair correlation it produces can only come from shared risk
    factors. That is the comparison the correlation figure is built around.
    """
    g = torch.Generator().manual_seed(seed)
    model.eval()
 
    df = long_bin.sort_values(["id", "interval"]).reset_index(drop=True)
    all_ids = df["id"].drop_duplicates().to_numpy()
    T = df["interval"].nunique()
    F = np.asarray(df[features].to_numpy(np.float32)).reshape(len(all_ids), T, -1)
 
    if ids is not None:
        pos = {v: i for i, v in enumerate(all_ids)}
        F = F[[pos[v] for v in ids]]
        all_ids = np.asarray(ids)
 
    n, _, p = F.shape
    K = len(event_types)
    kd = event_types.index(death_event) if death_event in event_types else None
    L = landmark_interval
 
    ever = np.zeros((n, n_sims, K), dtype=np.float32)
 
    for s in range(n_sims):
        alive = np.ones(n, dtype=bool)
        have = np.zeros((n, K), dtype=np.float32)
        for step in range(n_steps):
            t = min(L + step, T - 1)
            x = torch.tensor(F[:, t, :], dtype=torch.float32)
            x = torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32)
            it = torch.full((n,), min(t, model.n_intervals - 1), dtype=torch.long)
            haz = torch.sigmoid(model(x, it)).numpy()
 
            fired = (np.random.default_rng(seed + s * 1000 + step).random(haz.shape)
                     < haz).astype(np.float32)
            fired = fired * (1.0 - have) * alive[:, None]
 
            ever[:, s] = np.clip(ever[:, s] + fired, 0, 1)
            have = np.clip(have + fired, 0, 1)
            if kd is not None:
                alive &= fired[:, kd] == 0
            if not alive.any():
                break
 
    return ever, all_ids
 
 
# ====================================================== 1) CALIBRATION
 
def calibration_table(wide, preds, event, landmark, horizon, n_bins=10):
    """
    Observed vs predicted risk of `event` in (T0, T0+H], in bins of predicted
    risk. Observed risk is 1 - KM(H), so people censored before the horizon are
    handled rather than counted as non-events.
 
    preds : dict  model name -> (n,) predicted P(event by T0+H), aligned to `wide`
    """
    lf = landmark_frame(wide, event, landmark, horizon)
    elig = lf["elig"]
 
    rows = []
    for name, P in preds.items():
        p = np.asarray(P)[elig]
        d = lf["dur"][elig]
        e = lf["occ"][elig]
        if len(p) < 50 or e.sum() < 10:
            continue
        try:
            b = pd.qcut(p, n_bins, labels=False, duplicates="drop")
        except ValueError:
            b = np.zeros(len(p), dtype=int)
 
        for gbin in np.unique(b):
            m = b == gbin
            if m.sum() < 5 or e[m].sum() == 0:
                risk = lo = hi = np.nan
            else:
                km = KaplanMeierFitter().fit(d[m], e[m])
                s = float(km.predict(horizon))
                ci = km.confidence_interval_survival_function_
                j = max(np.searchsorted(ci.index.values, horizon, "right") - 1, 0)
                lo_s, hi_s = ci.iloc[j].values
                risk, lo, hi = 1 - s, 1 - hi_s, 1 - lo_s
            rows.append({"model": name, "bin": int(gbin), "n": int(m.sum()),
                         "n_events": int(e[m].sum()),
                         "pred_mean": float(p[m].mean()),
                         "obs": risk, "obs_lo": lo, "obs_hi": hi})
    return pd.DataFrame(rows)
 
 
def calibration_slope(wide, preds, event, landmark, horizon):
    """
    INDIVIDUAL-LEVEL calibration slope and calibration-in-the-large.
 
    The slope is the coefficient from a Cox model of the outcome on
    cloglog(predicted risk). It uses every person rather than a handful of
    binned points, so it does not depend on how the bins were drawn -- binning
    is for the picture, not for the number.
 
        slope = 1  predictions are correctly spread
        slope < 1  too extreme: high risks too high, low risks too low
                   (the usual overfitting signature)
        slope > 1  too compressed
 
    CITL compares mean predicted risk with the overall observed (KM) risk:
    0 means the average level is right.
    """
    from lifelines import CoxPHFitter
 
    lf = landmark_frame(wide, event, landmark, horizon)
    elig = lf["elig"]
    d, e = lf["dur"][elig], lf["occ"][elig]
 
    out = []
    for name, P in preds.items():
        p = np.clip(np.asarray(P)[elig], 1e-6, 1 - 1e-6)
        if e.sum() < 10:
            continue
        lp = np.log(-np.log(1 - p))                    # cloglog
        try:
            cph = CoxPHFitter().fit(
                pd.DataFrame({"T": np.maximum(d, 1e-3), "E": e, "lp": lp}),
                "T", "E")
            slope = float(cph.params_["lp"])
            se = float(cph.standard_errors_["lp"])
        except Exception:
            slope = se = np.nan
 
        km = KaplanMeierFitter().fit(d, e)
        obs = 1 - float(km.predict(horizon))
        out.append({"model": name, "n": int(elig.sum()), "n_events": int(e.sum()),
                    "pred_mean": float(p.mean()), "obs_overall": obs,
                    "CITL": obs - float(p.mean()),
                    "slope": slope,
                    "slope_lo": slope - 1.96 * se, "slope_hi": slope + 1.96 * se})
    return pd.DataFrame(out).set_index("model").round(3)
 
 
def calibration_metrics(tab):
    """CITL, slope and ICI per model, from a calibration table.
 
    The slope here is fitted through the BINNED points and so depends on the
    binning. Prefer calibration_slope() for a number you will report; this one
    is a quick check that matches what the plot shows.
    """
    out = []
    for name, g in tab.groupby("model"):
        g = g.dropna(subset=["obs"])
        if len(g) < 3:
            continue
        w = g["n"].to_numpy(float)
        slope = np.polyfit(g.pred_mean, g.obs, 1, w=w)[0]
        out.append({"model": name,
                    "CITL": float(np.average(g.obs - g.pred_mean, weights=w)),
                    "slope": float(slope),
                    "ICI": float(np.average(np.abs(g.obs - g.pred_mean), weights=w))})
    return pd.DataFrame(out).set_index("model").round(4)
 
 
def order_events_by_slope(slopes, model=None, ascending=True):
    """
    Event order by calibration quality: |slope - 1|, best first.
 
    Distance from 1 rather than the slope itself, because 0.7 (too extreme)
    and 1.3 (too compressed) are both miscalibrated -- in opposite directions.
 
    slopes : dict event -> DataFrame from calibration_slope()
    model  : which model to rank on; None averages |slope - 1| across models,
             which ranks the EVENT rather than any one model.
    """
    d = {}
    for e, tab in slopes.items():
        if tab is None or len(tab) == 0 or "slope" not in tab:
            continue
        s = tab["slope"]
        if model is not None:
            if model not in s.index:
                continue
            d[e] = abs(float(s.loc[model]) - 1)
        else:
            d[e] = float(np.nanmean(np.abs(s.to_numpy(float) - 1)))
    return [e for e, _ in sorted(d.items(), key=lambda kv: kv[1],
                                 reverse=not ascending)]
 
 
def plot_calibration_grid(tabs, landmark, horizon, ncols=4, figsize=(16, 9),
                          order=None, slopes=None, order_model=None,
                          annotate_slope=True):
    """
    One panel per event. tabs : dict event -> calibration_table.
 
    order        : explicit event order, best first
    slopes       : dict event -> calibration_slope() table; if given and
                   `order` is None, panels are sorted by |slope - 1|
    order_model  : rank on this model; None averages across models
    annotate_slope : print the slope in each panel, so the ordering is legible
                     rather than something the reader has to take on trust
    """
    import matplotlib.pyplot as plt
 
    if order is None and slopes is not None:
        order = order_events_by_slope(slopes, model=order_model)
 
    if order is not None:
        items = [(e, tabs[e]) for e in order if e in tabs and len(tabs[e])]
        items += [(e, t) for e, t in tabs.items()
                  if e not in set(order) and len(t)]
    else:
        items = [(e, t) for e, t in tabs.items() if len(t)]
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.tab10.colors
 
    for ax, (ev, tab) in zip(axes, items):
        hi = float(np.nanmax([tab.pred_mean.max(), tab.obs_hi.max()])) * 1.08
        ax.plot([0, hi], [0, hi], ls="--", c="grey", lw=1, zorder=0)
        for i, (name, g) in enumerate(tab.groupby("model")):
            g = g.sort_values("pred_mean")
            ax.errorbar(g.pred_mean, g.obs,
                        yerr=[g.obs - g.obs_lo, g.obs_hi - g.obs],
                        fmt="o-", ms=4, lw=1.3, capsize=2,
                        color=colors[i % 10], label=name)
        ax.set_title(ev, fontsize=10)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
 
        if annotate_slope and slopes is not None and ev in slopes:
            s = slopes[ev].get("slope")
            if s is not None and len(s):
                txt = "\n".join(f"{m[:14]} {v:.2f}" for m, v in s.items())
                ax.annotate(txt, (0.03, 0.97), xycoords="axes fraction",
                            va="top", ha="left", fontsize=7, color="#52514e")
    for a in axes[len(items):]:
        a.axis("off")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.supxlabel(f"Predicted risk by T0+{horizon:g}y")
    fig.supylabel(f"Observed risk at {horizon:g}y")
    fig.suptitle(f"Calibration, landmark T0 = {landmark:g}y, horizon {horizon:g}y")
    fig.tight_layout()
    return fig
 
 
# ============================================ 2) DISCRIMINATION PROGRESSION
 
def cindex_progression(wide, scorers, event_types, landmarks, horizon,
                       min_events=20):
    """
    C-index at each landmark, for each model, on an identical risk set.
 
    scorers : dict  model name -> callable(landmark) -> (n, K) risk scores
              aligned to `wide`, higher = higher risk.
    """
    rows = []
    for lm in landmarks:
        scores = {name: np.asarray(fn(lm)) for name, fn in scorers.items()}
        for k, ev in enumerate(event_types):
            lf = landmark_frame(wide, ev, lm, horizon)
            elig = lf["elig"]
            n_ev = int(lf["occ"][elig].sum())
            if n_ev < min_events:
                continue
            rec = {"event": ev, "landmark": lm,
                   "n_at_risk": int(elig.sum()), "n_events": n_ev}
            for name, S in scores.items():
                rec[name] = concordance_index(lf["dur"][elig], -S[elig, k],
                                              lf["occ"][elig])
            rows.append(rec)
    return pd.DataFrame(rows)
 
 
def plot_cindex_progression(prog, model_cols, event_types=None,
                            ncols=4, figsize=(16, 9)):
    import matplotlib.pyplot as plt
    evs = event_types or sorted(prog.event.unique())
    evs = [e for e in evs if e in set(prog.event)]
    nrows = int(np.ceil(len(evs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.tab10.colors
 
    for ax, ev in zip(axes, evs):
        g = prog[prog.event == ev].sort_values("landmark")
        for i, c in enumerate(model_cols):
            if c in g:
                ax.plot(g.landmark, g[c], "o-", lw=1.6, ms=4,
                        color=colors[i % 10], label=c)
        ax.axhline(0.5, ls=":", c="grey", lw=.8)
        ax.set_title(ev, fontsize=10)
        ax.set_ylim(0.45, 0.90)
    for a in axes[len(evs):]:
        a.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Available history at prediction (years since entry)")
    fig.supylabel("C-index")
    fig.tight_layout()
    return fig
 
 
# ================================================ 3) PAIR-WISE CORRELATIONS
 
def _phi(p_i, p_j, p_ij):
    den = np.sqrt(p_i * (1 - p_i) * p_j * (1 - p_j))
    return np.nan if den <= 0 else (p_ij - p_i * p_j) / den
 
 
def observed_phi(wide, event_types, landmark, horizon, min_n=100):
    """
    Population phi between I(event i in (T0, T0+H]) and I(event j in ...),
    among people at risk for BOTH at T0 with follow-up through the horizon.
 
    Restriction to complete follow-up is what makes the two indicators binary
    and comparable; note it as a complete-case restriction.
    """
    K = len(event_types)
    M = pd.DataFrame(np.nan, index=event_types, columns=event_types)
    lfs = {e: landmark_frame(wide, e, landmark, horizon) for e in event_types}
 
    for i, j in combinations(range(K), 2):
        a, b = lfs[event_types[i]], lfs[event_types[j]]
        keep = a["elig"] & b["elig"] & a["complete"] & b["complete"]
        if keep.sum() < min_n:
            continue
        yi, yj = a["occ"][keep], b["occ"][keep]
        v = _phi(yi.mean(), yj.mean(), (yi * yj).mean())
        M.iloc[i, j] = M.iloc[j, i] = v
    return M
 
 
def phi_matched(wide, ever, event_types, landmark, horizon, ids=None,
                min_n=100, exclude=("death",), report_n=False):
    """
    Observed and model phi on IDENTICAL subsets, pair by pair.
 
    This is the only fair comparison. Computing observed phi among people at
    risk for both events, while computing model phi over everyone in the
    rollout, compares different populations: someone prevalent for hypertension
    can never have an incident hypertension event, but is high risk for CVD,
    stroke and death. Pooled into the model matrix they contribute a block of
    (0, 1) pairs that pushes model phi DOWN -- most for the most prevalent
    condition, and for every pair involving death. The result looks like the
    model failing to learn co-occurrence when it is an artefact of the
    denominator.
 
    ever : (n, n_sims, K) rollout, rows aligned to `ids` (default: wide order)
 
    `exclude` drops events from the REPORTED matrix only -- death must stay in
    the rollout, where it absorbs, or the simulation keeps generating diagnoses
    for people it has already killed. That matters more the longer the horizon.
 
    Returns (M_observed, M_model), or (M_observed, M_model, N) with report_n.
    """
    K = len(event_types)
    lfs = {e: landmark_frame(wide, e, landmark, horizon) for e in event_types}
    N = pd.DataFrame(np.nan, index=event_types, columns=event_types)
 
    if ids is not None:                       # align the rollout to `wide`
        pos = {v: i for i, v in enumerate(np.asarray(ids))}
        order = np.array([pos[v] for v in wide["id"].to_numpy()])
        ever = ever[order]
 
    Mo = pd.DataFrame(np.nan, index=event_types, columns=event_types)
    Mm = pd.DataFrame(np.nan, index=event_types, columns=event_types)
 
    for i, j in combinations(range(K), 2):
        a, b = lfs[event_types[i]], lfs[event_types[j]]
        keep = a["elig"] & b["elig"] & a["complete"] & b["complete"]
        N.iloc[i, j] = N.iloc[j, i] = int(keep.sum())
        if keep.sum() < min_n:
            continue
 
        yi, yj = a["occ"][keep], b["occ"][keep]
        Mo.iloc[i, j] = Mo.iloc[j, i] = _phi(yi.mean(), yj.mean(), (yi * yj).mean())
 
        E = ever[keep].reshape(-1, K)          # same people, pooled over sims
        pi, pj = E[:, i].mean(), E[:, j].mean()
        Mm.iloc[i, j] = Mm.iloc[j, i] = _phi(pi, pj, (E[:, i] * E[:, j]).mean())
 
    drop = [e for e in (exclude or ()) if e in Mo.index]
    if drop:
        Mo = Mo.drop(index=drop, columns=drop)
        Mm = Mm.drop(index=drop, columns=drop)
        N = N.drop(index=drop, columns=drop)
 
    return (Mo, Mm, N) if report_n else (Mo, Mm)
 
 
def model_phi(ever, event_types):
    """Phi from a rollout: ever is (n, n_sims, K)."""
    K = len(event_types)
    M = pd.DataFrame(np.nan, index=event_types, columns=event_types)
    flat = ever.reshape(-1, K)                     # pool person x simulation
    p = flat.mean(0)
    for i, j in combinations(range(K), 2):
        pij = (flat[:, i] * flat[:, j]).mean()
        M.iloc[i, j] = M.iloc[j, i] = _phi(p[i], p[j], pij)
    return M
 
 
def plot_phi_grid(mats, landmarks, event_types, figsize=(16, 8), vmax=0.3):
    """
    mats : dict (landmark, model_name) -> phi matrix.
    Rows = landmarks, columns = models, mirroring the simulation figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
 
    names = list(dict.fromkeys(k[1] for k in mats))
    fig, axes = plt.subplots(len(landmarks), len(names),
                             figsize=figsize, squeeze=False)
    for r, lm in enumerate(landmarks):
        for c, name in enumerate(names):
            ax = axes[r][c]
            M = mats.get((lm, name))
            if M is None:
                ax.axis("off"); continue
            sns.heatmap(M.astype(float), ax=ax, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, center=0, annot=True, fmt=".2f",
                        square=True, linewidths=1.5, linecolor="white",
                        cbar=(c == len(names) - 1), annot_kws={"size": 6})
            if r == 0:
                ax.set_title(name, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"T0 = {lm:g}y", fontsize=10)
            ax.tick_params(labelsize=7)
    fig.suptitle("Event-pair correlation over the prediction horizon")
    fig.tight_layout()
    return fig
 
 
# ================================================= 4) INFLUENCE (LOG-ODDS)
 
@torch.no_grad()
def influence_matrix(model, long_tr, features, event_types, s, t,
                     ids=None, chunk=512):
    """
    M[j, k] = mean over people of
        logit h_k(t | event j present from s)  -  logit h_k(t | no history)
 
    A directed, covariate-adjusted contrast: everything that does not differ
    between the two arms differences away. Unlike phi it is asymmetric, so it
    can separate "j then k" from "k then j".
 
    In ELSA there is no ground truth, so this is a PREDICTIVE TEMPORAL
    ASSOCIATION, not a causal effect. Judge it by face validity against known
    progressions.
    """
    model.eval()
    ev = [e for e in event_types if f"prev_{e}" in features]
    K = len(ev)
    i_prev = [features.index(f"prev_{e}") for e in ev]
 
    df = long_tr.sort_values(["id", "interval"]).reset_index(drop=True)
    all_ids = df["id"].drop_duplicates().to_numpy()
    T = df["interval"].nunique()
    Z = torch.tensor(df[features].to_numpy(np.float32)).view(
        len(all_ids), T, len(features))
    if ids is not None:
        pos = {v: i for i, v in enumerate(all_ids)}
        Z = Z[[pos[v] for v in ids]]
 
    if not (0 <= s < t < T):
        raise ValueError(f"need 0 <= s < t < T (got s={s}, t={t}, T={T})")
 
    Z = Z[:, :t + 1, :].clone()
    n = Z.shape[0]
 
    def logits_at_t(x):
        out = []
        for a in range(0, n, chunk):
            out.append(model(x[a:a + chunk])[:, t, :])
        return torch.cat(out, 0)
 
    base = Z.clone()
    base[:, :, i_prev] = 0.0
    l0 = logits_at_t(base)
 
    kk = [event_types.index(e) for e in ev]
    M = pd.DataFrame(np.nan, index=ev, columns=ev, dtype=float)
    for a, j in enumerate(range(K)):
        z = base.clone()
        z[:, s:, i_prev[j]] = 1.0
        d = (logits_at_t(z) - l0).mean(0).numpy()
        for b, k in enumerate(kk):
            if a != b:
                M.iloc[a, b] = float(d[k])
    return M
 
 
def plot_influence(M, title=None, figsize=(7.5, 6.5), vmax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    v = vmax or float(np.nanmax(np.abs(M.values.astype(float))))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(M.astype(float), cmap="RdBu_r", vmin=-v, vmax=v, center=0,
                annot=True, fmt=".2f", square=True, linewidths=1.5,
                linecolor="white", ax=ax, annot_kws={"size": 7})
    ax.set_xlabel("effect on event k")
    ax.set_ylabel("cause: event j")
    ax.set_title(title or "Δ log-odds of k, given j present")
    fig.tight_layout()
    return fig
 


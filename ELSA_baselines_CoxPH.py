##############################################################################
## Landmark baselines: Cox PH per event, vs the transformer                 ##
##############################################################################
#
# The comparison only means something if every model is scored on EXACTLY the
# same risk set, outcome and censoring. So one helper builds the landmark
# dataset, and every model is scored from it.
#
#   risk set : under observation at `landmark_bin` AND still at risk for k
#   outcome  : k first diagnosed within the next `horizon_bins` bins
#   censored : at the last observed bin inside the window
#
# Three comparators, in increasing order of what they get to see:
#
#   Cox (baseline)    covariates at the landmark, no disease history
#   Cox (+ history)   the same, plus prev_* at the landmark      <- snapshot
#   Transformer       the full causal sequence up to the landmark <- history
#
# Cox(+history) vs Transformer is the contrast that isolates what SEQUENCE
# buys you over a point-in-time summary of the same information. Cox(baseline)
# vs Cox(+history) isolates what knowing the comorbidities buys you at all.
 
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
 
 
# ------------------------------------------------------- the landmark dataset
 
def landmark_data(d, landmark_bin, horizon_bins, k):
    """
    Returns a dict for event index k at one landmark age bin.
 
      elig  (n,) bool   under observation at the landmark and still at risk
      Xl    (n, p)      features AT the landmark row
      dur   (n,)        bins from the landmark to event or censoring
      ev    (n,)        1 if the event occurred inside the window
      row   (n,)        index of the landmark row, for scoring
    """
    X, Y, R, P, O = d["X"], d["Y"], d["R"], d["pos_idx"], d["obs"]
    n, T, _ = X.shape
    idx = torch.arange(n)
 
    at   = (P == landmark_bin) & (O > 0)
    has  = at.any(1)
    row  = at.float().argmax(1)
    elig = has & (R[idx, row, k] > 0)
 
    first = torch.full((n,), float("inf"))
    nobs  = torch.zeros(n)
    for j in range(horizon_bins + 1):
        t     = (row + j).clamp(max=T - 1)
        valid = (row + j <= T - 1).float()
        y     = Y[idx, t, k] * valid * O[idx, t]
        nobs += O[idx, t] * valid
        hit   = (y > 0) & torch.isinf(first)
        first[hit] = float(j)
 
    ev  = (~torch.isinf(first)).float()
    dur = torch.where(ev > 0, first, nobs.clamp(min=1.0))
 
    return dict(elig=elig.numpy(), Xl=X[idx, row].numpy(),
                dur=dur.numpy(), ev=ev.numpy().astype(int),
                row=row, n_eligible=int(elig.sum()),
                n_events=int(ev[elig].sum()))
 
 
# ------------------------------------------------------------------ scorers
 
@torch.no_grad()
def transformer_score(model, d, lm, horizon_bins, k, row):
    """Cumulative hazard over the window, history frozen at the landmark."""
    X, P = d["X"], d["pos_idx"]
    n, T, _ = X.shape
    haz = torch.sigmoid(model(X, P))[:, :, k]            # (n, T)
    idx = torch.arange(n)
    s = torch.zeros(n)
    for j in range(horizon_bins + 1):
        t = (row + j).clamp(max=T - 1)
        s += haz[idx, t] * (row + j <= T - 1).float()
    return s.numpy()
 
 
def _fit_cox(Xtr, dur, ev, cols, penalizer=0.1):
    df = pd.DataFrame(Xtr, columns=cols)
    keep = [c for c in cols if df[c].std() > 1e-8]       # drop constants
    df = df[keep]
    df["T"], df["E"] = np.maximum(dur, 1e-3), ev
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, "T", "E")
    return cph, keep
 
 
# ------------------------------------------------------------------ compare
 
def compare_at_landmark(model, tr, te, landmark_bin, horizon_bins, events,
                        feature_names, n_hist=None, min_events=20,
                        penalizer=0.1, verbose=False):
    """
    One row per event: C-index for Cox (baseline), Cox (+history) and the
    transformer, on an identical risk set and outcome.
 
    n_hist : number of trailing prev_* columns in `feature_names`
             (defaults to len(events), which is how build_arrays lays them out)
    """
    n_hist = len(events) if n_hist is None else n_hist
    base_cols = feature_names[:-n_hist]
    hist_cols = feature_names[-n_hist:]
 
    rows = []
    for k, ev_name in enumerate(events):
        a = landmark_data(tr, landmark_bin, horizon_bins, k)
        b = landmark_data(te, landmark_bin, horizon_bins, k)
        if b["n_events"] < min_events or a["n_events"] < min_events:
            continue
 
        ta, tb = a["elig"], b["elig"]
        rec = {"event": ev_name,
               "n_eligible": b["n_eligible"], "n_events": b["n_events"]}
 
        def cindex(score):
            return concordance_index(b["dur"][tb], -score[tb], b["ev"][tb])
 
        # --- Cox, baseline covariates only ---
        try:
            cph, keep = _fit_cox(a["Xl"][ta][:, :len(base_cols)],
                                 a["dur"][ta], a["ev"][ta], base_cols, penalizer)
            lp = cph.predict_partial_hazard(
                pd.DataFrame(b["Xl"][:, :len(base_cols)], columns=base_cols)[keep]
            ).values
            rec["cox_baseline"] = cindex(lp)
        except Exception as e:
            rec["cox_baseline"] = np.nan
            if verbose: print(f"  {ev_name} cox_baseline: {e}")
 
        # --- Cox, covariates + comorbidity snapshot ---
        try:
            cph, keep = _fit_cox(a["Xl"][ta], a["dur"][ta], a["ev"][ta],
                                 feature_names, penalizer)
            lp = cph.predict_partial_hazard(
                pd.DataFrame(b["Xl"], columns=feature_names)[keep]
            ).values
            rec["cox_history"] = cindex(lp)
        except Exception as e:
            rec["cox_history"] = np.nan
            if verbose: print(f"  {ev_name} cox_history: {e}")
 
        # --- transformer, full causal sequence ---
        rec["transformer"] = cindex(
            transformer_score(model, te, landmark_bin, horizon_bins, k, b["row"]))
 
        rows.append(rec)
 
    out = pd.DataFrame(rows).set_index("event")
    out["hist_gain"] = out["cox_history"] - out["cox_baseline"]
    out["seq_gain"]  = out["transformer"] - out["cox_history"]
    return out.round(3)
 
 
def compare_sweep(model, tr, te, landmarks, horizon_bins, events,
                  feature_names, age_min=50, bin_width=2, **kw):
    """compare_at_landmark across several landmarks, stacked long."""
    frames = []
    for lm in landmarks:
        r = compare_at_landmark(model, tr, te, lm, horizon_bins, events,
                                feature_names, **kw)
        frames.append(r.assign(landmark_bin=lm,
                               landmark_age=age_min + lm * bin_width))
    return pd.concat(frames)
 
 
# --------------------------------------------------------------- bootstrap
 
def bootstrap_cindex(score, dur, ev, elig, n_boot=500, seed=0):
    """Percentile CI for a C-index, resampling PEOPLE."""
    rng = np.random.default_rng(seed)
    s, d_, e = score[elig], dur[elig], ev[elig]
    n = len(s)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if e[i].sum() < 5:
            continue
        out.append(concordance_index(d_[i], -s[i], e[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
 
 
# ------------------------------------------------------------------- plot
 
def plot_comparison(sweep, events=None, ncols=4, figsize=(16, 9)):
    """C-index vs landmark age, one panel per event, one line per model."""
    import matplotlib.pyplot as plt
 
    evs = events or sorted(sweep.index.unique())
    evs = [e for e in evs if e in set(sweep.index)]
    nrows = int(np.ceil(len(evs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
 
    cols = {"cox_baseline": ("#2c7fb8", "Cox (baseline)"),
            "cox_history":  ("#31a354", "Cox (+ history)"),
            "transformer":  ("#e6550d", "Transformer")}
 
    for ax, ev in zip(axes, evs):
        g = sweep.loc[[ev]].sort_values("landmark_age")
        for c, (col, lab) in cols.items():
            if c in g:
                ax.plot(g.landmark_age, g[c], "o-", color=col, lw=1.6,
                        ms=4, label=lab)
        ax.axhline(0.5, ls=":", c="grey", lw=.8)
        ax.set_title(ev, fontsize=10)
        ax.set_ylim(0.45, 0.85)
 
    for a in axes[len(evs):]:
        a.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("landmark age"); fig.supylabel("C-index")
    fig.tight_layout()
    return fig
 


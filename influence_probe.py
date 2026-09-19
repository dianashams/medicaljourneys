##############################################
## COUNTERFACTUAL INFLUENCE PROBE            ##
##############################################
#
# Reads a DIRECTED, covariate-adjusted event-to-event effect out of a trained
# sequence model, and validates it against the simulator's known coefficients.
#
#   M[j, k] = E_i [ logit h_k(t | prev_j = 1 from s)  -  logit h_k(t | no history) ]
#
# Unlike phi (symmetric, and mixing shared risk factors with dependence), M is
# directed and differences away everything that does not change between arms.
#
# Ground truth is available because sim_population's hazards are
#     rate_k = c_k * exp( ... + beta_{j->k} * 1{event j occurred before t} + ... )
# so beta_{j->k} IS the effect of j on k, on the log-rate scale.
#
# Requires: a model trained on features that INCLUDE prev_a … prev_e.

import numpy as np
import pandas as pd
import torch


def _blank_diag(df):
    """Set the diagonal to NaN without touching a read-only .values view."""
    for j in df.index:
        df.loc[j, j] = np.nan
    return df


# ----------------------------------------------------------------- ground truth

def true_influence_matrix(event_types=("a", "b", "c", "d", "e")):
    """
    Direct effects hard-coded in simulate_population.py, on the log-rate scale.

      get_time_a : eventb_beta = 1{first_b} * 0.3          ->  b -> a = 0.3
      get_time_b : eventa_beta = 1{first_a} * 1.0          ->  a -> b = 1.0
      get_time_c : no event terms                          ->  * -> c = 0
      get_time_d : no event terms                          ->  * -> d = 0
      get_time_e : comorb_beta = 0.5*count + 0.5*(count>2) + 1*(count>3)
                   0 -> 1 comorbidity adds 0.5             ->  * -> e = 0.5

    NOTE: e -> b and e -> c are ZERO here. The simulator links them only through
    BMI drift, which this probe holds fixed. So the probe measures the DIRECT
    effect, while phi shows the TOTAL (direct + BMI-mediated) association.
    The gap between them is mediation, not disagreement.
    """
    ev = list(event_types)
    M = pd.DataFrame(0.0, index=ev, columns=ev)      # rows = cause, cols = effect
    if "b" in ev and "a" in ev:
        M.loc["b", "a"] = 0.3
    if "a" in ev and "b" in ev:
        M.loc["a", "b"] = 1.0
    if "e" in ev:
        for j in ev:
            if j != "e":
                M.loc[j, "e"] = 0.5
    return _blank_diag(M)


# ----------------------------------------------------------------- the probe

@torch.no_grad()
def influence_matrix(model, Z, s, t, feature_names, event_types,
                     n_boot=0, seed=0, chunk=512, return_per_patient=False):
    """
    Parameters
    ----------
    model  : trained model with forward(x) -> (B, T, K), trained on `feature_names`
    Z      : (n, T, p) RAW feature sequences (the model scales internally)
    s      : interval at which the counterfactual event is switched on
    t      : interval at which the effect is read off   (must have t > s)
    n_boot : if > 0, bootstrap over patients for 95% CIs

    Returns
    -------
    M            if n_boot == 0
    M, lo, hi    if n_boot > 0
    (M, D)       if return_per_patient (D is (K, n, K): cause, patient, effect)
    """
    model.eval()
    ev, K, p = list(event_types), len(event_types), len(feature_names)

    # ---- guards: these catch the two mistakes that produce silent nonsense ----
    missing = [f"prev_{e}" for e in ev if f"prev_{e}" not in feature_names]
    if missing:
        raise ValueError(f"feature_names lacks {missing} — this probe needs a model "
                         "trained WITH the prev_* history features")
    if getattr(model, "p", p) != p:
        raise ValueError(f"model expects p={model.p} features, got {p}. "
                         "Wrong model/feature-set pairing?")
    if Z.shape[2] != p:
        raise ValueError(f"Z has {Z.shape[2]} features, feature_names has {p}")
    if t <= s:
        raise ValueError("need t > s")
    if t >= Z.shape[1]:
        raise ValueError(f"t={t} out of range (T={Z.shape[1]})")

    i_prev = [feature_names.index(f"prev_{e}") for e in ev]
    Z = Z[:, :t + 1, :].clone()                  # nothing after t can matter
    n = Z.shape[0]

    def logits_at_t(x):
        out = []
        for a in range(0, x.shape[0], chunk):
            out.append(model(x[a:a + chunk])[:, t, :])
        return torch.cat(out, 0)

    # --- reference arm: clean history ---
    base = Z.clone()
    base[:, :, i_prev] = 0.0
    l0 = logits_at_t(base)                        # (n, K)

    # --- one counterfactual arm per cause ---
    D = np.zeros((K, n, K), dtype=np.float32)     # [cause, patient, effect]
    for j in range(K):
        z = base.clone()
        z[:, s:, i_prev[j]] = 1.0                 # event j present from s onward
        D[j] = (logits_at_t(z) - l0).numpy()

    M = _blank_diag(pd.DataFrame(D.mean(1), index=ev, columns=ev).astype(float))

    if return_per_patient:
        return M, D
    if n_boot == 0:
        return M

    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, K, K), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = D[:, idx, :].mean(1)
    lo = _blank_diag(pd.DataFrame(np.percentile(boots, 2.5, 0),
                                  index=ev, columns=ev).astype(float))
    hi = _blank_diag(pd.DataFrame(np.percentile(boots, 97.5, 0),
                                  index=ev, columns=ev).astype(float))
    return M, lo, hi


@torch.no_grad()
def lag_response(model, Z, cause, effect, t, feature_names, event_types,
                 s_grid=None, chunk=512):
    """
    Effect of `cause` on `effect` at interval t, as a function of how long ago
    the cause occurred. Recovers the shape of any time-since-event effect
    without specifying a decay function.

    Returns a Series indexed by elapsed intervals (t - s).
    """
    model.eval()
    ev = list(event_types)
    j, k = ev.index(cause), ev.index(effect)
    i_prev = [feature_names.index(f"prev_{e}") for e in ev]
    s_grid = list(range(0, t)) if s_grid is None else list(s_grid)

    Zt = Z[:, :t + 1, :].clone()
    base = Zt.clone(); base[:, :, i_prev] = 0.0

    def logits_at_t(x):
        out = []
        for a in range(0, x.shape[0], chunk):
            out.append(model(x[a:a + chunk])[:, t, k])
        return torch.cat(out, 0)

    l0 = logits_at_t(base)
    vals = []
    for s in s_grid:
        z = base.clone()
        z[:, s:, i_prev[j]] = 1.0
        vals.append((logits_at_t(z) - l0).mean().item())
    return pd.Series(vals, index=[t - s for s in s_grid],
                     name=f"{cause}→{effect}").sort_index()


# ----------------------------------------------------------------- validation

def validate_influence(M, M_true, lo=None, hi=None):
    """Long-format estimated-vs-true comparison, plus summary metrics."""
    rows = []
    for j in M.index:
        for k in M.columns:
            if j == k or pd.isna(M.loc[j, k]):
                continue
            rows.append({
                "pair": f"{j}->{k}", "cause": j, "effect": k,
                "true": float(M_true.loc[j, k]),
                "est":  float(M.loc[j, k]),
                "lo":   np.nan if lo is None else float(lo.loc[j, k]),
                "hi":   np.nan if hi is None else float(hi.loc[j, k]),
                "is_real": bool(M_true.loc[j, k] != 0),
            })
    d = pd.DataFrame(rows)

    nz, zr = d[d.is_real], d[~d.is_real]
    metrics = {
        "n_pairs":              len(d),
        "n_true_nonzero":       len(nz),
        "pearson_r":            d["true"].corr(d["est"]),
        "spearman_r":           d["true"].corr(d["est"], method="spearman"),
        "rmse":                 float(np.sqrt(((d.est - d.true) ** 2).mean())),
        "slope_est_on_true":    float(np.polyfit(d.true, d.est, 1)[0]),
        "mean_abs_est_at_true0": float(zr.est.abs().mean()) if len(zr) else np.nan,
        "min_abs_est_at_real":   float(nz.est.abs().min()) if len(nz) else np.nan,
    }
    if lo is not None:
        metrics["ci_covers_truth"]   = float(((nz.lo <= nz.true) & (nz.true <= nz.hi)).mean())
        metrics["false_positives"]   = int(((zr.lo > 0) | (zr.hi < 0)).sum())
    return d.sort_values("true", ascending=False).reset_index(drop=True), metrics


def plot_influence_validation(M, M_true, d, figsize=(14.5, 4.6)):
    """Three panels: true matrix | estimated matrix | estimated-vs-true scatter."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    vals = np.r_[M.values.astype(float).ravel(), M_true.values.astype(float).ravel()]
    vmax = float(np.nanmax(np.abs(vals)))
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    for a, mat, title in [(ax[0], M_true, "True (simulator coefficients)"),
                          (ax[1], M,      "Estimated (transformer probe)")]:
        sns.heatmap(mat.astype(float), ax=a, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    center=0, annot=True, fmt=".2f", square=True, linewidths=2,
                    linecolor="white", cbar=False, annot_kws={"size": 9})
        a.set_title(title, fontsize=11)
        a.set_xlabel("effect on event k")
        a.set_ylabel("cause: event j")

    lim = [-0.12 * vmax, 1.18 * vmax]
    ax[2].plot(lim, lim, ls="--", c="grey", lw=1, zorder=0)
    ax[2].axhline(0, c="grey", lw=.8, ls=":", zorder=0)
    if d["lo"].notna().all():
        ax[2].errorbar(d.true, d.est, yerr=[d.est - d.lo, d.hi - d.est],
                       fmt="none", ecolor="#9aa0a6", lw=1, zorder=1)
    ax[2].scatter(d.true, d.est, s=58, zorder=2,
                  c=np.where(d.is_real, "#e6550d", "#3182bd"),
                  edgecolor="white", linewidth=1)
    for _, r in d.iterrows():
        if r.is_real or abs(r.est) > 0.12 * vmax:
            ax[2].annotate(r["pair"], (r.true, r.est), fontsize=8,
                           xytext=(5, 3), textcoords="offset points")
    ax[2].set_xlabel("true log-rate coefficient")
    ax[2].set_ylabel("estimated Δ logit")
    ax[2].set_title("Recovery of generative structure", fontsize=11)
    ax[2].set_xlim(lim); ax[2].set_ylim(lim)
    plt.tight_layout()
    return fig

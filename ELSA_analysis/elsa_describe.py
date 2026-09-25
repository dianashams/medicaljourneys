##############################################################################
## ELSA cohort description: Table 1 and the descriptive figures             ##
##############################################################################
#
# Works off the `wide` table from prepare_elsa_models.prepare_elsa_wide().
#
# Palette: validated categorical slots (blue, orange, aqua) for identity, a
# single-hue blue ramp for magnitude, blue<->red for the correlation heatmap.
# Categorical hues are assigned in fixed order and never cycled.
 
import numpy as np
import pandas as pd
 
# ------------------------------------------------------------------ palette
 
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
       "#256abf", "#184f95", "#0d366b"]          # light -> dark
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"
 
 
def _style(ax, xgrid=False, ygrid=True):
    """Recessive axes: no box, one faint grid direction, muted tick text."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_facecolor(SURFACE)
    return ax
 
 
# ------------------------------------------------------------------ Table 1
 
def _fmt_cont(s, digits=1):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.mean():.{digits}f} ({s.std():.{digits}f})"
 
 
def _fmt_med(s, digits=1):
    s = pd.to_numeric(s, errors="coerce").dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})"
 
 
def _fmt_n(mask):
    n = int(np.nansum(mask))
    return f"{n:,} ({100 * np.nanmean(mask):.1f})"
 
 
def table1(wide, event_types, by=None, sex_labels=(("0", "Male"), ("1", "Female")),
           wealth_label="Wealth quintile", edu_label="Education"):
    """
    Cohort description: overall and, optionally, stratified by `by`.
 
    Continuous variables are mean (SD); follow-up is median (IQR) because it is
    censored and skewed. Conditions are reported twice -- PREVALENT at entry and
    INCIDENT during follow-up -- because the two answer different questions and
    the denominator for incidence excludes the prevalent cases.
    """
    groups = [("Overall", wide)]
    if by is not None:
        lab = dict(sex_labels) if by == "sex" else {}
        for v, g in wide.groupby(by):
            groups.append((lab.get(str(int(v)) if float(v).is_integer() else str(v),
                                   f"{by}={v:g}"), g))
 
    rows = []
 
    def add(label, fn, indent=False):
        rows.append({"Characteristic": ("    " if indent else "") + label,
                     **{name: fn(g) for name, g in groups}})
 
    add("N", lambda g: f"{len(g):,}")
    rows.append({"Characteristic": "", **{n: "" for n, _ in groups}})
 
    add("Age at entry, mean (SD)", lambda g: _fmt_cont(g["age"]))
    add("BMI, mean (SD)", lambda g: _fmt_cont(g["bmi"]))
    if "sex" in wide:
        add("Female, n (%)", lambda g: _fmt_n(g["sex"] == 1))
    add(f"{edu_label}, median (IQR)", lambda g: _fmt_med(g["edu"]))
    add(f"{wealth_label}, median (IQR)", lambda g: _fmt_med(g["wealth"]))
 
    rows.append({"Characteristic": "", **{n: "" for n, _ in groups}})
    add("Follow-up (y), median (IQR)",
        lambda g: _fmt_med(g["exit_dis"] - g["age"]))
    add("Person-years", lambda g: f"{(g['exit_dis'] - g['age']).sum():,.0f}")
 
    dis = [e for e in event_types if e != "death"]
    rows.append({"Characteristic": "", **{n: "" for n, _ in groups}})
    add("Conditions at entry, mean (SD)",
        lambda g: _fmt_cont(g[[f"prevalent_{e}" for e in dis]].sum(1)))
    add("Multimorbid (2+) at entry, n (%)",
        lambda g: _fmt_n(g[[f"prevalent_{e}" for e in dis]].sum(1) >= 2))
 
    rows.append({"Characteristic": "Prevalent at entry, n (%)",
                 **{n: "" for n, _ in groups}})
    for e in dis:
        add(e, lambda g, e=e: _fmt_n(g[f"prevalent_{e}"] == 1), indent=True)
 
    rows.append({"Characteristic": "Incident during follow-up, n (%)",
                 **{n: "" for n, _ in groups}})
    for e in event_types:
        add(e, lambda g, e=e: _fmt_n(
            g.loc[g[f"prevalent_{e}"] == 0, f"event_{e}"] == 1), indent=True)
 
    return pd.DataFrame(rows).set_index("Characteristic")
 
 
# ------------------------------------------------------------------ figures
 
def plot_cohort(wide, event_types, figsize=(13.5, 9)):
    """
    Four panels: who is in the cohort, how long they are followed, how much
    disease they arrive with, and what they acquire.
    """
    import matplotlib.pyplot as plt
 
    dis = [e for e in event_types if e != "death"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=SURFACE)
    (a, b), (c, d) = axes
 
    # --- A. age at entry, by sex (identity -> categorical slots 1, 2) -------
    bins = np.arange(np.floor(wide["age"].min()), wide["age"].max() + 2, 2)
    for i, (v, lab) in enumerate([(0, "Male"), (1, "Female")]):
        g = wide.loc[wide["sex"] == v, "age"]
        a.hist(g, bins=bins, histtype="step", lw=2, color=SERIES[i], label=lab)
    _style(a)
    a.set_xlabel("Age at entry (years)", color=INK2, fontsize=9)
    a.set_ylabel("Participants", color=INK2, fontsize=9)
    a.set_title("Age at study entry", color=INK, fontsize=11, loc="left")
    a.legend(frameon=False, fontsize=9, labelcolor=INK2)
 
    # --- B. follow-up (single series -> no legend, title names it) ---------
    fu = (wide["exit_dis"] - wide["age"]).clip(lower=0)
    b.hist(fu, bins=np.arange(0, fu.max() + 2, 2), color=SEQ[3],
           edgecolor=SURFACE, linewidth=2)
    b.axvline(fu.median(), color=INK2, lw=1.5, ls="--")
    b.annotate(f"median {fu.median():.1f}y", (fu.median(), b.get_ylim()[1] * 0.92),
               xytext=(6, 0), textcoords="offset points",
               color=INK2, fontsize=9, va="top")
    _style(b)
    b.set_xlabel("Disease follow-up (years)", color=INK2, fontsize=9)
    b.set_ylabel("Participants", color=INK2, fontsize=9)
    b.set_title("Observed follow-up", color=INK, fontsize=11, loc="left")
 
    # --- C. multimorbidity count at entry (ordinal magnitude -> blue ramp) --
    cnt = wide[[f"prevalent_{e}" for e in dis]].sum(1)
    cnt = cnt.clip(upper=4)
    vals = cnt.value_counts().sort_index()
    pct = 100 * vals / vals.sum()
    labels = ["0", "1", "2", "3", "4+"][:len(vals)]
    ramp = [SEQ[1], SEQ[2], SEQ[3], SEQ[4], SEQ[5]][:len(vals)]
    bars = c.bar(labels, pct.values, color=ramp, edgecolor=SURFACE, linewidth=2)
    for bar, p in zip(bars, pct.values):                  # selective labels
        c.annotate(f"{p:.0f}%", (bar.get_x() + bar.get_width() / 2, p),
                   xytext=(0, 4), textcoords="offset points",
                   ha="center", color=INK2, fontsize=9)
    _style(c)
    c.set_xlabel("Conditions already present at entry", color=INK2, fontsize=9)
    c.set_ylabel("% of cohort", color=INK2, fontsize=9)
    c.set_title("Multimorbidity at entry", color=INK, fontsize=11, loc="left")
 
    # --- D. prevalent vs incident, by condition (2 series -> slots 1, 2) ---
    prev = np.array([100 * wide[f"prevalent_{e}"].mean() for e in dis])
    inc = np.array([100 * (wide.loc[wide[f"prevalent_{e}"] == 0,
                                    f"event_{e}"] == 1).mean() for e in dis])
    order = np.argsort(prev + inc)
    names = [dis[i] for i in order]
    y = np.arange(len(names))
    h = 0.38
    d.barh(y + h / 2, prev[order], height=h, color=SERIES[0],
           edgecolor=SURFACE, linewidth=2, label="Prevalent at entry")
    d.barh(y - h / 2, inc[order], height=h, color=SERIES[1],
           edgecolor=SURFACE, linewidth=2, label="Incident in follow-up")
    d.set_yticks(y, [n.replace("_", " ") for n in names])
    _style(d, xgrid=True, ygrid=False)
    d.set_xlabel("% of those at risk", color=INK2, fontsize=9)
    d.set_title("Disease burden", color=INK, fontsize=11, loc="left")
    d.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
 
    fig.tight_layout()
    return fig
 
 
def plot_burden_by_group(wide, event_types, group="wealth",
                         group_name="Wealth quintile", figsize=(11, 6)):
    """
    Prevalence at entry by an ordered group (wealth, education).
 
    An ordered group takes an ORDINAL ramp -- one hue, light to dark -- not
    categorical hues: the groups have a natural order and the ramp shows it.
    """
    import matplotlib.pyplot as plt
 
    dis = [e for e in event_types if e != "death"]
    g = wide.copy()
    q = pd.qcut(g[group], 5, labels=False, duplicates="drop")
    levels = sorted(pd.Series(q).dropna().unique())
    ramp = [SEQ[i] for i in np.linspace(1, 6, len(levels)).round().astype(int)]
 
    prev = np.array([[100 * g.loc[q == L, f"prevalent_{e}"].mean() for e in dis]
                     for L in levels])
    order = np.argsort(prev.mean(0))
    names = [dis[i] for i in order]
 
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    x = np.arange(len(names))
    w = 0.8 / len(levels)
    for i, L in enumerate(levels):
        ax.bar(x + (i - (len(levels) - 1) / 2) * w, prev[i][order], width=w,
               color=ramp[i], edgecolor=SURFACE, linewidth=1.5,
               label=f"{int(L)+1}" + (" (lowest)" if i == 0 else
                                      " (highest)" if i == len(levels) - 1 else ""))
    ax.set_xticks(x, [n.replace("_", " ") for n in names],
                  rotation=40, ha="right")
    _style(ax)
    ax.set_ylabel("% prevalent at entry", color=INK2, fontsize=9)
    ax.set_title(f"Disease prevalence at entry by {group_name.lower()}",
                 color=INK, fontsize=11, loc="left")
    ax.legend(title=group_name, frameon=False, fontsize=9, title_fontsize=9,
              labelcolor=INK2, ncol=len(levels))
    fig.tight_layout()
    return fig
 
 
def plot_cooccurrence(wide, event_types, figsize=(7.5, 6.5), vmax=None):
    """
    Phi between conditions PRESENT AT ENTRY -- the observed co-occurrence the
    models are later asked to reproduce.
 
    Correlation is polarity, so it takes a DIVERGING pair (blue <-> red) with a
    neutral gray midpoint: zero must read as "nothing", which a rainbow or a
    single hue cannot do.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
 
    dis = [e for e in event_types if e != "death"]
    P = wide[[f"prevalent_{e}" for e in dis]].to_numpy(float)
    M = np.corrcoef(P.T)                      # phi for binary = Pearson
    np.fill_diagonal(M, np.nan)
    v = vmax or float(np.nanmax(np.abs(M)))
 
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#184f95", "#6da7ec", "#f0efec", "#ef8a8a", "#b11c1c"])
 
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v)
    ax.set_xticks(range(len(dis)), [e.replace("_", " ") for e in dis],
                  rotation=40, ha="right")
    ax.set_yticks(range(len(dis)), [e.replace("_", " ") for e in dis])
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for i in range(len(dis)):
        for j in range(len(dis)):
            if i == j:
                continue
            ax.annotate(f"{M[i, j]:.2f}", (j, i), ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(M[i, j]) > 0.6 * v else INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(len(dis) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(dis) + 1) - .5, minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_title("Co-occurrence of conditions at study entry (phi)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig
 

##############################################################################
## ELSA: discrete-time multi-outcome survival transformer on an AGE axis    ##
##############################################################################
#
# Differences from the simulation pipeline:
#
#  1. `bin` indexes AGE, not study time. So self.pos is an AGE embedding and
#     alpha[k, t] is the age-specific baseline hazard -- an interpretable,
#     transportable quantity you can plot against published incidence.
#
#  2. LEFT TRUNCATION. People enter at different ages. Sequences are
#     LEFT-ALIGNED to index 0 and the age bin is passed separately as
#     `pos_idx`, so padding only ever occurs at the TAIL. Causal attention
#     means tail padding cannot leak backwards, so no key-padding mask is
#     needed and no softmax ever sees an empty key set (which is what makes
#     the naive "pad the start" approach emit NaNs).
#
#  3. The loss is driven by at_risk_<k>, not by "mask after first occurrence".
#     Prevalent conditions (diagnosed before entry) are never at risk but DO
#     appear as prev_<k> = 1 from the first row -- they inform other events
#     without contributing a likelihood term of their own.
#
#  4. DEATH ABSORBS. at_risk is already zeroed after death by the R script.
#     For the rollout it must also terminate the trajectory (see rollout_elsa).
#
# Input: elsa_long_agegrid.csv from elsa_prepare.R
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ============================================================== 1. ARRAYS
 
def build_arrays(csv_path, events, static_cols, n_bins=None, device="cpu"):
    """
    Returns a dict of tensors, sequences LEFT-ALIGNED at index 0.
 
      X        (n, T, p)  float   features; prev_* are at the START of the bin
      Y        (n, T, K)  float   1 if event k first diagnosed in this bin
      R        (n, T, K)  float   1 if at risk for k in this bin  (loss mask)
      pos_idx  (n, T)     long    AGE BIN of each row (0-based); 0 where padded
      obs      (n, T)     float   1 while under observation
      entry    (n,)       long    first age bin (0-based)
      length   (n,)       long    number of observed rows
      ids      (n,)              idauniq
 
    p = len(static_cols) + K   (prev_* for every event)
    Age is NOT a feature -- it is the position. Including it as a column too
    would be redundant and worsens the alpha/eta identification problem.
    """
    df = pd.read_csv(csv_path)
    K  = len(events)
 
    prev_cols = [f"prev_{e}"    for e in events]
    ev_cols   = [f"event_{e}"   for e in events]
    rk_cols   = [f"at_risk_{e}" for e in events]
    missing   = [c for c in static_cols + prev_cols + ev_cols + rk_cols
                 if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {csv_path}: {missing}")
 
    df = df.sort_values(["idauniq", "bin"]).reset_index(drop=True)
 
    if n_bins is None:
        n_bins = int(df["bin"].max())
    df["bin0"] = df["bin"].astype(int) - 1                 # 0-based age bin
 
    # impute static covariates with the training median / mode, per column
    for c in static_cols:
        if df[c].isna().any():
            fill = df[c].median() if df[c].dtype.kind in "fc" else df[c].mode()[0]
            df[c] = df[c].fillna(fill)
            print(f"  imputed {c}: {fill}")
 
    ids     = df["idauniq"].values
    uniq, first_idx, counts = np.unique(ids, return_index=True, return_counts=True)
    order   = np.argsort(first_idx)                        # preserve file order
    uniq, first_idx, counts = uniq[order], first_idx[order], counts[order]
    n, T    = len(uniq), int(counts.max())
    p       = len(static_cols) + K
 
    X       = np.zeros((n, T, p),  dtype=np.float32)
    Y       = np.zeros((n, T, K),  dtype=np.float32)
    R       = np.zeros((n, T, K),  dtype=np.float32)
    pos_idx = np.zeros((n, T),     dtype=np.int64)
    obs     = np.zeros((n, T),     dtype=np.float32)
    entry   = np.zeros(n,          dtype=np.int64)
 
    feat = np.concatenate([df[static_cols].values.astype(np.float32),
                           df[prev_cols].values.astype(np.float32)], axis=1)
    ymat = df[ev_cols].values.astype(np.float32)
    rmat = df[rk_cols].values.astype(np.float32)
    bmat = df["bin0"].values
 
    for i, (s, c) in enumerate(zip(first_idx, counts)):
        sl = slice(s, s + c)
        X[i, :c]       = feat[sl]
        Y[i, :c]       = ymat[sl]
        R[i, :c]       = rmat[sl]
        pos_idx[i, :c] = bmat[sl]
        obs[i, :c]     = 1.0
        entry[i]       = bmat[s]
 
    # a padded row must never contribute to the loss
    R = R * obs[:, :, None]
 
    t = lambda a, d=None: torch.as_tensor(a, dtype=d, device=device)
    out = dict(X=t(X, torch.float32), Y=t(Y, torch.float32), R=t(R, torch.float32),
               pos_idx=t(pos_idx, torch.long), obs=t(obs, torch.float32),
               entry=t(entry, torch.long), length=t(counts.copy(), torch.long),
               ids=uniq, n_bins=n_bins,
               feature_names=list(static_cols) + prev_cols, events=list(events))
 
    print(f"n={n}  T={T}  p={p}  K={K}  age bins={n_bins}")
    print(f"events in risk set: {int(R.sum())} person-bin-events at risk, "
          f"{int((Y * R).sum())} observed")
    stray = int((Y * (1 - R)).sum())
    if stray:
        print(f"  WARNING: {stray} events fire outside the risk set")
    return out
 
 
def split_by_person(arrays, frac=0.7, seed=0):
    """Train/test split on PEOPLE, never on person-bins."""
    n   = arrays["X"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut  = int(frac * n)
    def take(idx):
        d = {k: (v[idx] if torch.is_tensor(v) else
                 (v[idx] if isinstance(v, np.ndarray) else v))
             for k, v in arrays.items()}
        return d
    return take(perm[:cut]), take(perm[cut:])
 
 
# ============================================================== 2. MODEL
 
class ELSATransformer(nn.Module):
    """
    Causal transformer over a patient's age-indexed sequence of clinical states.
 
        logit h_k(t) = alpha[k, age_bin(t)] + eta_k(t)
 
    alpha : per-event, per-AGE-BIN baseline hazard, shared by everyone
    eta   : patient-specific score; covariates and accumulated diagnoses
            combined non-linearly by attention over the causal history
 
    Differs from SimpleTransformerTimeSeries in taking `pos_idx` -- the age bin
    of each row -- so that left-aligned, ragged-entry sequences are handled
    without start-padding.
    """
 
    def __init__(self, p, K, n_bins, d_model=64, nhead=4, nlayers=2,
                 dim_feedforward=None, dropout=0.1, center_eta=True):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
 
        self.proj = nn.Linear(p, d_model)
        self.pos  = nn.Embedding(n_bins, d_model)       # AGE-bin embedding
 
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head    = nn.Linear(d_model, K, bias=False)   # no bias: alpha is
        self.alpha   = nn.Parameter(torch.zeros(K, n_bins)) # the only intercept
 
        self.p, self.K, self.n_bins = p, K, n_bins
        self.center_eta = center_eta
 
        # running per-age-bin mean of eta, so that alpha is identified as the
        # baseline rather than absorbing an arbitrary share of eta's level.
        self.register_buffer("eta_mean", torch.zeros(n_bins, K))
        self.register_buffer("eta_seen", torch.zeros(n_bins))
        self.register_buffer("mu", torch.zeros(p))
        self.register_buffer("sd", torch.ones(p))
 
    # ---------------- scaling ----------------
    def set_scaler(self, X, obs):
        """Fit on TRAIN rows that are actually observed. Binary cols untouched."""
        flat = X[obs > 0]
        mu, sd = flat.mean(0), flat.std(0).clamp(min=1e-6)
        binary = ((flat == 0) | (flat == 1)).all(0)
        mu[binary], sd[binary] = 0.0, 1.0
        self.mu.copy_(mu); self.sd.copy_(sd)
        return self
 
    # ---------------- forward ----------------
    def get_eta(self, x, pos_idx):
        h = self.proj((x - self.mu) / self.sd) + self.pos(pos_idx)
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(h)                                  # (B, T, K)
 
    def forward(self, x, pos_idx, obs=None, update_center=False):
        eta = self.get_eta(x, pos_idx)
 
        if self.center_eta:
            if update_center and obs is not None:
                self._update_center(eta.detach(), pos_idx, obs)
            eta = eta - self.eta_mean[pos_idx]               # (B, T, K)
 
        return eta + self.alpha.T[pos_idx]                   # (B, T, K)
 
    @torch.no_grad()
    def _update_center(self, eta, pos_idx, obs, momentum=0.05):
        """EMA of eta's mean per age bin, over observed rows only."""
        m = obs > 0
        flat_pos, flat_eta = pos_idx[m], eta[m]
        for b in flat_pos.unique():
            sel = flat_pos == b
            self.eta_mean[b] = ((1 - momentum) * self.eta_mean[b]
                                + momentum * flat_eta[sel].mean(0))
            self.eta_seen[b] += 1
 
    def hazards(self, x, pos_idx):
        return torch.sigmoid(self.forward(x, pos_idx))
 
    def baseline_hazard(self):
        """alpha as probabilities: the age-specific baseline incidence."""
        return pd.DataFrame(torch.sigmoid(self.alpha).detach().cpu().numpy())
 
 
# ============================================================== 3. LOSS
 
def masked_bce(logits, Y, R):
    """
    Discrete-time survival likelihood. One Bernoulli term per (person, age bin,
    event) while AT RISK; everything else contributes nothing.
    """
    loss = F.binary_cross_entropy_with_logits(logits, Y, reduction="none")
    return (loss * R).sum() / R.sum().clamp(min=1.0)
 
 
def alpha_smoothness(alpha):
    """
    Second-difference penalty along the age axis. Keeps the baseline hazard
    from becoming ragged in thin age bins, and damps the sawtooth that digit
    heaping in self-reported diagnosis ages produces (arthritis especially).
    """
    d2 = alpha[:, 2:] - 2 * alpha[:, 1:-1] + alpha[:, :-2]
    return (d2 ** 2).mean()
 
 
@torch.no_grad()
def init_alpha(model, d, prior_strength=20.0):
    """
    Start alpha at the EMPIRICAL log-odds of each (event, age bin).
 
    Without this the model begins at hazard = 0.5 for everything, while the
    true rate is ~2%. It then has to learn the base rate through the encoder
    rather than through the intercept that exists for exactly that purpose --
    slow, and a common cause of divergence. Initialised here, training starts
    from the null model and only has to learn the deviations.
 
    Bins are shrunk toward each event's overall rate by `prior_strength`
    pseudo-observations, so thin age bins do not start at +-inf.
    """
    Y, R, P = d["Y"], d["R"], d["pos_idx"]
    K, T_bins = model.K, model.n_bins
 
    ev = torch.zeros(K, T_bins); at = torch.zeros(K, T_bins)
    for b in range(T_bins):
        m = (P == b)
        if not m.any():
            continue
        ev[:, b] = (Y * m[:, :, None]).sum((0, 1))
        at[:, b] = (R * m[:, :, None]).sum((0, 1))
 
    overall = (ev.sum(1) / at.sum(1).clamp(min=1.0)).clamp(1e-5, 1 - 1e-5)
    p = (ev + prior_strength * overall[:, None]) / (at + prior_strength)
    p = p.clamp(1e-5, 1 - 1e-5)
    model.alpha.copy_(torch.log(p / (1 - p)))
 
    print(f"alpha initialised: baseline hazard {p.min():.2e} - {p.max():.2e}, "
          f"overall rate {(ev.sum()/at.sum()):.4f}")
    return model
 
 
def null_loss(d):
    """BCE of a constant per-event rate. Any model must beat this."""
    Y, R = d["Y"], d["R"]
    p = (Y * R).sum((0, 1)) / R.sum((0, 1)).clamp(min=1.0)
    p = p.clamp(1e-6, 1 - 1e-6)
    ll = Y * torch.log(p) + (1 - Y) * torch.log(1 - p)
    return float(-(ll * R).sum() / R.sum())
 
 
def pos_weights(Y, R):
    """Per-event positive weighting. Event rates span 337 to 4445 here."""
    pos = (Y * R).sum((0, 1))
    neg = R.sum((0, 1)) - pos
    return (neg / pos.clamp(min=1.0)).clamp(1.0, 50.0)
 
 
# ============================================================== 4. TRAIN
 
def train_elsa_transformer(tr, te=None, d_model=64, nhead=4, nlayers=2,
                           dropout=0.1, lr=3e-4, weight_decay=1e-4,
                           epochs=80, batch_size=256, lam_smooth=1e-2,
                           use_pos_weight=False, center_eta=True,
                           warmup_frac=0.05, patience=12, seed=0, verbose=True):
    torch.manual_seed(seed)
    X, Y, R, P, O = tr["X"], tr["Y"], tr["R"], tr["pos_idx"], tr["obs"]
    n, T, p = X.shape
    K = Y.shape[2]
 
    model = ELSATransformer(p, K, tr["n_bins"], d_model, nhead, nlayers,
                            dropout=dropout, center_eta=center_eta)
    model.set_scaler(X, O)
    init_alpha(model, tr)                      # start from the null model
 
    nl_tr = null_loss(tr)
    print(f"null loss (constant per-event rate): train {nl_tr:.5f}"
          + (f"  test {null_loss(te):.5f}" if te is not None else ""))
 
    pw = pos_weights(Y, R) if use_pos_weight else None
 
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max(1, (n + batch_size - 1) // batch_size) * epochs
    warm  = max(1, int(warmup_frac * steps))
 
    # Short warmup then cosine decay. NOT OneCycle: its LR RISES for the first
    # 30% of total_steps, so an early stop lands mid-ramp with the LR still
    # climbing -- which is what made the loss diverge upward.
    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        t = (s - warm) / max(1, steps - warm)
        return 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
 
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm, tot = torch.randperm(n), 0.0
        for a in range(0, n, batch_size):
            idx = perm[a:a + batch_size]
            logits = model(X[idx], P[idx], O[idx], update_center=True)
 
            if pw is None:
                loss = masked_bce(logits, Y[idx], R[idx])
            else:
                l = F.binary_cross_entropy_with_logits(
                        logits, Y[idx], reduction="none",
                        pos_weight=pw)
                loss = (l * R[idx]).sum() / R[idx].sum().clamp(min=1.0)
 
            loss = loss + lam_smooth * alpha_smoothness(model.alpha)
 
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
 
        msg = (f"epoch {ep+1:3d}  lr {opt.param_groups[0]['lr']:.2e}"
               f"  train {tot/n:.5f}")
        if te is not None:
            model.eval()
            with torch.no_grad():
                vl = masked_bce(model(te["X"], te["pos_idx"]),
                                te["Y"], te["R"]).item()
            msg += f"  test {vl:.5f}"
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose: print(msg + "   [early stop]")
                    break
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(msg)
 
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
 
 
# ============================================================== 5. EVALUATE
 
@torch.no_grad()
def cindex_at_landmark(model, d, landmark_bin, horizon_bins, events,
                       min_events=20):
    """
    Discrimination for each event from a landmark AGE BIN.
 
    Eligible : under observation at `landmark_bin` AND still at risk for k.
    Outcome  : k first diagnosed within the next `horizon_bins`.
    Score    : cumulative hazard over the horizon with the history FROZEN at
               the landmark -- no rollout, so this measures ranking only.
               Absolute risk needs the Monte-Carlo rollout (see rollout_elsa).
    """
    from lifelines.utils import concordance_index
 
    X, Y, R, P, O = d["X"], d["Y"], d["R"], d["pos_idx"], d["obs"]
    n, T, _ = X.shape
 
    # row index of the landmark age bin, per person (-1 if not observed then)
    row = (P == landmark_bin).float().argmax(1)
    has = (P == landmark_bin).any(1) & (O.gather(1, row[:, None]).squeeze(1) > 0)
 
    logits = model(X, P)                                   # (n, T, K)
    haz    = torch.sigmoid(logits)
 
    out = []
    for k, ev in enumerate(events):
        lo = row
        hi = torch.clamp(row + horizon_bins, max=T - 1)
 
        # cumulative hazard over the horizon, frozen history
        score = torch.zeros(n)
        for i in range(n):
            if not has[i]:
                continue
            score[i] = haz[i, lo[i]:hi[i] + 1, k].sum()
 
        at_risk = R[torch.arange(n), row, k] > 0
        elig    = (has & at_risk).numpy()
        if elig.sum() < 50:
            continue
 
        # outcome and time, in bins from the landmark
        occurred = np.zeros(n, dtype=int)
        dur      = np.zeros(n, dtype=float)
        for i in range(n):
            if not elig[i]:
                continue
            w = Y[i, lo[i]:hi[i] + 1, k]
            obs_w = O[i, lo[i]:hi[i] + 1]
            if w.sum() > 0:
                occurred[i] = 1
                dur[i] = float(w.argmax())
            else:
                dur[i] = float(obs_w.sum())          # censored at last observed
 
        if occurred[elig].sum() < min_events:
            continue
        out.append({
            "event": ev,
            "n_eligible": int(elig.sum()),
            "n_events": int(occurred[elig].sum()),
            "cindex": concordance_index(dur[elig], -score.numpy()[elig],
                                        occurred[elig]),
        })
    return pd.DataFrame(out).set_index("event").round(3)
 
 
@torch.no_grad()
def plot_baseline_hazard(model, events, age_min=40, bin_width=2, ax=None):
    """alpha on the probability scale, against age. Sanity-check against
    published incidence curves -- this is the payoff of the age time scale."""
    import matplotlib.pyplot as plt
    a   = torch.sigmoid(model.alpha).cpu().numpy()          # (K, n_bins)
    age = age_min + (np.arange(a.shape[1]) + 0.5) * bin_width
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for k, ev in enumerate(events):
        ax.plot(age, a[k], label=ev, lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("age"); ax.set_ylabel(f"baseline hazard per {bin_width}y")
    ax.set_title("Age-specific baseline hazard (alpha)")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    return ax
 
 
# ============================================================== 6. ROLLOUT
 
@torch.no_grad()
def rollout_elsa(model, d, landmark_bin, n_steps, events, n_sims=100,
                 death_event="died", seed=0):
    """
    Monte-Carlo forward simulation from a landmark age bin.
 
    DEATH ABSORBS: once the death head fires, the trajectory stops and no
    further event can occur. Without this the model would keep generating
    diagnoses for dead people and every absolute risk would be overstated.
 
    Returns ever[n, n_sims, K]: 1 if event k occurred by the end of the horizon.
 
    NOTE: death here is UNDER-ASCERTAINED in ELSA (end-of-life interviews
    only), so the competing risk is only partly removed and absolute risks at
    older ages are biased UPWARD. Use for ranking and correlation structure;
    do not report absolute incidence without ONS mortality linkage.
    """
    g = torch.Generator().manual_seed(seed)
    X, P, O = d["X"], d["pos_idx"], d["obs"]
    n, T, p = X.shape
    K       = len(events)
    kd      = events.index(death_event) if death_event in events else None
    n_prev  = K                                   # prev_* are the last K cols
    i_prev0 = p - n_prev
 
    row = (P == landmark_bin).float().argmax(1)
 
    ever  = torch.zeros(n, n_sims, K)
    alive = torch.ones(n, n_sims, dtype=torch.bool)
 
    for s in range(n_sims):
        # history prefix up to and including the landmark
        L    = int(row.max().item()) + 1
        x    = X[:, :L].clone()
        pidx = P[:, :L].clone()
 
        for step in range(n_steps):
            logits = model(x, pidx)[:, -1, :]                 # (n, K)
            haz    = torch.sigmoid(logits)
            fired  = (torch.rand(haz.shape, generator=g) < haz).float()
 
            # cannot re-acquire what you already have
            fired = fired * (1.0 - x[:, -1, i_prev0:])
            # the dead acquire nothing
            fired = fired * alive[:, s][:, None].float()
 
            ever[:, s] = torch.clamp(ever[:, s] + fired, max=1.0)
            if kd is not None:
                alive[:, s] &= fired[:, kd] == 0
 
            # build the next row: advance the age bin, update prev_*
            nxt = x[:, -1:].clone()
            nxt[:, 0, i_prev0:] = torch.clamp(x[:, -1, i_prev0:] + fired, max=1.0)
            x    = torch.cat([x, nxt], dim=1)
            pidx = torch.cat([pidx, (pidx[:, -1:] + 1).clamp(max=model.n_bins - 1)],
                             dim=1)
 
            if not alive[:, s].any():
                break
 
    return ever
 


"""
ELSA-specific model definitions for the model progression:
 
1) Separate baseline Cox PH models
2) Deep multi-outcome Cox neural network
3) Simple multi-outcome discrete-time binary neural network
4) Causal discrete-time Transformer
 
Designed to work with prepare_elsa_models.py.
 
Key ELSA adaptation
-------------------
Prevalent disease is outcome-specific. A participant can therefore remain in
the risk set for diabetes even if arthritis was already present at entry.
The MultiCox loss below uses an outcome-specific `risk_mask` rather than
dropping anyone prevalent for any one of the other outcomes.
 
Death is an ordinary output channel for prediction but an absorbing terminal
state in the long-format data produced by prepare_elsa_models.py.
 
Baseline-hazard initialisation
------------------------------
Both discrete-time models carry a free per-event, per-interval intercept
alpha[k, t]. Left at zero it starts every hazard at 0.5 while the true rate is
around 2%, so the network has to learn the base rate through its trunk instead
of through the intercept that exists for the purpose -- slow, and a common
cause of the loss rising rather than falling. `init_alpha_from_data` starts it
at the empirical log-odds, shrunk towards each event's overall rate so thin
intervals do not start at +-inf. `null_loss` prints the floor any model must
beat.
"""
 
from __future__ import annotations
 
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
 
class FeatureStandardizer:
    """Train-set-only standardisation; binary columns are left unchanged."""
 
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
 
    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        binary = np.array([
            np.isin(np.unique(col[~np.isnan(col)]), [0.0, 1.0]).all()
            for col in x.T
        ])
        mean[binary] = 0.0
        scale[binary] = 1.0
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        self.mean_ = mean.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self
 
    def transform(self, x):
        if self.mean_ is None:
            raise RuntimeError("FeatureStandardizer has not been fitted.")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.scale_
 
    def fit_transform(self, x):
        return self.fit(x).transform(x)
 
 
def concordance_index(time, risk, event):
    """Harrell C-index; larger `risk` means earlier event / higher risk.
 
    Backed by lifelines (C implementation) -- the equivalent pure-Python
    double loop takes minutes at ~4k test people x 12 outcomes. lifelines
    orders by PREDICTED SURVIVAL, so the risk score is negated.
    """
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    event = np.asarray(event, dtype=int)
 
    ok = np.isfinite(time) & np.isfinite(risk) & np.isfinite(event)
    time, risk, event = time[ok], risk[ok], event[ok]
    if len(time) == 0 or event.sum() == 0:
        return np.nan
 
    try:
        from lifelines.utils import concordance_index as _ci
        return float(_ci(time, -risk, event))
    except ImportError:
        return _concordance_index_python(time, risk, event)
 
 
def _concordance_index_python(time, risk, event):
    """Reference implementation. O(n^2); kept only as a fallback."""
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        js = np.where(time > time[i])[0]
        if len(js) == 0:
            continue
        comparable += len(js)
        concordant += np.sum(risk[i] > risk[js])
        concordant += 0.5 * np.sum(risk[i] == risk[js])
    return np.nan if comparable == 0 else concordant / comparable
 
 
def split_ids(ids, test_size=0.25, seed=42):
    """Patient-level train/test split."""
    ids = np.asarray(pd.unique(ids))
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    n_test = max(1, int(round(len(ids) * test_size)))
    test_ids = ids[:n_test]
    train_ids = ids[n_test:]
    return train_ids, test_ids
 
 
def cindex_table_from_scores(
    wide: pd.DataFrame,
    scores: np.ndarray,
    event_types: Sequence[str],
    ids: Optional[Sequence] = None,
    model_name: str = "model",
):
    """Outcome-specific C-index, excluding prevalent cases for that outcome."""
    d = wide.copy()
    if ids is not None:
        d = d.set_index("id").loc[list(ids)].reset_index()
 
    scores = np.asarray(scores)
    if scores.shape != (len(d), len(event_types)):
        raise ValueError(
            f"scores shape {scores.shape}, expected {(len(d), len(event_types))}"
        )
 
    rows = []
    for k, e in enumerate(event_types):
        at_risk = d[f"prevalent_{e}"].to_numpy() == 0
        rows.append({
            "event": e,
            "model": model_name,
            "n_test_at_risk": int(at_risk.sum()),
            "n_test_events": int(d.loc[at_risk, f"event_{e}"].sum()),
            "c_index": concordance_index(
                d.loc[at_risk, f"time_{e}"],
                scores[at_risk, k],
                d.loc[at_risk, f"event_{e}"],
            ),
        })
    return pd.DataFrame(rows)
 
 
# ---------------------------------------------------------------------------
# Baseline-hazard initialisation (shared by models 3 and 4)
# ---------------------------------------------------------------------------
 
@torch.no_grad()
def init_alpha_from_data(alpha: nn.Parameter, ev: torch.Tensor, at: torch.Tensor,
                         prior_strength: float = 20.0, label: str = "alpha"):
    """Set alpha[k, t] to the empirical log-odds of event k in interval t.
 
    ev, at : (K, n_intervals) event and at-risk counts.
 
    Counts are shrunk towards each event's overall rate by `prior_strength`
    pseudo-observations so that thin intervals do not start at +-inf.
    """
    overall = (ev.sum(1) / at.sum(1).clamp(min=1.0)).clamp(1e-5, 1 - 1e-5)
    p = ((ev + prior_strength * overall[:, None]) / (at + prior_strength))
    p = p.clamp(1e-5, 1 - 1e-5)
    alpha.copy_(torch.log(p / (1 - p)))
    print(f"{label} initialised: baseline hazard {p.min():.2e} - {p.max():.2e}, "
          f"overall rate {(ev.sum() / at.sum().clamp(min=1.0)):.4f}")
    return alpha
 
 
def _counts_from_long(events: torch.Tensor, masks: torch.Tensor,
                      interval: torch.Tensor, n_intervals: int):
    """(K, n_intervals) event / at-risk counts from a flat long table."""
    K = events.shape[1]
    ev = torch.zeros(K, n_intervals)
    at = torch.zeros(K, n_intervals)
    for t in range(n_intervals):
        sel = interval == t
        if sel.any():
            ev[:, t] = (events[sel] * masks[sel]).sum(0)
            at[:, t] = masks[sel].sum(0)
    return ev, at
 
 
def _counts_from_sequence(events: torch.Tensor, masks: torch.Tensor):
    """(K, T) counts from (n, T, K) tensors."""
    ev = (events * masks).sum(0).T.contiguous()
    at = masks.sum(0).T.contiguous()
    return ev, at
 
 
def null_loss(events: torch.Tensor, masks: torch.Tensor) -> float:
    """Masked BCE of a constant per-event rate. Any model must beat this."""
    flat_e = events.reshape(-1, events.shape[-1])
    flat_m = masks.reshape(-1, masks.shape[-1])
    p = (flat_e * flat_m).sum(0) / flat_m.sum(0).clamp(min=1.0)
    p = p.clamp(1e-6, 1 - 1e-6)
    ll = flat_e * torch.log(p) + (1 - flat_e) * torch.log(1 - p)
    return float(-(ll * flat_m).sum() / flat_m.sum().clamp(min=1.0))
 
 
# ---------------------------------------------------------------------------
# 1) Separate Cox PH
# ---------------------------------------------------------------------------
 
def fit_separate_coxph(
    train_wide: pd.DataFrame,
    covariates: Sequence[str],
    event_types: Sequence[str],
):
    """Fit one lifelines CoxPHFitter per outcome using baseline covariates only."""
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError(
            "lifelines is required for the Basic Cox analysis. "
            "Install with: pip install lifelines"
        ) from exc
 
    models = {}
    for e in event_types:
        cols = list(covariates) + [f"time_{e}", f"event_{e}", f"prevalent_{e}"]
        d = train_wide[cols].copy()
        d = d.loc[d[f"prevalent_{e}"] == 0].drop(columns=f"prevalent_{e}")
        d = d.dropna()
        d = d.rename(columns={f"time_{e}": "_time", f"event_{e}": "_event"})
 
        cph = CoxPHFitter()
        cph.fit(d, duration_col="_time", event_col="_event")
        models[e] = cph
    return models
 
 
def score_separate_coxph(
    models: Dict[str, object],
    test_wide: pd.DataFrame,
    covariates: Sequence[str],
    event_types: Sequence[str],
):
    scores = np.zeros((len(test_wide), len(event_types)), dtype=np.float32)
    for k, e in enumerate(event_types):
        scores[:, k] = np.asarray(
            models[e].predict_log_partial_hazard(test_wide[list(covariates)])
        ).reshape(-1)
    return scores
 
 
def cox_coefficient_table(models: Dict[str, object]):
    rows = []
    for event, model in models.items():
        s = model.summary.reset_index()
        cov_col = s.columns[0]
        for _, r in s.iterrows():
            rows.append({
                "event": event,
                "covariate": r[cov_col],
                "coef": r["coef"],
                "hazard_ratio": np.exp(r["coef"]),
                "se": r.get("se(coef)", np.nan),
                "p": r.get("p", np.nan),
            })
    return pd.DataFrame(rows)
 
 
# ---------------------------------------------------------------------------
# 2) Deep MultiCox NN with outcome-specific prevalence mask
# ---------------------------------------------------------------------------
 
def cox_partial_loss_masked(eta, time, event, eligible):
    """Cox partial loss for one outcome with an outcome-specific eligible set."""
    eligible = eligible.bool()
    event = event.bool() & eligible
 
    if event.sum() == 0:
        return eta.sum() * 0.0
 
    idx = torch.where(eligible)[0]
    eta_e = eta[idx]
    time_e = time[idx]
    event_e = event[idx]
 
    order = torch.argsort(time_e, descending=True)
    eta_e = eta_e[order]
    event_e = event_e[order]
 
    log_risk = torch.logcumsumexp(eta_e, dim=0)
    pll = eta_e[event_e] - log_risk[event_e]
    # Mean per event makes outcomes with different event counts comparable.
    return -pll.mean()
 
 
class MultiCoxNN(nn.Module):
    def __init__(self, p, K, hidden_dims=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        in_dim = p
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, K, bias=False))
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
 
 
def fit_multicox_nn(
    train_wide: pd.DataFrame,
    covariates: Sequence[str],
    event_types: Sequence[str],
    hidden_dims=(64, 32),
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=500,
    verbose_every=50,
    seed=42,
):
    """Pass the FULL wide table here.
 
    Eligibility is built per outcome from the prevalent_* columns, so a person
    prevalent for arthritis still contributes to the diabetes risk set. Do not
    pre-filter with make_multicox_nn(prevalent_policy='drop_any') -- in ELSA
    that discards most of the cohort.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    x_np = train_wide[list(covariates)].to_numpy(dtype=np.float32)
    scaler = FeatureStandardizer()
    x = torch.tensor(scaler.fit_transform(x_np), dtype=torch.float32)
 
    time = torch.tensor(
        train_wide[[f"time_{e}" for e in event_types]].to_numpy(dtype=np.float32)
    )
    event = torch.tensor(
        train_wide[[f"event_{e}" for e in event_types]].to_numpy(dtype=np.float32)
    )
    eligible = torch.tensor(
        (1 - train_wide[[f"prevalent_{e}" for e in event_types]]
         .to_numpy(dtype=np.float32))
    )
 
    n_elig = eligible.sum(0)
    print("MultiCox eligible per outcome: "
          + ", ".join(f"{e}={int(n)}" for e, n in zip(event_types, n_elig)))
 
    model = MultiCoxNN(
        p=len(covariates), K=len(event_types),
        hidden_dims=hidden_dims, dropout=dropout
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        eta = model(x)
        losses = [
            cox_partial_loss_masked(
                eta[:, k], time[:, k], event[:, k], eligible[:, k]
            )
            for k in range(len(event_types))
        ]
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        history.append(float(loss.detach()))
 
        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            print(f"MultiCox epoch {epoch+1:4d}/{epochs}: loss={history[-1]:.4f}")
 
    return model, scaler, history
 
 
def score_multicox_nn(model, scaler, wide, covariates):
    model.eval()
    x = torch.tensor(
        scaler.transform(wide[list(covariates)].to_numpy(dtype=np.float32)),
        dtype=torch.float32,
    )
    with torch.no_grad():
        return model(x).cpu().numpy()
 
 
# ---------------------------------------------------------------------------
# 3) Simple multi-outcome binary time series
# ---------------------------------------------------------------------------
 
class SimpleBinaryTimeSeries(nn.Module):
    """Baseline covariates + event-specific interval intercepts."""
 
    def __init__(self, p, K, n_intervals, hidden_dims=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        in_dim = p
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, K, bias=False))
        self.net = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.zeros(K, n_intervals))
        self.K = K
        self.n_intervals = n_intervals
 
    def get_eta(self, x):
        return self.net(x)
 
    def forward(self, x, interval):
        eta = self.get_eta(x)
        k = torch.arange(self.K, device=x.device)[None, :].expand(len(x), -1)
        t = interval[:, None].expand(-1, self.K)
        return eta + self.alpha[k, t]
 
 
def fit_simple_binary(
    train_long: pd.DataFrame,
    features: Sequence[str],
    event_types: Sequence[str],
    hidden_dims=(64, 32),
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=150,
    batch_size=1024,
    verbose_every=20,
    init_alpha=True,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    scaler = FeatureStandardizer()
    x_np = train_long[list(features)].to_numpy(dtype=np.float32)
    x = torch.tensor(scaler.fit_transform(x_np), dtype=torch.float32)
    interval = torch.tensor(train_long["interval"].to_numpy(), dtype=torch.long)
    events = torch.tensor(
        train_long[[f"event_{e}" for e in event_types]].to_numpy(dtype=np.float32)
    )
    masks = torch.tensor(
        train_long[[f"mask_{e}" for e in event_types]].to_numpy(dtype=np.float32)
    )
 
    n_intervals = int(train_long["interval"].max()) + 1
    model = SimpleBinaryTimeSeries(
        p=len(features), K=len(event_types), n_intervals=n_intervals,
        hidden_dims=hidden_dims, dropout=dropout
    )
 
    if init_alpha:
        ev, at = _counts_from_long(events, masks, interval, n_intervals)
        init_alpha_from_data(model.alpha, ev, at, label="Binary TS alpha")
    print(f"Binary TS null loss (constant per-event rate): "
          f"{null_loss(events, masks):.5f}")
 
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
 
    N = len(train_long)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        total, denom = 0.0, 0.0
 
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(x[idx], interval[idx])
            raw = F.binary_cross_entropy_with_logits(
                logits, events[idx], reduction="none"
            )
            m = masks[idx]
            loss = (raw * m).sum() / m.sum().clamp(min=1.0)
 
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
 
            w = float(m.sum())
            total += float(loss.detach()) * w
            denom += w
 
        history.append(total / max(denom, 1.0))
        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            print(f"Binary TS epoch {epoch+1:4d}/{epochs}: loss={history[-1]:.4f}")
 
    return model, scaler, history
 
 
def score_simple_binary_baseline(model, scaler, wide, features):
    """Baseline risk score eta, directly comparable to Cox ranking."""
    model.eval()
    x = torch.tensor(
        scaler.transform(wide[list(features)].to_numpy(dtype=np.float32)),
        dtype=torch.float32,
    )
    with torch.no_grad():
        return model.get_eta(x).cpu().numpy()
 
 
# ---------------------------------------------------------------------------
# 4) Causal discrete-time Transformer
# ---------------------------------------------------------------------------
 
class CausalSurvivalTransformer(nn.Module):
    """
    Multi-outcome discrete-time survival Transformer.
 
    At interval t it sees only the feature state at intervals <= t. The
    prepare_elsa_models history columns are themselves lagged, so prev_* at t
    contains only diagnoses known before interval t.
    """
 
    def __init__(
        self, p, K, n_intervals, d_model=64, nhead=2, nlayers=2,
        dim_feedforward=None, dropout=0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
 
        self.proj = nn.Linear(p, d_model)
        self.pos = nn.Embedding(n_intervals, d_model)
 
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, K, bias=False)
        self.alpha = nn.Parameter(torch.zeros(K, n_intervals))
 
        self.p = p
        self.K = K
        self.n_intervals = n_intervals
 
        self.register_buffer("mu", torch.zeros(p))
        self.register_buffer("sd", torch.ones(p))
 
    def set_scaler(self, X):
        flat = X.reshape(-1, X.shape[-1])
        mu = flat.mean(0)
        sd = flat.std(0).clamp(min=1e-6)
        is_binary = ((flat == 0) | (flat == 1)).all(0)
        mu[is_binary] = 0.0
        sd[is_binary] = 1.0
        self.mu.copy_(mu)
        self.sd.copy_(sd)
        return self
 
    def _scale(self, x):
        return (x - self.mu) / self.sd
 
    def get_eta(self, x):
        B, T, _ = x.shape
        h = self.proj(self._scale(x))
        h = h + self.pos(torch.arange(T, device=x.device))[None, :, :]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device
        )
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        return self.head(h)
 
    def forward(self, x):
        eta = self.get_eta(x)
        T = x.shape[1]
        return eta + self.alpha[:, :T].T[None, :, :]
 
 
def long_to_sequence_tensors(
    df_long: pd.DataFrame,
    features: Sequence[str],
    event_types: Sequence[str],
):
    df = df_long.sort_values(["id", "interval"]).reset_index(drop=True)
    ids = df["id"].drop_duplicates().to_numpy()
    n = len(ids)
    T = df["interval"].nunique()
 
    sizes = df.groupby("id", sort=False).size().to_numpy()
    if len(df) != n * T or not np.all(sizes == T):
        raise ValueError("Transformer requires a rectangular patient x interval table.")
 
    X = torch.tensor(
        df[list(features)].to_numpy(dtype=np.float32),
        dtype=torch.float32
    ).view(n, T, len(features))
    events = torch.tensor(
        df[[f"event_{e}" for e in event_types]].to_numpy(dtype=np.float32),
        dtype=torch.float32
    ).view(n, T, len(event_types))
    masks = torch.tensor(
        df[[f"mask_{e}" for e in event_types]].to_numpy(dtype=np.float32),
        dtype=torch.float32
    ).view(n, T, len(event_types))
    return X, events, masks, ids
 
 
def fit_transformer(
    train_long: pd.DataFrame,
    features: Sequence[str],
    event_types: Sequence[str],
    d_model=64,
    nhead=4,
    nlayers=2,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=100,
    batch_size=128,
    verbose_every=10,
    init_alpha=True,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    X, events, masks, ids = long_to_sequence_tensors(
        train_long, features, event_types
    )
    n, T, p = X.shape
 
    model = CausalSurvivalTransformer(
        p=p, K=len(event_types), n_intervals=T,
        d_model=d_model, nhead=nhead, nlayers=nlayers, dropout=dropout
    )
    model.set_scaler(X)
 
    if init_alpha:
        ev, at = _counts_from_sequence(events, masks)
        init_alpha_from_data(model.alpha, ev, at, label="Transformer alpha")
    print(f"Transformer null loss (constant per-event rate): "
          f"{null_loss(events, masks):.5f}")
 
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
 
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total, denom = 0.0, 0.0
 
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X[idx])
            raw = F.binary_cross_entropy_with_logits(
                logits, events[idx], reduction="none"
            )
            m = masks[idx]
            loss = (raw * m).sum() / m.sum().clamp(min=1.0)
 
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
 
            w = float(m.sum())
            total += float(loss.detach()) * w
            denom += w
 
        history.append(total / max(denom, 1.0))
        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            print(f"Transformer epoch {epoch+1:4d}/{epochs}: loss={history[-1]:.4f}")
 
    check_transformer_causality(model, X)
    return model, history
 
 
def check_transformer_causality(model, X, t_break=None, atol=1e-5):
    model.eval()
    T = X.shape[1]
    if T < 2:
        return True
    t_break = T // 2 if t_break is None else int(t_break)
 
    with torch.no_grad():
        x1 = X[:1].clone()
        y1 = model.get_eta(x1)
        x2 = x1.clone()
        x2[:, t_break:, :] += 10.0
        y2 = model.get_eta(x2)
 
    if not torch.allclose(y1[:, :t_break], y2[:, :t_break], atol=atol):
        raise AssertionError("Causal leakage detected: future inputs changed earlier logits.")
    print(f"Causality check passed: intervals >= {t_break} do not affect earlier logits.")
    return True
 
 
def score_transformer_at_baseline(model, test_long, features, event_types):
    """
    Scores at interval 0. This is the fair baseline-prediction comparison:
    Transformer has baseline prevalent-history features but no future diagnoses.
    """
    X, _, _, ids = long_to_sequence_tensors(test_long, features, event_types)
    model.eval()
    with torch.no_grad():
        eta = model.get_eta(X)[:, 0, :].cpu().numpy()
    return eta, ids
 
 
def score_transformer_by_interval(model, test_long, features, event_types):
    """Return eta for every patient, interval and outcome."""
    X, events, masks, ids = long_to_sequence_tensors(
        test_long, features, event_types
    )
    model.eval()
    with torch.no_grad():
        eta = model.get_eta(X).cpu().numpy()
    return eta, events.numpy(), masks.numpy(), ids
 
 
def dynamic_cindex_table(
    model,
    test_long: pd.DataFrame,
    features: Sequence[str],
    event_types: Sequence[str],
):
    """
    Interval-specific discrimination among people at risk at the start of each
    interval. This is descriptive dynamic discrimination, not a causal effect.
    """
    from sklearn.metrics import roc_auc_score
 
    eta, events, masks, ids = score_transformer_by_interval(
        model, test_long, features, event_types
    )
    T = eta.shape[1]
 
    rows = []
    for t in range(T):
        for k, e in enumerate(event_types):
            m = masks[:, t, k] == 1
            y = events[:, t, k]
            # Within a single interval all event/censor times are tied, so use
            # binary rank discrimination (AUC) when both classes exist.
            if m.sum() == 0 or len(np.unique(y[m])) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y[m], eta[m, t, k])
            rows.append({
                "interval": t,
                "event": e,
                "n_at_risk": int(m.sum()),
                "n_events": int(y[m].sum()),
                "auc": auc,
            })
    return pd.DataFrame(rows)
 


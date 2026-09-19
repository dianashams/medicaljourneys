
import numpy as np
import pandas as pd
import random
# load et simulation function from simulate_population.py: 
from simulate_population import sim_population
import torch
import torch.nn as nn
import torch.nn.functional as F


from lifelines.utils import concordance_index
def get_cindex_for_event(predictions, df , event ):
    time_col = f"time_{event}"
    risk_col = f"event_{event}"
    c = concordance_index(df[time_col], predictions, df[risk_col])
    return(c)

##############################################
## 1) SIMPLE COX ##
##############################################

from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv

#def simplecox(df, covariate_cols = ["age_start", "bmi", "hyp", "smoke", "sex", "eth1", "eth2"], event_type = "a"):
#    time_col = f"time_{event_type}"
#    event_col = f"event_{event_type}"
#    cph = CoxPHFitter()
#    cph.fit(df[[time_col, event_col] + covariate_cols], duration_col= time_col, event_col=event_col)
#    #s=cph.summary[['coef', 'se(coef)', 'p']]
#    #beta_cox = cph.params_.values
#    return (cph)

def simplecox(df, covariate_cols = ["age_start", "bmi", "hyp", "smoke", "sex", "eth1", "eth2"], event_type = "a"):
    # https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html
    time_col = f"time_{event_type}"
    event_col = f"event_{event_type}"
    cph = CoxPHSurvivalAnalysis()
    cph.fit(df[covariate_cols], Surv.from_dataframe(event_col, time_col,df[[time_col, event_col]]))
    #beta_cox = pd.Series(cox0.coef_, index=covariate_cols)
    return (cph)
    
##############################################
## 2) CoxNN ##
##############################################

class CoxNN(nn.Module):
    def __init__(self, p, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        in_dim = p
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # scalar risk score
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape (n,)

#### Cox PH partial log-likelihood
def cox_partial_loglik(eta, time, event):
    """ eta: (n,) risk scores;     time: (n,) observed times;     event: (n,) 1 if event, 0 if censored """
    # sort by decreasing time
    order = torch.argsort(time, descending=True)
    eta = eta[order]
    event = event[order]
    # log cumulative sum of exp(eta)
    log_cumsum_exp = torch.logcumsumexp(eta, dim=0)
    # 3. contribution only from observed events
    loglik = eta[event == 1] - log_cumsum_exp[event == 1]
    return -loglik.sum()

#### Training CoxNN
def train_cox(x, time, event, hidden_dims = (), epochs = 10):
    p = len(x[0])
    model = CoxNN(p=p, hidden_dims = hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   
    for epoch in range(epochs):
        optimizer.zero_grad()
        eta = model(x)                     # risk scores
        loss = cox_partial_loglik(eta, time, event)
        loss.backward()
        optimizer.step()
        if ((epoch % 200 == 0)|(epoch +1 == epochs)):
            print(f"Epoch {epoch}, loss = {loss.item():.4f}")
    return (model)

#### prepare the data for CoxNN
def prepare_data_for_cox(df, 
                         covariate_cols = ["age_start", "bmi", "hyp", "smoke", "sex", "eth1", "eth2"], 
                         event_type = "a"):
    time_col = f"time_{event_type}"
    event_col = f"event_{event_type}"
    x = torch.tensor( df[covariate_cols].values, dtype=torch.float32)
    time = torch.tensor( df[time_col].values, dtype=torch.float32)
    event = torch.tensor(df[event_col].values, dtype=torch.float32)
    return (x, time, event)

##############################################
## 3) Multi Outcome CoxNN (MultiCox) ##
##############################################

#By default, nn.Linear(in_dim, K) includes a bias term (one per outcome).
#Just like in the single-outcome Cox case:
#the bias cancels out in the partial likelihood
#but it’s cleaner and more identifiable to remove it

#### Class definition for MultiCox
# K - number of events (K outcomes)
class MultiCoxNN(nn.Module):
    def __init__(self, p, hidden_dims=(64, 32), K=5):
        super().__init__()
        layers = []
        in_dim = p
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, K,bias=False))  #  risk score for each of the K outcomes
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)  # shape (n,k)

#### prepare data  for MultiCox
def prepare_data_for_multicox(df, 
                         covariate_cols = ["age_start", "bmi", "hyp", "smoke", "sex", "eth1", "eth2"], 
			time_cols = ["time_a", "time_b", "time_c", "time_d","time_e"],
			event_cols = ["event_a", "event_b", "event_c", "event_d","event_e"]):
	x2 = torch.tensor( df[covariate_cols].values, dtype=torch.float32)
	time2 = torch.tensor( df[time_cols].values, dtype=torch.float32)
	event2 = torch.tensor(df[event_cols].values, dtype=torch.float32)
	return (x2, time2, event2)

#### training function for MultiCox
def train_coxmulti(x2, time2, event2, hidden_dims = (), epochs = 300, K=5, lr = 0.01):
    p = len(x2[0])
    # p -number of initial params, len(covariate_cols)
    model = MultiCoxNN(p=p, K=K, hidden_dims = hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   
    for epoch in range(epochs):
        optimizer.zero_grad()
        eta2 = model(x2)                     # risk scores
        loss = 0
        for k in range(K):
            loss += cox_partial_loglik(
            eta2[:, k],
            time2[:, k],
            event2[:, k])
        loss = loss / K
        loss.backward()
        optimizer.step()
        if ((epoch % 50 == 0)|(epoch +1 == epochs)):
            print(f"Epoch {epoch}, loss = {loss.item():.4f}")
    return (model)

##############################################
## 4) DISCREET TIME BINARY ##
##############################################

#### Class definition
class DiscreteTimeNN(nn.Module):
    def __init__(self, p, n_intervals, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        in_dim = p
        # hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        # output = eta (scalar)
        layers.append(nn.Linear(in_dim, 1, bias=False))
        self.net = nn.Sequential(*layers)
        # baseline hazard
        self.alpha = nn.Parameter(torch.zeros(n_intervals))
        
    def forward(self, x, interval_idx):
        eta = self.net(x).squeeze()
        logit = eta + self.alpha[interval_idx]
        return logit
        
    def get_eta(self, x):
        """Extract f(x)"""
        return self.net(x).squeeze()
        
    def predict_survival(self, x):
        """Predict survival probabilities for all intervals"""
        # eta: (n,)
        eta = self.net(x).squeeze()   # or self.beta(x) if linear
        # expand to (n, n_intervals)
        eta = eta.unsqueeze(1)  # (n, 1)
        alpha = self.alpha.unsqueeze(0)  # (1, n_intervals)
        logits = eta + alpha  # (n, n_intervals)
        hazards = torch.sigmoid(logits)  # (n, n_intervals)
        survival_probs = torch.cumprod(1 - hazards, dim=1)
        return survival_probs
        
    def return_logits(self, x):
        # eta: (n,)
        eta = self.net(x).squeeze()   # or self.beta(x) if linear
        # expand to (n, n_intervals)
        eta = eta.unsqueeze(1)  # (n, 1)
        alpha = self.alpha.unsqueeze(0)  # (1, n_intervals)
        logits = eta + alpha  # (n, n_intervals)
        return logits
        
    def return_hazards(self, x):
        eta = self.net(x).squeeze()   # or self.beta(x) if linear
        eta = eta.unsqueeze(1)  # (n, 1)
        alpha = self.alpha.unsqueeze(0)  # (1, n_intervals)
        logits = eta + alpha  # (n, n_intervals)
        hazards = torch.sigmoid(logits)  # (n, n_intervals)
        return hazards

    def return_etas(self, x):
        eta = self.net(x).squeeze()   # or self.beta(x) if linear
        return  eta.unsqueeze(1)

#### Prepare data for BINARY model, for a specific event 

def prepare_data_for_event(df, event_type, features, n_intervals=50, even_split=False, event_ratio=0.8):
    events = df[f'event_{event_type}'].values
    times = df[f'time_{event_type}'].values
    X = df[features].values
    if even_split:
        # equal-width bins
        max_time = times.max()
        interval_width = max_time / n_intervals
        time_intervals = np.floor(times / interval_width).astype(int)
        time_intervals = np.clip(time_intervals, 0, n_intervals - 1)
    else:
        # split times
        event_times = np.sort(times[events == 1])
        cens_times  = np.sort(times[events == 0])
        # number of cuts
        n_event = int(n_intervals * event_ratio)
        n_cens  = n_intervals - n_event
        # quantile-based cuts
        event_cuts = np.quantile(event_times, np.linspace(0, 1, n_event + 2)[1:-1])
        cens_cuts  = np.quantile(cens_times,  np.linspace(0, 1, n_cens + 2)[1:-1])
        # combine + clean
        cut_points = np.sort(np.concatenate([event_cuts, cens_cuts]))
        cut_points = np.unique(np.round(cut_points, 6))
        # ensure correct number of intervals
        # (digitize creates len(cuts)+1 bins)
        if len(cut_points) > n_intervals - 1:
            cut_points = cut_points[:n_intervals - 1]
        # map times → interval indices
        time_intervals = np.digitize(times, bins=cut_points, right=True)
        # ensure bounds
        time_intervals = np.clip(time_intervals, 0, n_intervals - 1)
    return X, time_intervals, events, n_intervals

#### Training function for BINARY
def train_event_model(df, event_type, features, lr=0.002, epochs=100, batch_size=512, n_intervals=20, hidden_dims= ()):
    """Train a model for a specific event type"""
    # Prepare data
    X, time_intervals, events, n_intervals = prepare_data_for_event(df, event_type, features, n_intervals)
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    intervals_tensor = torch.LongTensor(time_intervals)
    events_tensor = torch.FloatTensor(events)
    # Initialize model
    model = DiscreteTimeNN(len(features), n_intervals, hidden_dims = hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Training loop
    n_samples = X_tensor.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        # Shuffle data
        indices = torch.randperm(n_samples)
        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            X_batch = X_tensor[batch_indices]
            intervals_batch = intervals_tensor[batch_indices]
            events_batch = events_tensor[batch_indices]
            # Forward pass
            logits = model(X_batch, intervals_batch)
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(logits, events_batch)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_indices)
        avg_loss = total_loss / n_samples
        if ((epoch % 50 == 0) |(epoch +1 == epochs)):
            print(f'Event {event_type} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    return model

##############################################
## 5) DISCREET TIME MULTIBINARY ##
##############################################

# define multibinary
class MultiDiscreteTimeNN(nn.Module):
    def __init__(self, p, n_intervals, K, hidden_dims=()):
        super().__init__()
        # -------- feature network (eta) --------
        layers = []
        in_dim = p
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, K, bias=False))  # (n, K)
        self.net = nn.Sequential(*layers)
        # -------- baseline hazards:  
        # alpha[k, t] = baseline logit for outcome k at interval t
        self.alpha = nn.Parameter(torch.zeros(K, n_intervals))

    def forward(self, x, interval_idx):
        """       x:            (n, p)        interval_idx: (n, K)   integers in [0, n_intervals-1]         """
        eta = self.net(x) # (n, K)
        n, K = interval_idx.shape
        # create outcome indices: (n, K)
        k_idx = torch.arange(K, device=x.device).unsqueeze(0).expand(n, K)
        # safe lookup: alpha[k, t_ik]
        alpha_k = self.alpha[k_idx, interval_idx]  # (n, K)
        logits = eta + alpha_k
        return logits

    def get_eta(self, x):
        """Return linear predictors (no baseline)"""
        return self.net(x)

# prepare the data for multibinary
def prepare_data_for_multibinary (df, features, event_types = ["a","b","c","d","e"], 
                                  n_intervals=50, even_split=False, event_ratio=0.8):
    time_cols  = [f"time_{e}" for e in event_types]
    event_cols = [f"event_{e}" for e in event_types]
    times  = df[time_cols].values        # (n, K)
    events = df[event_cols].values       # (n, K)
    K = len(event_types)
    time_intervals = np.zeros_like(times, dtype=int)
    for kk in range(K):
        _, ti_k, _, _ = prepare_data_for_event(
            df, event_type=event_types[kk], features=features, 
            n_intervals=n_intervals,  even_split=even_split, event_ratio= event_ratio)
        time_intervals[:, kk] = ti_k
    return df[features].values, time_intervals, events, n_intervals

# train multibinary
def train_binmulti(    X,    time_intervals,    events,    n_intervals,
    hidden_dims=(),    lr=0.01,    epochs=300,    batch_size=1024):
    """     Train multi-outcome discrete-time binary model    (strict analogue of train_coxmulti)    """
    n, p = X.shape
    K = events.shape[1]
    X_tensor = torch.FloatTensor(X)
    intervals_tensor = torch.LongTensor(time_intervals)   # (n, K)
    events_tensor = torch.FloatTensor(events)             # (n, K)
    model = MultiDiscreteTimeNN(p=p, n_intervals=n_intervals, K=K, hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = (n + batch_size - 1) // batch_size
    # training loop
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            X_b = X_tensor[idx]
            intervals_b = intervals_tensor[idx]   # (b, K)
            events_b = events_tensor[idx]         # (b, K)
            # forward
            logits = model(X_b, intervals_b)      # (b, K)
            # -------- strict Cox analogy --------
            loss = 0.0
            for k in range(K): loss += F.binary_cross_entropy_with_logits(logits[:, k],events_b[:, k])
            loss = loss / K
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        avg_loss = total_loss / n
        if (epoch % 50 == 0) or (epoch + 1 == epochs):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model
    
def get_cindex_multibinary(binmulti, df_test, covariate_cols):
    with torch.no_grad():
        eta = binmulti.get_eta(
            torch.FloatTensor(df_test[covariate_cols].values)
        ).cpu().numpy()
    cindex_dict = {}
    for k, e in enumerate(["a","b","c","d","e"]):
        risk = eta[:, k]
        cindex_dict[e] = get_cindex_for_event(-risk, df=df_test,event=e)
    
    return cindex_dict


##############################################
## 6) SIMPLE BINARY MODEL WITH HIDDEN LAYERS ##
##############################################

#Let's just do a simple binary loss for the data where 
# 1) from the population we create a long-term data frame, 
# with id, start and end time stamps, and event_a etc 1/0. 
# 2) each patient contributes independently binary loss for each time period. 

class SimpleBinaryTimeSeries(nn.Module):
    """
    Discrete-time multi-outcome survival model.
    logit P(event k in interval t | x) = MLP(x)[k] + α_k[t]
    MLP(x) is shared across all intervals (proportional hazards assumption).
    α_k[t] is a learned per-event per-interval baseline hazard.
    """
    def __init__(self, p, K, n_intervals, hidden_dims=()):
        super().__init__()

        layers, in_dim = [], p
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, K, bias=False))
        self.mlp = nn.Sequential(*layers)

        self.alpha       = nn.Parameter(torch.zeros(K, n_intervals))
        self.K           = K
        self.n_intervals = n_intervals
        self.hidden_dims = hidden_dims

    def forward(self, x, interval_idx):
        """
        x            : (B, p)
        interval_idx : (B, K)  — same interval index for all K events per row
        returns logits (B, K)
        """
        eta   = self.mlp(x)                                          # (B, K)
        k_idx = torch.arange(self.K, device=x.device).unsqueeze(0).expand_as(interval_idx)
        alpha = self.alpha[k_idx, interval_idx]                      # (B, K)
        return eta + alpha

    def get_eta(self, x):
        """Covariate risk score (no interval baseline). Used for C-index ranking."""
        return self.mlp(x)

    def get_alpha(self):
        return self.alpha.detach().cpu().numpy()

    def get_beta(self):
        if self.hidden_dims:
            raise ValueError("get_beta() is only valid for linear models (hidden_dims=()).")
        return self.mlp[0].weight.data.cpu().numpy()   # (K, p)


def prepare_data_simple_timeseries(population):
    """
    Build long-format training data from a sim_population object.

    Returns df_long with one row per (patient, interval), plus mask columns
    mask_{e} = 1 if the patient is still in the risk set for event e at that
    interval (i.e. event e has NOT occurred in any prior interval).

    The interval in which the event first occurs IS included in the risk set
    (it contributes the positive loss signal). Only subsequent intervals
    are masked out.
    """
    df_long = population.to_long_format()
    event_types = ['a', 'b', 'c', 'd', 'e']

    for e in event_types:
        # cummax gives 1 from the first-occurrence interval onwards
        # shift(1) asks: "had it happened BEFORE this interval?"
        prior = (
            df_long.groupby('id')[f'event_{e}']
            .transform(lambda s: s.cummax().shift(1).fillna(0))
        )
        df_long[f'mask_{e}'] = (1 - prior).astype(np.float32)
        df_long[f'prev_{e}'] = (prior).astype(np.float32)

    return df_long


def train_simple_timeseries(
    df_long,
    features,
    event_types=["a", "b", "c", "d", "e"],
    hidden_dims=(),
    lr=0.01,
    epochs=50,
    batch_size=512,
):
    """
    Train SimpleBinaryTimeSeries with a masked BCE loss.

    Post-event rows are excluded from the loss via mask_{e} columns that must
    be present in df_long (produced by prepare_data_simple_timeseries).
    """
    p           = len(features)
    K           = len(event_types)
    n_intervals = int(df_long['interval'].max()) + 1

    X         = torch.FloatTensor(df_long[features].values)
    intervals = torch.LongTensor(
        df_long['interval'].values.reshape(-1, 1)
    ).expand(-1, K).contiguous()

    events = torch.stack(
        [torch.FloatTensor(df_long[f'event_{e}'].values) for e in event_types],
        dim=1
    )  # (N, K)

    masks = torch.stack(
        [torch.FloatTensor(df_long[f'mask_{e}'].values) for e in event_types],
        dim=1
    )  # (N, K)  — 1 = in risk set, 0 = already had event

    model     = SimpleBinaryTimeSeries(p=p, K=K, n_intervals=n_intervals, hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N        = len(df_long)
    n_batches = (N + batch_size - 1) // batch_size
    total_at_risk = masks.sum().item()

    for epoch in range(epochs):
        perm       = torch.randperm(N)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]

            logits = model(X[idx], intervals[idx])                # (B, K)

            # Raw BCE per element — then zero out post-event rows
            loss_raw = F.binary_cross_entropy_with_logits(
                logits, events[idx], reduction='none'
            )                                                     # (B, K)
            loss = (loss_raw * masks[idx]).sum() / masks[idx].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * masks[idx].sum().item()

        if (epoch % 50 == 0) or (epoch + 1 == epochs):
            print(f"Epoch {epoch+1}/{epochs}  loss = {epoch_loss / total_at_risk:.4f}")

    return model


def get_cindex_simple_timeseries(
    model,
    df_test,
    features,
    event_types=["a", "b", "c", "d", "e"],
):
    """
    Compute C-index for each event on a wide-format test set.

    Uses get_eta (covariate risk score, no interval baseline) as the ranking
    score — consistent with how MultiCoxNN and MultiDiscreteTimeNN are evaluated,
    and correct because covariates are static so eta is identical across all
    intervals for the same patient.

    Parameters
    ----------
    df_test : wide-format dataframe with columns time_{e} and event_{e}
              (e.g. from population.to_cox_format() test split)
    """
    X = torch.FloatTensor(df_test[features].values)

    with torch.no_grad():
        eta = model.get_eta(X).numpy()   # (n_test, K)

    cindex_dict = {}
    for k, e in enumerate(event_types):
        c_index = concordance_index(
            df_test[f'time_{e}'].values,
            -eta[:, k],
            df_test[f'event_{e}'].values,
        )
        cindex_dict[e] = c_index
        print(f"  Event {e}: C-index = {c_index:.4f}")

    return cindex_dict




##############################################
## 7) SEQUENCE TRANSFORMER (SimpleTransformer) ##
##############################################

# Sequence analogue of SimpleBinaryTimeSeries.
#   logit P(event k in interval t | history_0..t) = f(x_0..t)[k] + alpha_k[t]
# The MLP is replaced by a CAUSAL transformer encoder over the patient's
# interval sequence, so eta at interval t may depend on all intervals <= t.
# Loss, masking and the alpha_k[t] baseline are identical to the MLP version,
# which makes the two strictly nested and directly comparable.


class SimpleTransformerTimeSeries(nn.Module):
    """
    x            : (B, T, p)  one row per (patient, interval), chronological
    returns      : (B, T, K)  logits for each event in each interval

    Feature standardisation is stored INSIDE the model (buffers), so scoring
    and Monte-Carlo rollouts never need to remember the training mu/sd.
    """

    def __init__(self, p, K, n_intervals, d_model=64, nhead=4, nlayers=2,
                 dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.proj = nn.Linear(p, d_model)
        self.pos  = nn.Embedding(n_intervals, d_model)      # categorical interval index

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head    = nn.Linear(d_model, K, bias=False)
        self.alpha   = nn.Parameter(torch.zeros(K, n_intervals))

        self.p, self.K, self.n_intervals = p, K, n_intervals
        self.d_model = d_model

        # identity scaler until set_scaler() is called
        self.register_buffer("mu", torch.zeros(p))
        self.register_buffer("sd", torch.ones(p))

    # ---------------- scaling ----------------
    def set_scaler(self, X):
        """Fit standardisation on TRAIN data only. X: (n, T, p).
        Columns taking only {0, 1} are left untouched."""
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

    # ---------------- forward ----------------
    def get_eta(self, x):
        """Covariate risk score, no interval baseline. (B, T, p) -> (B, T, K)."""
        B, T, _ = x.shape
        if T > self.n_intervals:
            raise ValueError(f"sequence length {T} exceeds n_intervals={self.n_intervals}")

        h = self.proj(self._scale(x))
        h = h + self.pos(torch.arange(T, device=x.device))[None]

        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)      # strictly causal
        return self.head(h)

    def forward(self, x):
        """Full logits including the per-event per-interval baseline."""
        T = x.shape[1]
        return self.get_eta(x) + self.alpha[:, :T].T[None]  # (B, T, K)

    def return_hazards(self, x):
        return torch.sigmoid(self.forward(x))

    def predict_survival(self, x):
        """Cumulative survival per event: (B, T, K)."""
        return torch.cumprod(1.0 - self.return_hazards(x), dim=1)

    def get_alpha(self):
        return self.alpha.detach().cpu().numpy()


def prepare_data_simple_transformer(df_long, features,
                                    event_types=["a", "b", "c", "d", "e"]):
    """
    Reshape the long format into per-patient sequences.

    df_long must already carry mask_{e} / prev_{e}, i.e. come from
    prepare_data_simple_timeseries(population).

    Returns
    -------
    X       : (n_pat, T, p)  float tensor, features in `features` order
    events  : (n_pat, T, K)  1 if event k occurred in this interval
    masks   : (n_pat, T, K)  1 if still in the risk set for event k
    ids     : (n_pat,)       patient ids, aligned with axis 0 of X
    T       : int            number of intervals
    """
    df = df_long.sort_values(["id", "interval"]).reset_index(drop=True)

    ids = df["id"].drop_duplicates().values
    n_pat = len(ids)
    T = df["interval"].nunique()

    # ---- guard the reshape: silent misalignment here poisons everything ----
    if len(df) != n_pat * T:
        raise ValueError(f"expected {n_pat}*{T}={n_pat*T} rows, got {len(df)}; "
                         "sequences are ragged (patients with missing intervals)")
    sizes = df.groupby("id", sort=False)["interval"].size().values
    if not (sizes == T).all():
        raise ValueError("not every patient has the same number of intervals")
    iv = df["interval"].values.reshape(n_pat, T)
    if not (np.diff(iv, axis=1) > 0).all():
        raise ValueError("intervals are not strictly increasing within patient")

    event_cols = [f"event_{e}" for e in event_types]
    mask_cols  = [f"mask_{e}"  for e in event_types]
    missing = [c for c in features + event_cols + mask_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns in df_long: {missing}")

    K = len(event_types)
    X      = torch.FloatTensor(df[features].values).view(n_pat, T, len(features))
    events = torch.FloatTensor(df[event_cols].values).view(n_pat, T, K)
    masks  = torch.FloatTensor(df[mask_cols].values).view(n_pat, T, K)

    return X, events, masks, ids, T


def check_causality(model, X, t_break=None, atol=1e-5):
    """
    Assert the causal mask works: perturbing intervals >= t_break must not
    change eta at intervals < t_break. Run this before trusting any result.
    """
    model.eval()
    T = X.shape[1]
    t_break = T // 2 if t_break is None else t_break
    with torch.no_grad():
        x1 = X[:1].clone()
        eta_before = model.get_eta(x1)
        x2 = x1.clone()
        x2[0, t_break:] = x2[0, t_break:] + 10.0
        eta_after = model.get_eta(x2)
    ok = torch.allclose(eta_before[0, :t_break], eta_after[0, :t_break], atol=atol)
    if not ok:
        raise AssertionError("LOOK-AHEAD LEAK: eta before the break changed. "
                             "The causal mask is not being applied.")
    print(f"  causality check passed (no leakage from intervals >= {t_break})")
    return True


def train_simple_transformer(df_long, features,
                             event_types=["a", "b", "c", "d", "e"],
                             d_model=64, nhead=4, nlayers=2, dropout=0.1,
                             lr=1e-3, epochs=100, batch_size=128,
                             weight_decay=1e-4, verbose_every=10,
                             seed=None):
    """
    Train SimpleTransformerTimeSeries with the same masked BCE loss as
    train_simple_timeseries. Batching is per PATIENT, not per row.
    """
    if seed is not None:
        torch.manual_seed(seed)

    X, events, masks, ids, T = prepare_data_simple_transformer(
        df_long, features, event_types)
    n_pat, _, p = X.shape
    K = len(event_types)

    model = SimpleTransformerTimeSeries(
        p=p, K=K, n_intervals=T, d_model=d_model, nhead=nhead,
        nlayers=nlayers, dropout=dropout)
    model.set_scaler(X)                       # fitted on train only

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (n_pat + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_pat)
        tot, denom = 0.0, 0.0

        for i in range(0, n_pat, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X[idx])                                    # (B, T, K)
            loss_raw = F.binary_cross_entropy_with_logits(
                logits, events[idx], reduction="none")
            m = masks[idx]
            loss = (loss_raw * m).sum() / m.sum().clamp(min=1.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # transformers need this
            opt.step()
            sched.step()

            w = m.sum().item()
            tot += loss.item() * w
            denom += w

        if (epoch % verbose_every == 0) or (epoch + 1 == epochs):
            print(f"Epoch {epoch+1}/{epochs}  loss = {tot/denom:.4f}")

    check_causality(model, X)
    return model


def get_eta_transformer_at_interval(model, df_long, features, interval,
                                    event_types=["a", "b", "c", "d", "e"]):
    """
    Risk scores at a landmark interval, using only intervals 0..interval.

    Returns (eta, ids): eta is (n_pat, K) as a numpy array, ids aligned to it.
    Feed eta[:, k] (negated) into concordance_index, exactly as for the
    MLP models. Unlike the MLP, eta genuinely varies with the landmark.
    """
    X, _, _, ids, T = prepare_data_simple_transformer(df_long, features, event_types)
    if interval >= T:
        raise ValueError(f"interval {interval} out of range (T={T})")

    model.eval()
    with torch.no_grad():
        eta = model.get_eta(X[:, :interval + 1, :])      # truncate: no future rows at all
    return eta[:, interval, :].cpu().numpy(), ids


def get_cindex_simple_transformer(model, df_long_test, features, df_test,
                                  event_types=["a", "b", "c", "d", "e"],
                                  interval=0):
    """
    C-index per event at a landmark interval, scored against df_test
    (wide, from to_cox_format) with time_{e} / event_{e}.
    """
    eta, ids = get_eta_transformer_at_interval(
        model, df_long_test, features, interval, event_types)

    truth = df_test.set_index("id").loc[ids]
    cindex_dict = {}
    for k, e in enumerate(event_types):
        c = concordance_index(truth[f"time_{e}"].values,
                              -eta[:, k],
                              truth[f"event_{e}"].values)
        cindex_dict[e] = c
        print(f"  Event {e}: C-index = {c:.4f}")
    return cindex_dict


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

def simplecox(df, covariate_cols = ["age_start", "bmi", "hyp", "smoke", "sex", "eth1", "eth2"], event_type = "a"):
    time_col = f"time_{event_type}"
    event_col = f"event_{event_type}"
    cph = CoxPHFitter()
    cph.fit(df[[time_col, event_col] + covariate_cols], duration_col= time_col, event_col=event_col)
    #s=cph.summary[['coef', 'se(coef)', 'p']]
    #beta_cox = cph.params_.values
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
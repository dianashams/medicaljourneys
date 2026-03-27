
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
## 6) DISCRETE TIME MULTIBINARY WITH ATTENTION ##
##############################################
#┌─────────────────────────────────────────────────────────────┐
#│                                                             │
#│  BASELINE PATH (existing):                                  │
#│  Covariates → Dense Layers → η_baseline (K risk scores)     │
#│                                                             │
#│  ATTENTION PATH (new):                                      │
#│  Event History → Transformer → Attention Weights            │
#│                → Aggregate weighted history                 │
#│                → Dense Layer → η_attention (K adjustments)  │
#│                                                             │
#│  FUSION:                                                    │
#│  η_final = η_baseline + λ · η_attention                     │
#│           (or other fusion: concatenate + MLP)              │
#│ η_final = Dense(32) → ReLU → Dense(32) → ReLU → Dense(5)(η_fused)│
#│  OUTPUT:                                                    │
#│  logits = η_final + α[k,t]  (baseline hazards per event)    │
#│                                                             │
#└─────────────────────────────────────────────────────────────┘

#### Class definition: MultiDiscreteTimeNN augmented with Attention
class MultiDiscreteTimeNNWithAttention(nn.Module):
    """
    Multi-outcome discrete-time model with attention enrichment.
    
    Architecture:
    - Baseline path: Covariates → Dense layers → η_baseline (K risk scores)
    - Attention path: Event history → Transformer → η_attention (K adjustments)
    - Fusion: η_baseline + λ · η_attention
    - Refinement: Dense layer(s) to learn better fusion → η_final
    - Output: logits = η_final + α[k,t]
    """
    
    def __init__(self, p, n_intervals, K, hidden_dims=(),
                 attention_heads=4, attention_dim=32, num_transformer_layers=2,
                 attention_weight=1.0, fusion_hidden_dims=(32,)):
        """
        Args:
            p: number of covariates
            n_intervals: number of time intervals
            K: number of outcome events
            hidden_dims: tuple of hidden layer dimensions for baseline path
            attention_heads: number of attention heads in transformer
            attention_dim: dimension of attention embeddings
            num_transformer_layers: number of transformer encoder layers
            attention_weight: fixed scaling factor for attention contribution (λ)
            fusion_hidden_dims: tuple of hidden dims for fusion refinement layer(s)
        """
        super().__init__()
        
        # ===== BASELINE PATH =====
        layers = []
        in_dim = p
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, K, bias=False))
        self.baseline_net = nn.Sequential(*layers)
        
        # ===== ATTENTION PATH =====
        self.event_embedding = nn.Linear(K, attention_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attention_dim,
            nhead=attention_heads,
            dim_feedforward=attention_dim * 2,
            batch_first=True,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        self.attention_pool = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, K)
        )
        
        # ===== FUSION REFINEMENT (NEW) =====
        # Dense layer(s) after fusion to learn better combinations
        fusion_layers = []
        fusion_in_dim = K  # Input: fused eta (K scores)
        
        for h in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(fusion_in_dim, h))
            fusion_layers.append(nn.ReLU())
            fusion_in_dim = h
        
        # Final layer: back to K scores
        fusion_layers.append(nn.Linear(fusion_in_dim, K, bias=False))
        self.fusion_refinement = nn.Sequential(*fusion_layers)
        
        # ===== ATTENTION WEIGHT =====
        self.attention_weight = attention_weight
        
        # ===== BASELINE HAZARDS =====
        self.alpha = nn.Parameter(torch.zeros(K, n_intervals))
        
        self.K = K
        self.n_intervals = n_intervals
    
    def forward(self, x, interval_idx, event_history=None):
        """
        Forward pass with attention enrichment and fusion refinement.
        
        Args:
            x: (batch, p) static covariates
            interval_idx: (batch, K) current time interval per event
            event_history: (batch, history_length, K) past events or None
            
        Returns:
            logits: (batch, K) hazard logits
        """
        batch_size = x.shape[0]
        
        # ===== BASELINE PATH =====
        eta_baseline = self.baseline_net(x)  # (batch, K)
        
        # ===== ATTENTION PATH + FUSION =====
        if event_history is not None:
            event_embedded = self.event_embedding(event_history)
            attention_output = self.transformer_encoder(event_embedded)
            attention_pooled = attention_output.mean(dim=1)
            eta_attention = self.attention_pool(attention_pooled)  # (batch, K)
            
            # Fuse baseline + attention
            eta_fused = eta_baseline + self.attention_weight * eta_attention  # (batch, K)
        else:
            eta_fused = eta_baseline
        
        # ===== FUSION REFINEMENT (NEW) =====
        # Pass through dense layer(s) to learn better combinations
        eta = self.fusion_refinement(eta_fused)  # (batch, K)
        
        # ===== BASELINE HAZARDS =====
        n, K = interval_idx.shape
        k_idx = torch.arange(K, device=x.device).unsqueeze(0).expand(n, K)
        alpha_k = self.alpha[k_idx, interval_idx]
        
        logits = eta + alpha_k
        return logits
    
    def get_eta(self, x, event_history=None):
        """Extract linear predictors (for C-index calculation)."""
        eta_baseline = self.baseline_net(x)
        
        if event_history is not None:
            event_embedded = self.event_embedding(event_history)
            attention_output = self.transformer_encoder(event_embedded)
            attention_pooled = attention_output.mean(dim=1)
            eta_attention = self.attention_pool(attention_pooled)
            eta_fused = eta_baseline + self.attention_weight * eta_attention
        else:
            eta_fused = eta_baseline
        
        # Apply fusion refinement
        eta = self.fusion_refinement(eta_fused)
        return eta


#### Prepare data for MultiDiscreteTimeNN with Attention
def prepare_data_for_multibinary_with_attention(df, features, event_types=["a","b","c","d","e"],
                                               n_intervals=50, lookback_window=3,
                                               even_split=True, event_ratio=0.8):
    """
    Prepare data with event history for attention-enriched model.
    
    Args:
        df: input dataframe with time_* and event_* columns
        features: list of covariate column names
        event_types: list of event types (e.g., ["a","b","c","d","e"])
        n_intervals: number of time discretization intervals
        lookback_window: how many past steps to include in history
        even_split: if True, use equal-width bins; else quantile-based
        event_ratio: ratio of event/censoring cuts (when even_split=False)
        
    Returns:
        X: (n, p) static covariates
        time_intervals: (n, K) discretized time intervals
        events: (n, K) event indicators
        n_intervals: number of intervals
        event_history: (n, lookback_window, K) past event history
    """
    time_cols = [f"time_{e}" for e in event_types]
    event_cols = [f"event_{e}" for e in event_types]
    
    times = df[time_cols].values        # (n, K)
    events = df[event_cols].values      # (n, K)
    K = len(event_types)
    
    # Discretize times
    time_intervals = np.zeros_like(times, dtype=int)
    for kk in range(K):
        _, ti_k, _, _ = prepare_data_for_event(
            df, event_type=event_types[kk], features=features,
            n_intervals=n_intervals, even_split=even_split, event_ratio=event_ratio)
        time_intervals[:, kk] = ti_k
    
    # ===== BUILD EVENT HISTORY =====
    event_history = np.zeros((len(df), lookback_window, K))
    
    # Check if we have step-wise data in wide format
    step_cols_exist = f"event_a_step0" in df.columns
    
    if step_cols_exist:
        # Extract from wide format (e.g., event_a_step0, event_a_step1, ...)
        n_steps = max([int(col.split('step')[1]) for col in df.columns 
                      if 'step' in col and 'event_' in col]) + 1
        
        for step_idx in range(lookback_window):
            lookback_step = step_idx
            if lookback_step < n_steps:
                for e_idx, e in enumerate(event_types):
                    col_name = f"event_{e}_step{lookback_step}"
                    if col_name in df.columns:
                        event_history[:, step_idx, e_idx] = df[col_name].values
    
    return (df[features].values,  # X_static: (n, p)
            time_intervals,        # time_intervals: (n, K)
            events,               # events: (n, K)
            n_intervals,
            event_history)        # event_history: (n, lookback, K)


#### Training function for MultiDiscreteTimeNN with Attention
def train_binmulti_with_attention(X, time_intervals, events, event_history, n_intervals,
                                 hidden_dims=(), attention_heads=4, attention_dim=32,
                                 num_transformer_layers=2, attention_weight=1.0,
                                 fusion_hidden_dims=(32,),
                                 lr=0.01, epochs=300, batch_size=1024):
    """
    Train multi-outcome discrete-time model with attention enrichment and fusion refinement.
    
    Args:
        X: (n, p) static covariates
        time_intervals: (n, K) time interval indices
        events: (n, K) event indicators
        event_history: (n, lookback, K) past event history
        n_intervals: number of time intervals
        hidden_dims: tuple of hidden dimensions for baseline path
        attention_heads: number of attention heads
        attention_dim: dimension of attention embeddings
        num_transformer_layers: number of transformer layers
        attention_weight: scaling factor for attention (λ)
        fusion_hidden_dims: tuple of hidden dims for fusion refinement MLP
        lr: learning rate
        epochs: number of training epochs
        batch_size: batch size
        
    Returns:
        model: trained MultiDiscreteTimeNNWithAttention
    """
    n, p = X.shape
    K = events.shape[1]
    
    X_tensor = torch.FloatTensor(X)
    intervals_tensor = torch.LongTensor(time_intervals)
    events_tensor = torch.FloatTensor(events)
    history_tensor = torch.FloatTensor(event_history)
    
    model = MultiDiscreteTimeNNWithAttention(
        p=p, n_intervals=n_intervals, K=K,
        hidden_dims=hidden_dims,
        attention_heads=attention_heads,
        attention_dim=attention_dim,
        num_transformer_layers=num_transformer_layers,
        attention_weight=attention_weight,
        fusion_hidden_dims=fusion_hidden_dims
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = (n + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            X_b = X_tensor[idx]
            intervals_b = intervals_tensor[idx]
            events_b = events_tensor[idx]
            history_b = history_tensor[idx]
            
            logits = model(X_b, intervals_b, event_history=history_b)
            
            loss = 0.0
            for k in range(K):
                loss += F.binary_cross_entropy_with_logits(logits[:, k], events_b[:, k])
            loss = loss / K
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        
        avg_loss = total_loss / n
        if (epoch % 50 == 0) or (epoch + 1 == epochs):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


#### C-index computation for attention model
def get_cindex_multibinary_with_attention(model, df_test, covariate_cols, event_history_test=None):
    """     Compute C-index for attention-enriched model.
    Args:  model: trained MultiDiscreteTimeNNWithAttention
            df_test: test dataframe
            covariate_cols: list of covariate column names
            event_history_test: (n_test, lookback, K) test event history
    Returns: cindex_dict: dict with C-index for each event
    """
    with torch.no_grad():
        eta = model.get_eta(
            torch.FloatTensor(df_test[covariate_cols].values),
            event_history=torch.FloatTensor(event_history_test) if event_history_test is not None else None
        ).cpu().numpy()
    
    cindex_dict = {}
    for k, e in enumerate(["a", "b", "c", "d", "e"]):
        risk = eta[:, k]
        cindex_dict[e] = get_cindex_for_event(-risk, df=df_test, event=e)
    
    return cindex_dict



###################################################


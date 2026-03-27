import numpy as np
import pandas as pd
import random

def get_event_from_time(time_event, step_forward):
    return (time_event < step_forward).astype(int)

# "a" linear times  (diabetes), b => higher prob of a
def get_time_a(df, rng):
    #2a_ linear times  (diabetes), b => higher prob of a
    eth_beta = np.array([0 if (x==0) else 0.5 if (x==1) else 2 for x in df["eth"]])
    # chances are higher if there was a heart failure before 
    eventb_beta = (~df["first_b"].isna()).astype(int) * 0.3
    lp1 = 0.1*np.exp(0.5*df.bmi + 0.7*df.hyp + 0.4*(df.age-50)/15 + eth_beta + eventb_beta)
    time_a = 0.01 + np.round(rng.exponential(1/lp1,len(df)),3)
    return(time_a)

# "b" non-linear effect of bmi
def get_time_b(df,rng):
    # ~ heart failure,  a => higher prob of b
    #BMI impact is 2 for very low and high levels, 1 for high/ low level, 0 for normal range
    bmi_beta2 = np.array([2 if (np.abs(x)> 1.5) else 1 if (np.abs(x)>1) else 0 for x in df["bmi"]])
    #Age impact is 1 for age>=55; linear age impact is also present, but is smaller than in linear simulation
    age_normalised = (df.age-50)/15
    age_beta2nonl = np.array([1 if (x >=0.75) else 0 for x in age_normalised])
    
    eventa_beta = (~df["first_a"].isna()).astype(int) * 1

    lp2 = 0.07*np.exp(bmi_beta2 + 0.7*df.hyp+ 0.2*age_normalised + age_beta2nonl + eventa_beta)
    time_b = 0.01 + np.round(rng.exponential(1/lp2,len(df)),3)
    return(time_b)

# "c" non-linear effect of bmi plus smoking x age interaction term 
def get_time_c(df,rng):
    #2c_ x-terms times    (cancer)
    #BMI impact is 2 for very low and high levels, 1 for high/ low level, 0 for normal range
    bmi_beta3 = np.array([2 if (np.abs(x)> 1.5) else 1 if (np.abs(x)>1) else 0 for x in df["bmi"]])  
    # smoking x age interaction, lets assume it affects younger people stronger 
    age_normalised = (df["age"] - 50) / 15
    smoke_age_beta = np.select([(df["smoke"] == 1) & (age_normalised <= 0.5),
                                (df["smoke"] == 1) & (age_normalised < 0.5)],
                                [2, 1],default=0)
    lp3 = 0.07*np.exp(bmi_beta3 + smoke_age_beta + 0.2*df["hyp"] + 0.4*age_normalised)
    time_c = 0.01 + np.round(rng.exponential(1/lp3,len(df)),3)
    return(time_c)

# "d"  accelerating with age, dependency on ethnicity
def get_time_d(df,rng):
    #2d_ (dementia)
    age_beta =  (np.maximum(df.age/10 - 45,0)/10*0.1 + np.maximum(df.age/10 - 6,0)**2*0.4)
    eth_beta = np.array([0 if (x==0) else 0.2 if (x==1) else 0.5 for x in df["eth"]])
    lp4 = 0.01*np.exp(age_beta + eth_beta)
    time_d = 0.01 + np.round(rng.exponential(1/lp4,len(df)),3)
    return(time_d)

# "e"  any comorbidity => higher chances, otherwise weak dependency on eth
def get_time_e(df,rng):
    #2e_ (depression,  any comorbidity => higher chances)
    age_beta =  0
    eth_beta = np.where(df["eth"] == 0, 0.0, 0.5)
    a_beta = (~df["first_a"].isna()).astype(int)
    b_beta = (~df["first_b"].isna()).astype(int)
    c_beta = (~df["first_c"].isna()).astype(int)
    d_beta = (~df["first_d"].isna()).astype(int)
    e_beta = (~df["first_e"].isna()).astype(int)
    comorb_count = a_beta + b_beta + c_beta + d_beta + e_beta
    comorb_beta = 0.5*(comorb_count)+0.5*(comorb_count>2)+1*(comorb_count>3)
    lp5 = 0.01*np.exp(age_beta + eth_beta + comorb_beta)
    time_e = 0.1 + np.round(rng.exponential(1/lp5,len(df)),3)
    return(time_e)

# construct a wide format dataset with baseline covariates and time/event columns for each of the 5 steps
class sim_population:
    def __init__(self, N, step_forward, randomseed=None):
        self.N = N
        self.step_forward = step_forward
        self.randomseed = randomseed
        rng = np.random.default_rng(randomseed)
        
        self.df = pd.DataFrame({
            "id": np.arange(1, N + 1),
            "start": np.zeros(N),
            "end": np.full(N, step_forward),
            "age_start": np.round(rng.uniform(18, 75, N), 1),
            "bmi": np.round(rng.normal(0, 1, N), 1),
            "hyp": rng.binomial(1, 0.20, N),
            "smoke": rng.binomial(1, 0.15, N),
            "sex": rng.binomial(1, 0.5, N),
            "eth": rng.choice(3, size=N, p=[0.6, 0.3, 0.1]),
            "first_a": np.nan, "first_b": np.nan, 
            "first_c": np.nan, "first_d":np.nan, "first_e":np.nan            
            })
        self.df["age"] = self.df["age_start"]
        self.df["eth1"] = (self.df["eth"] == 1).astype(int)
        self.df["eth2"] = (self.df["eth"] == 2).astype(int)
        
        self._generate_times_and_events(rng)
        
        # store history
        self.history = []
        self._save_state()
    
    def _save_state(self):
        self.history.append(self.df.copy())

    def _generate_times_and_events(self, rng):
        self.df["time_a"] = get_time_a(self.df, rng)
        self.df["time_b"] = get_time_b(self.df, rng)
        self.df["time_c"] = get_time_c(self.df, rng)
        self.df["time_d"] = get_time_d(self.df,rng)
        self.df["time_e"] = get_time_d(self.df,rng)
        self.df["event_a"] = get_event_from_time(self.df["time_a"], self.step_forward)
        self.df["event_b"] = get_event_from_time(self.df["time_b"],self.step_forward)
        self.df["event_c"] = get_event_from_time(self.df["time_c"],self.step_forward)
        self.df["event_d"] = get_event_from_time(self.df["time_d"],self.step_forward)
        self.df["event_e"] = get_event_from_time(self.df["time_e"],self.step_forward)
        self._update_first_events()

    def _update_first_events(self):
            for ev in ["a", "b", "c", "d", "e"]:
                # check which ones have event_a ==1 but first_a as na 
                mask = ((self.df[f"event_{ev}"] == 1)& (self.df[f"first_{ev}"].isna()))
                # populate identified "mask" values with the time at which this happened = start + time_a
                self.df.loc[mask, f"first_{ev}"] = (self.df.loc[mask, "start"]+ self.df.loc[mask, f"time_{ev}"])

    def step(self):
        """   Move population forward and regenerate time_a with updated age.    """
        self.df["start"] = self.df["end"]
        self.df["end"] = self.df["end"] + self.step_forward
        self.df["age"] = np.round(self.df["age"] + self.step_forward, 1)
        # deterministic but unique per step
        step_seed = None
        if self.randomseed is not None: step_seed = self.randomseed + len(self.history)
        # create a temporary generator for this step
        rng_step = np.random.default_rng(step_seed)
        
        self._generate_times_and_events(rng_step)
        self._save_state()

    def to_wide_format(self):
        """  Constructs a wide format dataframe with one row per patient.
        Returns:- pd.DataFrame: Wide format dataframe containing:
                - Baseline covariates (from initial state): age_start, age_baseline, bmi, hyp, smoke, sex, eth, eth1, eth2
                - Step-specific columns for each step in history: time_{event}_step{i}, event_{event}_step{i}
                  (time values are ABSOLUTE from the beginning of simulation)
                  (if event == 0, time is set to start + step_forward, indicating no event in this step)
                - Cumulative first event columns: first_a, first_b, first_c, first_d, first_e
        """
        # Get baseline covariates from the initial state (history[0])
        baseline_df = self.history[0][['id', 'age_start', 'bmi', 'hyp', 'smoke', 'sex', 'eth1', 'eth2']].copy()
        
        # Start building the wide format dataframe
        wide_df = baseline_df.copy()

        event_types = ['a', 'b', 'c', 'd', 'e']
        # Extract step-specific columns from history
        for step_idx in range(len(self.history)):  # Changed from range(1, ...) to range(...)
            step_data = self.history[step_idx]
            
            for event_type in event_types:
                # Add time and event columns for each step
                time_col_name = f'time_{event_type}_step{step_idx}'
                event_col_name = f'event_{event_type}_step{step_idx}'
                wide_df[time_col_name] = step_data[f'time_{event_type}'].values
                wide_df[event_col_name] = step_data[f'event_{event_type}'].values
        
        # Add the "first" event columns (cumulative across all steps)
        for event_type in event_types: 
            wide_df[f'first_{event_type}'] = self.df[f'first_{event_type}'].values
        
        return wide_df
        
    def to_cox_format(self):
        """  takes baseline covariates + first_ever column to check if event ever happened, 
        returns baseline + event_{a} + time_{a} columns - ready for Cox-like time-to-event analysis
        """
        cox_df = self.history[0][['id', 'age_start', 'bmi', 'hyp', 'smoke', 'sex', 'eth1', 'eth2']].copy()  
        # Take the last population (use -1 to get the last element)
        step_data = self.history[-1]
        for e in ['a', 'b', 'c', 'd', 'e']:
            # event: 1 if first_{e} is not NaN, 0 otherwise
            cox_df[f'event_{e}'] = (~step_data[f'first_{e}'].isna()).astype(int)
            # time: use first_{e} (absolute time from beginning) if event occurred, 
            # otherwise use the final end time (censored)
            cox_df[f'time_{e}'] = np.where(cox_df[f'event_{e}'] == 1,
                                           step_data[f'first_{e}'].values,
                                           step_data['end'].values)
        return cox_df
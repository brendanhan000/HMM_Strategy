# HMM_market_regimes.py
# Requires: pip install yfinance numpy pandas matplotlib scikit-learn hmmlearn ta
# If ta not available, the code computes RSI manually.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import math
import warnings
warnings.filterwarnings("ignore")

# --- 1. Download data (~10 years daily)
ticker = "^GSPC"   # S&P 500 index; change as desired
end_date = None    # None => today
start_date = "2015-01-01"  # ~10+ years from 2025-09-23; adjust if you need exactly 10y
df = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Basic checks
df = df.dropna()
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# --- 2. Feature engineering
# Returns:
df['ret'] = df['Close'].pct_change().fillna(0)           # simple returns
df['logret'] = np.log(df['Close']).diff().fillna(0)     # log returns

# ATR (14)
high = df['High']
low = df['Low']
close = df['Close']
prev_close = close.shift(1)
tr1 = high - low
tr2 = (high - prev_close).abs()
tr3 = (low - prev_close).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR14'] = tr.rolling(14, min_periods=1).mean()

# Squared returns (vol proxy)
df['sqret'] = df['logret']**2

# Volume change (log diff)
df['vol_logdiff'] = np.log(df['Volume']).diff().fillna(0)

# Moving averages and MA spread
df['ma50'] = df['Close'].rolling(50, min_periods=1).mean()
df['ma200'] = df['Close'].rolling(200, min_periods=1).mean()
df['ma_spread'] = (df['ma50'] - df['ma200']) / df['ma200']   # normalized

# RSI (14) - simple implementation
delta = df['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
avg_gain = up.ewm(alpha=1/14, adjust=False).mean()
avg_loss = down.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / (avg_loss + 1e-12)
df['rsi14'] = 100 - (100 / (1 + rs))

# Drop initial rows where indicators are NaN
df = df.dropna().copy()

# Choose features (tweak as desired)
features = ['logret', 'sqret', 'ATR14', 'vol_logdiff', 'ma_spread', 'rsi14']
X = df[features].values

# Standardize
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# --- 3. Fit a 3-state Gaussian HMM
n_states = 3
model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=2000, random_state=42, verbose=False)
model.fit(Xs)   # Baum-Welch happens here

# Decode with Viterbi
hidden_states = model.predict(Xs)
df['state'] = hidden_states

# State posterior probabilities (optional)
posteriors = model.predict_proba(Xs)  # shape (n_obs, n_states)

# --- 4. Analyze states
state_stats = []
for s in range(n_states):
    mask = df['state'] == s
    mean_ret = df.loc[mask, 'logret'].mean()
    std_ret = df.loc[mask, 'logret'].std()
    median_atr = df.loc[mask, 'ATR14'].median()
    count = mask.sum()
    state_stats.append({'state': s, 'count': int(count),
                        'mean_logret': float(mean_ret), 'std_logret': float(std_ret),
                        'median_ATR14': float(median_atr)})
state_stats_df = pd.DataFrame(state_stats).sort_values('state').reset_index(drop=True)
print("Per-state summary:")
print(state_stats_df)

# Transition matrix
transmat = model.transmat_
print("\nTransition matrix (rows = from-state, cols = to-state):")
print(np.round(transmat, 3))

# Persistence (average run length)
def avg_run_length(arr, state):
    runs = []
    current_run = 0
    for x in arr:
        if x == state:
            current_run += 1
        elif current_run > 0:
            runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    return np.mean(runs) if runs else 0

persistence = {s: avg_run_length(hidden_states, s) for s in range(n_states)}
print("\nAverage persistence (days):", persistence)

# --- 5. Map states to regimes by mean return (and volatility)
# Find ordering
state_order = sorted(state_stats, key=lambda x: x['mean_logret'])
mapping = {}
# state with max mean -> Bull, min mean -> Bear, middle -> Sideways
sorted_by_mean = sorted(state_stats, key=lambda x: x['mean_logret'])
mapping[sorted_by_mean[-1]['state']] = 'Bullish'
mapping[sorted_by_mean[0]['state']] = 'Bearish'
# the remaining
middle_state = [s['state'] for s in state_stats if s['state'] not in (sorted_by_mean[-1]['state'], sorted_by_mean[0]['state'])][0]
mapping[middle_state] = 'Sideways'
df['regime'] = df['state'].map(mapping)

print("\nState -> Regime mapping:", mapping)

# --- 6. Evaluation: log-likelihood, AIC, BIC
logL = model.score(Xs) * len(Xs) / len(Xs)  # .score returns avg log-likelihood per sample times n_samples? hmmlearn.score returns total log prob
# For hmmlearn, .score(X) returns log likelihood of X (not per-sample), so use that directly:
logL = model.score(Xs)
n = Xs.shape[0]
# Number of parameters k estimation for Gaussian HMM:
# k = initial state probs (n_states-1) + transition matrix (n_states*(n_states-1)) + emission params:
# for each state: mean vector (d) + covariance matrix (d*(d+1)/2)
d = Xs.shape[1]
k_emission = n_states * (d + d*(d+1)/2)
k = (n_states - 1) + n_states*(n_states - 1) + k_emission
AIC = 2*k - 2*logL
BIC = math.log(n)*k - 2*logL

print(f"\nLog-likelihood: {logL:.2f}")
print(f"Parameters (k) approx: {k}")
print(f"AIC: {AIC:.2f}, BIC: {BIC:.2f}")

# --- 7. Visualization suggestions
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Close'], label='Close')
# color background by regime
regime_colors = {'Bullish':'#b7e4c7', 'Bearish':'#ffd6a5', 'Sideways':'#d3d3f9'}
last_idx = df.index[0]
last_regime = df['regime'].iloc[0]
start = df.index[0]
for i in range(1, len(df)):
    if df['regime'].iloc[i] != last_regime:
        plt.axvspan(start, df.index[i], color=regime_colors[last_regime], alpha=0.25)
        start = df.index[i]
        last_regime = df['regime'].iloc[i]
# final span
plt.axvspan(start, df.index[-1], color=regime_colors[last_regime], alpha=0.25)
plt.title(f"{ticker} price with regime shading")
plt.legend()
plt.show()

# State sequence plot
plt.figure(figsize=(14,3))
plt.plot(df.index, df['state'], drawstyle='steps-post')
plt.yticks(range(n_states))
plt.title("HMM inferred states (0..n_states-1)")
plt.show()

# Per-state returns boxplot
plt.figure(figsize=(8,5))
df_box = df[['logret', 'state']].copy()
df_box['state'] = df_box['state'].astype(str)
df_box.boxplot(column='logret', by='state')
plt.suptitle('')
plt.title('Distribution of log returns by state')
plt.show()

# Save results for further analysis
df.to_csv("hmm_regimes_output.csv")
print("\nSaved output to hmm_regimes_output.csv")

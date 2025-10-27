import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.multitest import multipletests

# ========= Parameters =========
W_PRE_SHORT = 2   # Short bug window (days)
W_PRE_LONG  = 14  # Long bug window (days)
N_PIECES = 20     # Number of relative time pieces
H_LEAD = 2        # Lead label length (pieces)
LAG_MAX = 3       # Maximum Granger lag order
FEATURES = ["comments_per_day","reassignments_per_day","field_changes_per_day"]

# ========= Utility Functions =========
def make_lead_label(series_len, h_lead):
    """Generate lead label: last h_lead pieces are 1, others are 0"""
    y = np.zeros(series_len)
    y[-h_lead:] = 1
    return y

def to_relative_pieces(series, n_pieces=N_PIECES):
    """Normalize arbitrary length series to n_pieces pieces"""
    idxs = np.linspace(0, len(series)-1, n_pieces).astype(int)
    rel_series = series.iloc[idxs].reset_index(drop=True)
    return rel_series

def adf_stationary(series):
    """log1p + first difference (if non-stationary)"""
    x = np.log1p(series.astype(float))
    try:
        pval = adfuller(x, autolag='AIC')[1]
    except:
        pval = 1.0
    diffed = False
    if pval > 0.05 and len(x) > 5:
        x = x.diff().dropna()
        diffed = True
    return x, diffed

def granger_one(event_row, ts_df, features, w_pre_days, n_pieces=N_PIECES, h_lead=H_LEAD, lag_max=LAG_MAX):
    bug_id = event_row['bug_id']
    t_c = pd.to_datetime(event_row['t_c'])
    
    sub = ts_df[ts_df['bug_id']==bug_id].copy()
    if sub.empty:
        return []
    sub['date'] = pd.to_datetime(sub['date'])
    sub = sub.sort_values('date')
    
    # 取事件前 W_PRE 窗口
    start = t_c - timedelta(days=w_pre_days)
    sub = sub[sub['date'] < t_c]
    
    results = []
    for feat in features:
        x_raw = sub[feat]
        if len(x_raw) < 2:  # Too few samples
            continue
        # Convert to relative time pieces
        x_rel = to_relative_pieces(x_raw, n_pieces)
        y_rel = make_lead_label(len(x_rel), h_lead)
        # Stationarity processing
        x_proc, diffed = adf_stationary(x_rel)
        min_len = min(len(x_proc), len(y_rel))
        x_proc = x_proc.iloc[-min_len:]
        y_rel = y_rel[-min_len:]
        # Build matrix
        data = np.column_stack([y_rel, x_proc.values])
        try:
            gc = grangercausalitytests(data, maxlag=lag_max, verbose=False)
            lag_pvals = {lag: gc[lag][0]['ssr_ftest'][1] for lag in gc.keys()}
            lag_F = {lag: gc[lag][0]['ssr_ftest'][0] for lag in gc.keys()}
            best_lag = min(lag_pvals, key=lag_pvals.get)
            best_p = lag_pvals[best_lag]
            best_F = lag_F[best_lag]
        except:
            best_lag, best_p, best_F = np.nan, np.nan, np.nan
        results.append({
            'event_id': event_row['event_id'],
            'bug_id': bug_id,
            'feature': feat,
            'best_lag': best_lag,
            'p_value': best_p,
            'F_stat': best_F,
            'x_diffed': diffed
        })
    return results

# ========= Main Process =========
def run_granger_all(events_csv, ts_csv, w_pre_short=W_PRE_SHORT, w_pre_long=W_PRE_LONG):
    events = pd.read_csv(events_csv)
    ts = pd.read_csv(ts_csv)
    all_results = []
    for idx, ev in events.iterrows():
        # Select window based on event span
        bug_id = ev['bug_id']
        ts_sub = ts[ts['bug_id']==bug_id]
        if ts_sub.empty:
            continue
        span_days = (pd.to_datetime(ev['t_c']) - ts_sub['date'].min()).days
        w_pre = w_pre_short if span_days <= w_pre_short else w_pre_long
        rows = granger_one(ev, ts, FEATURES, w_pre_days=w_pre)
        all_results.extend(rows)
    
    res_df = pd.DataFrame(all_results)
    
    # FDR correction
    res_df['q_value'] = np.nan
    for feat, df_feat in res_df.groupby('feature'):
        pvals = df_feat['p_value'].values
        mask = ~np.isnan(pvals)
        q = np.full_like(pvals, np.nan, dtype=float)
        if mask.sum() > 0:
            _, qvals, _, _ = multipletests(pvals[mask], alpha=0.05, method='fdr_bh')
            q[mask] = qvals
        res_df.loc[df_feat.index, 'q_value'] = q
    
    # Summarize feature-level results
    summary = (res_df.assign(sig=lambda d: d['q_value']<0.05)
               .groupby('feature')
               .agg(
                   n_tests=('p_value','count'),
                   n_sig=('sig','sum'),
                   pct_sig=('sig', lambda s: 100*s.mean()),
                   median_lag=('best_lag', lambda x: np.nanmedian(x)),
                   median_F=('F_stat', lambda x: np.nanmedian(x)),
                   median_p=('p_value', lambda x: np.nanmedian(x)),
                   median_q=('q_value', lambda x: np.nanmedian(x))
               ).reset_index())
    
    return res_df, summary

# Usage example:
# res, summary = run_granger_all("events.csv", "timeseries.csv")
# print(summary)

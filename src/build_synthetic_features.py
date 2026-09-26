"""Turns synthetic_resource_telemetry.parquet into the same two model-ready
tables the real pipeline produced, so train_cost_forecast.py / train_optimizer.py
run unchanged."""
import sys
sys.path.insert(0, '/home/claude/cloudlens-ml/src')
import pandas as pd
from features import add_time_series_features  # reuse unchanged

PROC = '/home/claude/cloudlens-ml/data/processed'

telem = pd.read_parquet(f'{PROC}/synthetic_resource_telemetry.parquet')

# ---- forecast panel: (date, provider, service, region) daily grain
panel = telem.groupby(['date', 'provider', 'service', 'region']).agg(
    cost_usd=('cost_usd', 'sum'), usage_quantity=('usage_quantity', 'sum'),
    cpu_util_pct=('cpu_util_pct', 'mean'), mem_util_pct=('mem_util_pct', 'mean'),
    n_resources=('resource_id', 'nunique'),
).reset_index()
panel['has_utilization_data'] = 1
panel = panel.sort_values(['provider', 'service', 'region', 'date']).reset_index(drop=True)
panel = add_time_series_features(panel)
panel.to_parquet(f'{PROC}/cost_forecast_features.parquet', index=False)
print('cost_forecast_features:', panel.shape)

# ---- optimization table: per-resource-day grain
opt = telem.copy()
opt['is_underutilized'] = ((opt['cpu_util_pct'] < 30) & (opt['mem_util_pct'] < 30)).astype(int)
opt['day_of_week'] = opt['date'].dt.dayofweek
opt['month'] = opt['date'].dt.month
cols = ['resource_id', 'provider', 'service', 'region', 'usage_quantity', 'unit_price_usd',
        'cost_usd', 'cpu_util_pct', 'mem_util_pct', 'net_in_bytes', 'net_out_bytes',
        'net_total_bytes', 'duration_hours', 'day_of_week', 'month', 'is_underutilized']
opt = opt[cols]
opt.to_parquet(f'{PROC}/optimization_features.parquet', index=False)
print('optimization_features:', opt.shape, '| underutilized rate:', round(opt.is_underutilized.mean(), 3))

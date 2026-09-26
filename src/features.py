"""
CloudLens - Feature engineering for both models.
"""
import pandas as pd
import numpy as np

PROC = '/home/claude/cloudlens-ml/data/processed'


def build_daily_cost_panel():
    """Daily cost panel across providers -> input for the forecasting model.
    Grain: (date, provider, service, region). Azure monthly series is kept as a
    separate reference file (different grain) rather than merged in here, since
    mixing daily and monthly frequencies would corrupt the lag/rolling features."""
    gcp = pd.read_parquet(f'{PROC}/gcp_unified.parquet')
    gcp['date'] = gcp['usage_start'].dt.floor('D')
    gcp_daily = gcp.groupby(['date', 'provider', 'service', 'region']).agg(
        cost_usd=('cost_usd', 'sum'),
        usage_quantity=('usage_quantity', 'sum'),
        cpu_util_pct=('cpu_util_pct', 'mean'),
        mem_util_pct=('mem_util_pct', 'mean'),
        n_resources=('resource_id', 'nunique'),
    ).reset_index()

    az = pd.read_parquet(f'{PROC}/azure_resource_level.parquet')
    az['date'] = az['usage_start'].dt.floor('D')
    # NOTE: service_family is a constant ("Compute") across every row in this
    # export -- using it as the service feature would mean the model never
    # sees real Azure service names like "Virtual Machines" or "Storage"
    # during training, only "Compute", making every real service name an
    # unseen category at inference time. meter_category has genuine variety
    # (34 distinct values) and is what a live caller would actually send.
    az_daily = az.groupby(['date', 'provider', 'meter_category', 'region']).agg(
        cost_usd=('cost_usd', 'sum'),
        usage_quantity=('usage_quantity', 'sum'),
        n_resources=('meter_id', 'nunique'),
    ).reset_index().rename(columns={'meter_category': 'service'})
    az_daily['cpu_util_pct'] = np.nan
    az_daily['mem_util_pct'] = np.nan

    panel = pd.concat([gcp_daily, az_daily], ignore_index=True)
    panel['has_utilization_data'] = panel['cpu_util_pct'].notna().astype(int)
    panel[['cpu_util_pct', 'mem_util_pct']] = panel[['cpu_util_pct', 'mem_util_pct']].fillna(0)
    panel = panel.sort_values(['provider', 'service', 'region', 'date']).reset_index(drop=True)
    return panel


def add_time_series_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel['day_of_week'] = panel['date'].dt.dayofweek
    panel['day_of_month'] = panel['date'].dt.day
    panel['month'] = panel['date'].dt.month
    panel['is_weekend'] = (panel['day_of_week'] >= 5).astype(int)

    grp_key = ['provider', 'service', 'region']
    grp = panel.groupby(grp_key)
    panel['cost_lag_1'] = grp['cost_usd'].shift(1)
    panel['cost_lag_7'] = grp['cost_usd'].shift(7)
    panel['cost_roll_mean_7'] = grp['cost_usd'].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    panel['cost_roll_std_7'] = grp['cost_usd'].transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())
    panel['usage_lag_1'] = grp['usage_quantity'].shift(1)
    panel['usage_roll_mean_7'] = grp['usage_quantity'].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    # cold-start fill: first observation per group has no history yet
    fill_cols = ['cost_lag_1', 'cost_lag_7', 'cost_roll_mean_7', 'cost_roll_std_7', 'usage_lag_1', 'usage_roll_mean_7']
    panel[fill_cols] = panel.groupby(['provider', 'service', 'region'])[fill_cols].transform(lambda s: s.bfill().fillna(0))
    return panel


def build_resource_optimization_table():
    """Row-level resource usage records -> input for the optimization/right-sizing
    classifier. Grain: one row per (resource, usage window)."""
    gcp = pd.read_parquet(f'{PROC}/gcp_unified.parquet').copy()
    gcp['avg_util_pct'] = gcp[['cpu_util_pct', 'mem_util_pct']].mean(axis=1)
    # Weak-supervision label: FinOps rule-of-thumb for over-provisioned compute
    # (both CPU and memory sitting under 30% utilization over the usage window)
    gcp['is_underutilized'] = ((gcp['cpu_util_pct'] < 30) & (gcp['mem_util_pct'] < 30)).astype(int)
    gcp['day_of_week'] = gcp['usage_start'].dt.dayofweek
    gcp['month'] = gcp['usage_start'].dt.month
    gcp['net_total_bytes'] = gcp['net_in_bytes'] + gcp['net_out_bytes']

    cols = ['resource_id', 'provider', 'service', 'region', 'usage_quantity', 'unit_price_usd',
            'cost_usd', 'cpu_util_pct', 'mem_util_pct', 'avg_util_pct', 'net_in_bytes',
            'net_out_bytes', 'net_total_bytes', 'duration_hours', 'day_of_week', 'month',
            'is_underutilized']
    return gcp[cols]


if __name__ == '__main__':
    panel = build_daily_cost_panel()
    panel = add_time_series_features(panel)
    panel.to_parquet(f'{PROC}/cost_forecast_features.parquet', index=False)
    print('cost_forecast_features:', panel.shape)

    opt = build_resource_optimization_table()
    opt.to_parquet(f'{PROC}/optimization_features.parquet', index=False)
    print('optimization_features:', opt.shape)
    print('underutilized rate:', opt['is_underutilized'].mean().round(3))

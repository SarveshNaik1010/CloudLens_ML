"""
CloudLens API - bridges validated request payloads to the two trained models
and back into response schemas. No direct cloud-provider API calls happen
here -- that's the job of the caller's own helper API; this module only
takes usage/cost data already fetched and runs it through the models.
"""
from typing import List
import numpy as np
import pandas as pd

from schemas import ResourceUsageIn, ForecastGroup, ForecastResponse, OptimizationTip, OptimizeResponse, AnalyzeResponse


def _records_to_df(aws, azure, gcp) -> pd.DataFrame:
    rows = []
    for provider, items in (('aws', aws), ('azure', azure), ('gcp', gcp)):
        for r in (items or []):
            d = r.model_dump()
            d['provider'] = provider
            rows.append(d)
    df = pd.DataFrame(rows)
    numeric_cols = ['usage_quantity', 'cost_usd', 'unit_price_usd', 'cpu_util_pct', 'mem_util_pct',
                     'net_in_bytes', 'net_out_bytes', 'duration_hours']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def run_forecast(forecaster, aws, azure, gcp, now=None) -> ForecastResponse:
    """Aggregates resources into (provider, service, region) groups -- the
    same grain the model was trained on -- then predicts each group's next
    reporting-period cost."""
    now = now or pd.Timestamp.utcnow()
    df = _records_to_df(aws, azure, gcp)

    # history-aware lag/rolling features when recent_daily_costs was supplied,
    # otherwise a steady-state assumption (see ResourceUsageIn docstring)
    def lag1(row):
        h = row.get('recent_daily_costs') or []
        return h[-1] if h else row['cost_usd']

    def lag7(row):
        h = row.get('recent_daily_costs') or []
        return h[0] if len(h) >= 7 else row['cost_usd']

    def roll_mean(row):
        h = row.get('recent_daily_costs') or []
        return float(np.mean(h)) if h else row['cost_usd']

    def roll_std(row):
        h = row.get('recent_daily_costs') or []
        return float(np.std(h)) if len(h) > 1 else 0.0

    df['cost_lag_1'] = df.apply(lag1, axis=1)
    df['cost_lag_7'] = df.apply(lag7, axis=1)
    df['cost_roll_mean_7'] = df.apply(roll_mean, axis=1)
    df['cost_roll_std_7'] = df.apply(roll_std, axis=1)
    df['usage_lag_1'] = df['usage_quantity']
    df['usage_roll_mean_7'] = df['usage_quantity']
    df['has_utilization_data'] = df['cpu_util_pct'].notna().astype(int)
    df[['cpu_util_pct', 'mem_util_pct']] = df[['cpu_util_pct', 'mem_util_pct']].fillna(0)

    grouped = df.groupby(['provider', 'service', 'region']).agg(
        cost_usd=('cost_usd', 'sum'),
        usage_quantity=('usage_quantity', 'sum'),
        n_resources=('resource_id', 'nunique'),
        cost_lag_1=('cost_lag_1', 'sum'),
        cost_lag_7=('cost_lag_7', 'sum'),
        cost_roll_mean_7=('cost_roll_mean_7', 'sum'),
        cost_roll_std_7=('cost_roll_std_7', 'mean'),
        usage_lag_1=('usage_lag_1', 'sum'),
        usage_roll_mean_7=('usage_roll_mean_7', 'sum'),
        has_utilization_data=('has_utilization_data', 'max'),
    ).reset_index()

    grouped['day_of_week'] = now.dayofweek
    grouped['day_of_month'] = now.day
    grouped['month'] = now.month
    grouped['is_weekend'] = int(now.dayofweek >= 5)

    preds = forecaster.predict(grouped)
    grouped['model_raw_prediction_usd'] = preds

    # Safety clamp: the model was trained on daily panels dominated by
    # service/region groups with many contributing resources (n_resources is
    # its single biggest signal, ~45% importance). A live call with only a
    # handful of resources sits far outside that training distribution and
    # the raw prediction can overshoot badly (see reports/eda_report.md,
    # "API calibration" section). Until the model is retrained on a live-call
    # distribution, clamp to a defensible multiple of the group's current
    # cost rather than return an unbounded extrapolation.
    lower = grouped['cost_usd'] * 0.2
    upper = grouped['cost_usd'] * 5 + 50
    grouped['predicted_next_period_cost_usd'] = grouped['model_raw_prediction_usd'].clip(lower=lower, upper=upper)

    groups = [
        ForecastGroup(
            provider=r.provider, service=r.service, region=r.region,
            n_resources=int(r.n_resources), current_cost_usd=round(float(r.cost_usd), 2),
            predicted_next_period_cost_usd=round(float(r.predicted_next_period_cost_usd), 2),
            model_raw_prediction_usd=round(float(r.model_raw_prediction_usd), 2),
        )
        for r in grouped.itertuples()
    ]
    return ForecastResponse(
        groups=groups,
        total_current_cost_usd=round(float(grouped['cost_usd'].sum()), 2),
        total_predicted_next_period_cost_usd=round(float(grouped['predicted_next_period_cost_usd'].sum()), 2),
    )


def run_optimize(optimizer, aws, azure, gcp, threshold: float = 0.5) -> OptimizeResponse:
    """Per-resource right-sizing scan -- the grain the classifier was trained
    on -- using whatever utilization/network telemetry each provider supplied."""
    df = _records_to_df(aws, azure, gcp)
    df['net_in_bytes'] = df['net_in_bytes'].fillna(np.nan)
    df['net_out_bytes'] = df['net_out_bytes'].fillna(np.nan)
    df['net_total_bytes'] = df['net_in_bytes'] + df['net_out_bytes']
    df['duration_hours'] = df['duration_hours'].fillna(df['usage_quantity'])
    now = pd.Timestamp.utcnow()
    df['day_of_week'] = now.dayofweek
    df['month'] = now.month

    proba = optimizer.predict_proba(df)
    df['underutilized_probability'] = proba
    flagged = proba >= threshold

    avg_util = df[['cpu_util_pct', 'mem_util_pct']].mean(axis=1, skipna=True)
    df['avg_utilization_pct'] = avg_util
    df['est_monthly_savings_usd'] = 0.0
    df.loc[flagged, 'est_monthly_savings_usd'] = (
        df.loc[flagged, 'cost_usd'] * (1 - avg_util.loc[flagged].fillna(50) / 100) * 0.6 * 30
    ).round(2)
    df.loc[flagged, 'tip'] = (
        'Right-size or downgrade "' + df.loc[flagged, 'resource_id'].astype(str) + '" ('
        + df.loc[flagged, 'service'] + ', ' + df.loc[flagged, 'region']
        + ') -- running at ' + avg_util.loc[flagged].round(1).astype(str)
        + '% average utilization.'
    )

    tips_df = df[flagged].sort_values('est_monthly_savings_usd', ascending=False)
    tips = [
        OptimizationTip(
            resource_id=r.resource_id, provider=r.provider, service=r.service, region=r.region,
            underutilized_probability=round(float(r.underutilized_probability), 4),
            cost_usd=round(float(r.cost_usd), 2),
            avg_utilization_pct=None if pd.isna(r.avg_utilization_pct) else round(float(r.avg_utilization_pct), 1),
            est_monthly_savings_usd=round(float(r.est_monthly_savings_usd), 2),
            tip=r.tip,
        )
        for r in tips_df.itertuples()
    ]
    return OptimizeResponse(
        n_resources_scanned=len(df), n_flagged=len(tips_df),
        total_est_monthly_savings_usd=round(float(tips_df['est_monthly_savings_usd'].sum()), 2),
        tips=tips,
    )

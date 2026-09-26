"""
CloudLens - production entrypoint.
Pulls live usage from AWS/Azure/GCP (src/live_collectors), scores it with
both models, and returns a forecast + a prioritized list of cost-saving tips.

Example:
    python predict.py --provider gcp --project my-gcp-project --zone us-central1-a
"""
import argparse
import pandas as pd
import pickle
from pathlib import Path

from model_wrappers import CloudLensForecaster, CloudLensOptimizer  # noqa: F401
from recommend import rightsizing_tips

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / 'models'


def load_models():
    with open(MODELS / 'cost_forecast_model.pkl', 'rb') as f:
        forecaster = pickle.load(f)
    with open(MODELS / 'resource_optimizer_model.pkl', 'rb') as f:
        optimizer = pickle.load(f)
    return forecaster, optimizer


def collect_live_usage(provider: str, **kwargs) -> pd.DataFrame:
    if provider == 'aws':
        from live_collectors.aws import fetch_ec2_usage
        return fetch_ec2_usage(region=kwargs['region'])
    elif provider == 'azure':
        from live_collectors.azure import fetch_vm_usage
        return fetch_vm_usage(subscription_id=kwargs['subscription_id'])
    elif provider == 'gcp':
        from live_collectors.gcp import fetch_compute_usage
        return fetch_compute_usage(project_id=kwargs['project'], zone=kwargs['zone'])
    else:
        raise ValueError(f'unknown provider: {provider}')


def run(provider: str, **kwargs):
    forecaster, optimizer = load_models()
    usage = collect_live_usage(provider, **kwargs)

    if usage.empty:
        print('No running resources found.')
        return

    forecast_input = usage.rename(columns={'service': 'service'})  # same names already
    # forecaster needs the daily-panel feature set (lag/rolling cost history);
    # for a single live snapshot, backfill those from the resource's own
    # recent cost if available, else 0 (cold start -- see features.py)
    forecast_cols = forecaster.feature_order
    missing = [c for c in forecast_cols if c not in forecast_input.columns]
    for c in missing:
        forecast_input[c] = 0
    forecast_input['n_resources'] = 1
    predicted_cost = forecaster.predict(forecast_input)
    usage = usage.assign(predicted_next_period_cost_usd=predicted_cost)

    tips = rightsizing_tips(usage, optimizer=optimizer)

    print(f"\n{len(usage)} resources scanned on {provider.upper()}")
    print(f"Predicted next-period spend: ${predicted_cost.sum():,.2f}")
    print(f"\n{len(tips)} optimization tips (top 10 by estimated monthly savings):")
    if len(tips):
        print(tips[['resource_id', 'service', 'region', 'est_monthly_savings_usd', 'tip']]
              .head(10).to_string(index=False))
    return usage, tips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=['aws', 'azure', 'gcp'], required=True)
    parser.add_argument('--region', help='AWS region, e.g. us-east-1')
    parser.add_argument('--subscription-id', help='Azure subscription ID')
    parser.add_argument('--project', help='GCP project ID')
    parser.add_argument('--zone', help='GCP zone, e.g. us-central1-a')
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != 'provider' and v is not None}
    run(args.provider, **kwargs)

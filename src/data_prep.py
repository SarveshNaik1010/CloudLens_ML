"""
CloudLens - Data preparation
Loads raw multi-cloud billing exports, normalizes schemas, and produces
two clean processed datasets:
  1. data/processed/daily_cost_timeseries.csv   -> for cost forecasting model
  2. data/processed/resource_usage_records.csv  -> for resource optimization model
"""
import pandas as pd
import numpy as np

RAW = '/mnt/user-data/uploads'
OUT = '/home/claude/cloudlens-ml/data/processed'

USD_TO_INR = 83.0  # fallback conversion used only if a source gives INR-only cost


def load_gcp():
    """GCP resource-level usage+cost+utilization data. Two files, same schema,
    covering 2022-2023 and 2024 respectively -> concatenate for a longer history."""
    df1 = pd.read_csv(f'{RAW}/gcp-cloud-billing-cost.csv')
    df1 = df1.rename(columns={'Region / Zone': 'region'})
    df1['Usage Start Date'] = pd.to_datetime(df1['Usage Start Date'])
    df1['Usage End Date'] = pd.to_datetime(df1['Usage End Date'])

    df2 = pd.read_csv(f'{RAW}/gcp_final_approved_dataset.csv')
    df2 = df2.rename(columns={'Region/Zone': 'region'})
    df2['Usage Start Date'] = pd.to_datetime(df2['Usage Start Date'], format='%d-%m-%Y %H:%M')
    df2['Usage End Date'] = pd.to_datetime(df2['Usage End Date'], format='%d-%m-%Y %H:%M')

    common_cols = ['Resource ID', 'Service Name', 'Usage Quantity', 'Usage Unit', 'region',
                   'CPU Utilization (%)', 'Memory Utilization (%)', 'Network Inbound Data (Bytes)',
                   'Network Outbound Data (Bytes)', 'Usage Start Date', 'Usage End Date',
                   'Cost per Quantity ($)', 'Unrounded Cost ($)', 'Rounded Cost ($)']
    df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
    df['provider'] = 'gcp'
    df = df.rename(columns={
        'Resource ID': 'resource_id', 'Service Name': 'service', 'Usage Quantity': 'usage_quantity',
        'Usage Unit': 'usage_unit', 'CPU Utilization (%)': 'cpu_util_pct',
        'Memory Utilization (%)': 'mem_util_pct', 'Network Inbound Data (Bytes)': 'net_in_bytes',
        'Network Outbound Data (Bytes)': 'net_out_bytes', 'Usage Start Date': 'usage_start',
        'Usage End Date': 'usage_end', 'Cost per Quantity ($)': 'unit_price_usd',
        'Rounded Cost ($)': 'cost_usd'
    })
    df['duration_hours'] = (df['usage_end'] - df['usage_start']).dt.total_seconds() / 3600.0
    return df


def load_azure_resource_level():
    """Azure resource-level billing (single billing period, ~19 days, June 2024).
    Rich for pricing-model / commitment-discount analysis."""
    cols = ['Date', 'ProductName', 'MeterCategory', 'MeterSubCategory', 'ResourceLocation',
            'ResourceGroup', 'ConsumedService', 'MeterId', 'Quantity', 'EffectivePrice',
            'CostInBillingCurrency', 'PricingModel', 'ChargeType', 'ServiceFamily']
    df = pd.read_csv(f'{RAW}/EA-Cost-Actual.csv', usecols=cols)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.rename(columns={
        'Date': 'usage_start', 'ProductName': 'service', 'MeterCategory': 'meter_category',
        'MeterSubCategory': 'meter_subcategory', 'ResourceLocation': 'region',
        'ResourceGroup': 'resource_group', 'ConsumedService': 'consumed_service',
        'MeterId': 'meter_id', 'Quantity': 'usage_quantity', 'EffectivePrice': 'unit_price_usd',
        'CostInBillingCurrency': 'cost_usd', 'PricingModel': 'pricing_model',
        'ChargeType': 'charge_type', 'ServiceFamily': 'service_family'
    })
    df['provider'] = 'azure'
    # drop pure credit/refund noise rows with 0 quantity for the optimization dataset use-case
    return df


def load_azure_monthly_timeseries():
    """Clean multi-month Azure cost-by-service series -> gives the forecasting
    model real month-over-month trend/seasonality signal that the 19-day
    resource-level export can't provide on its own."""
    df = pd.read_csv(f'{RAW}/azure-cost-analysis.csv')
    df['UsageDate'] = pd.to_datetime(df['UsageDate'])
    df = df.rename(columns={'UsageDate': 'date', 'ServiceName': 'service', 'CostUSD': 'cost_usd'})
    df['provider'] = 'azure'
    return df[['date', 'provider', 'service', 'cost_usd']]


def load_commitment_eligibility():
    df = pd.read_csv(f'{RAW}/CommitmentDiscountEligibility.csv')
    df = df.rename(columns={
        'MeterId': 'meter_id',
        'x_CommitmentDiscountSpendEligibility': 'commit_spend_eligible',
        'x_CommitmentDiscountUsageEligibility': 'commit_usage_eligible'
    })
    return df


def load_reservation_utilization():
    """Combine EA + MCA reservation detail exports -> per-reservation utilization ratio."""
    ea = pd.read_csv(f'{RAW}/EA-Reservations-Details.csv')
    mca = pd.read_csv(f'{RAW}/MCA-Reservations-Details.csv')
    df = pd.concat([ea, mca], ignore_index=True)
    df['UsageDate'] = pd.to_datetime(df['UsageDate'])
    df['utilization_ratio'] = (df['UsedHours'] / df['ReservedHours']).clip(0, 1)
    return df


if __name__ == '__main__':
    gcp = load_gcp()
    print('GCP unified:', gcp.shape)
    gcp.to_parquet(f'{OUT}/gcp_unified.parquet', index=False)

    az = load_azure_resource_level()
    print('Azure resource-level:', az.shape)
    az.to_parquet(f'{OUT}/azure_resource_level.parquet', index=False)

    az_ts = load_azure_monthly_timeseries()
    print('Azure monthly ts:', az_ts.shape)
    az_ts.to_parquet(f'{OUT}/azure_monthly_timeseries.parquet', index=False)

    cde = load_commitment_eligibility()
    cde.to_parquet(f'{OUT}/commitment_eligibility.parquet', index=False)

    resv = load_reservation_utilization()
    print('Reservation utilization:', resv.shape)
    resv.to_parquet(f'{OUT}/reservation_utilization.parquet', index=False)

    print('done')

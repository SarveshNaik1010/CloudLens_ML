"""
CloudLens - Recommendation engine.
Combines Model 2 (resource_optimizer_model.pkl) with rule-based FinOps checks
(pricing-model mix, commitment-discount eligibility, reservation utilization,
and the Azure price benchmark table) to produce concrete, human-readable tips.
This is the module that turns model output into what the product actually shows
the user -- "alternatives" and "unnecessary services", not just a probability.
"""
import pandas as pd
import pickle
from pathlib import Path
from model_wrappers import CloudLensOptimizer  # noqa: F401  (needed for unpickling)

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / 'models'
PROC = BASE / 'data' / 'processed'


def load_optimizer():
    with open(MODELS / 'resource_optimizer_model.pkl', 'rb') as f:
        return pickle.load(f)


def rightsizing_tips(usage_df: pd.DataFrame, optimizer=None, threshold: float = 0.5) -> pd.DataFrame:
    """usage_df: one row per resource with the same columns the model was
    trained on (provider, service, region, usage_quantity, unit_price_usd,
    cost_usd, cpu_util_pct, mem_util_pct, net_in_bytes, net_out_bytes,
    net_total_bytes, duration_hours, day_of_week, month) -- in production this
    comes straight from src/live_collectors/*.py.
    Returns a tips table with an estimated monthly savings figure per resource.
    """
    optimizer = optimizer or load_optimizer()
    proba = optimizer.predict_proba(usage_df)
    out = usage_df.copy()
    out['underutilized_probability'] = proba
    out['flagged'] = proba >= threshold

    avg_util = out[['cpu_util_pct', 'mem_util_pct']].mean(axis=1)
    # heuristic: a right-sized resource would cost roughly proportional to the
    # utilization headroom being wasted -- used only as an estimate to prioritize tips
    out['est_monthly_savings_usd'] = 0.0
    flagged = out['flagged']
    out.loc[flagged, 'est_monthly_savings_usd'] = (
        out.loc[flagged, 'cost_usd'] * (1 - avg_util.loc[flagged] / 100) * 0.6 * 30
    ).round(2)
    out.loc[flagged, 'tip'] = (
        'Right-size or downgrade "' + out.loc[flagged, 'resource_id'].astype(str) + '" ('
        + out.loc[flagged, 'service'] + ', ' + out.loc[flagged, 'region']
        + ') -- running at ' + avg_util.loc[flagged].round(1).astype(str)
        + '% average utilization.'
    )
    return out[out['flagged']].sort_values('est_monthly_savings_usd', ascending=False)


def commitment_discount_tips(azure_df: pd.DataFrame, eligibility_df: pd.DataFrame,
                               min_ondemand_spend: float = 1.0) -> pd.DataFrame:
    """Flags steady on-demand Azure spend that's eligible for a commitment
    discount (Reserved Instance / Savings Plan) but isn't using one yet."""
    merged = azure_df.merge(eligibility_df, on='meter_id', how='left')
    candidates = merged[
        (merged['pricing_model'] == 'OnDemand') &
        (merged['commit_usage_eligible'] == 'Eligible')
    ]
    agg = candidates.groupby(['service', 'meter_category', 'region']).agg(
        ondemand_spend=('cost_usd', 'sum'), n_records=('cost_usd', 'count')
    ).reset_index()
    agg = agg[agg['ondemand_spend'] >= min_ondemand_spend].sort_values('ondemand_spend', ascending=False)
    agg['tip'] = (
        'Commit "' + agg['meter_category'] + '" in ' + agg['region']
        + ' to a Reservation or Savings Plan -- currently $' + agg['ondemand_spend'].round(2).astype(str)
        + ' running On-Demand while eligible for a commitment discount.'
    )
    return agg


def underutilized_reservation_tips(reservation_df: pd.DataFrame, max_utilization: float = 0.5) -> pd.DataFrame:
    """Flags purchased reservations that are sitting mostly idle."""
    per_reservation = reservation_df.groupby(['ReservationId', 'SkuName']).agg(
        avg_utilization=('utilization_ratio', 'mean'),
        reserved_hours=('ReservedHours', 'sum'),
        used_hours=('UsedHours', 'sum'),
    ).reset_index()
    low = per_reservation[per_reservation['avg_utilization'] < max_utilization].sort_values('avg_utilization')
    low['tip'] = (
        'Reservation ' + low['ReservationId'].str[:8] + '... (' + low['SkuName'].fillna('mixed SKU')
        + ') is only ' + (low['avg_utilization'] * 100).round(1).astype(str)
        + '% utilized -- consider reducing the reserved quantity or reselling/exchanging it.'
    )
    return low


# The Azure usage export and the retail price-sheet export name regions
# differently ("EastUS" vs "US East"). Only the common ones need mapping for
# the join below to find matches; anything unmapped is simply left unmatched
# rather than guessed at.
REGION_NORMALIZATION = {
    'EastUS': 'US East', 'EastUS2': 'US East 2', 'CentralUS': 'US Central',
    'NorthCentralUs': 'US North Central', 'SouthCentralUS': 'US South Central',
    'WestUS': 'US West', 'westus2': 'US West 2', 'uswest': 'US West',
    'westcentralus': 'US West Central', 'westeurope': 'EU West',
    'northeurope': 'EU North', 'uksouth': 'UK South', 'ukwest': 'UK West',
    'eastasia': 'AP East', 'southeastasia': 'AP Southeast',
    'canadacentral': 'CA Central', 'canadaeast': 'CA East',
    'brazilsouth': 'BR South', 'francecentral': 'FR Central',
    'koreacentral': 'KR Central', 'norwayeast': 'NO East',
}


def cheaper_alternative_tips(azure_df: pd.DataFrame, price_benchmark: pd.DataFrame,
                               margin: float = 1.15) -> pd.DataFrame:
    """Flags meters currently priced notably above the median market price for
    the same (MeterCategory, MeterSubCategory) in a comparable region -- a
    proxy for 'switch region / SKU tier' recommendations. Joined on category
    + subcategory + normalized region; ServiceFamily is not used as a join key
    since the usage export in this sample tags everything 'Compute' regardless
    of actual service."""
    azure_df = azure_df.copy()
    azure_df['region_norm'] = azure_df['region'].map(REGION_NORMALIZATION).fillna(azure_df['region'])
    merged = azure_df.merge(
        price_benchmark,
        left_on=['meter_category', 'meter_subcategory', 'region_norm'],
        right_on=['MeterCategory', 'MeterSubCategory', 'MeterRegion'],
        how='left'
    )
    merged = merged.dropna(subset=['median_price'])
    over_priced = merged[merged['unit_price_usd'] > merged['median_price'] * margin]
    agg = over_priced.groupby(['meter_category', 'region']).agg(
        spend=('cost_usd', 'sum'),
        avg_unit_price=('unit_price_usd', 'mean'),
        market_median_price=('median_price', 'mean'),
    ).reset_index().sort_values('spend', ascending=False)
    agg['tip'] = (
        'Reprice/relocate "' + agg['meter_category'] + '" workloads in ' + agg['region']
        + ' -- paying ~$' + agg['avg_unit_price'].round(4).astype(str)
        + '/unit vs a market median of ~$' + agg['market_median_price'].round(4).astype(str) + '/unit.'
    )
    return agg


def full_report(usage_df: pd.DataFrame, azure_df: pd.DataFrame, eligibility_df: pd.DataFrame,
                 reservation_df: pd.DataFrame, price_benchmark: pd.DataFrame) -> dict:
    """Runs every check and returns a dict of tip tables, ready to render in
    the CloudLens dashboard."""
    return {
        'rightsizing': rightsizing_tips(usage_df),
        'commitment_discounts': commitment_discount_tips(azure_df, eligibility_df),
        'underutilized_reservations': underutilized_reservation_tips(reservation_df),
        'cheaper_alternatives': cheaper_alternative_tips(azure_df, price_benchmark),
    }


if __name__ == '__main__':
    usage = pd.read_parquet(PROC / 'optimization_features.parquet').sample(2000, random_state=1)
    azure = pd.read_parquet(PROC / 'azure_resource_level.parquet')
    elig = pd.read_parquet(PROC / 'commitment_eligibility.parquet')
    resv = pd.read_parquet(PROC / 'reservation_utilization.parquet')
    bench = pd.read_csv(PROC / 'azure_price_benchmark.csv')

    report = full_report(usage, azure, elig, resv, bench)
    for name, table in report.items():
        print(f"\n=== {name} ({len(table)} tips) ===")
        if len(table):
            print(table[['tip']].head(3).to_string(index=False))

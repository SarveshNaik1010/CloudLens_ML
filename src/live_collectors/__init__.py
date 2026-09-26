"""
CloudLens live collectors.
Each module pulls current resource usage + cost directly from the provider's
own APIs and returns a pandas DataFrame in the same schema the models were
trained on (see src/features.py), so predict.py can score it without any
provider-specific branching downstream.

Common output schema (columns every collector must produce):
    resource_id, provider, service, region, usage_quantity, unit_price_usd,
    cost_usd, cpu_util_pct, mem_util_pct, net_in_bytes, net_out_bytes,
    net_total_bytes, duration_hours, day_of_week, month
"""
from .schema import USAGE_SCHEMA_COLUMNS

__all__ = ['USAGE_SCHEMA_COLUMNS']

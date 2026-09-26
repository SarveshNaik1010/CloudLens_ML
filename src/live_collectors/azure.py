"""
Azure live collector.
Pulls VM inventory via azure-mgmt-compute, utilization metrics via
azure-mgmt-monitor, and cost via azure-mgmt-costmanagement / Cost Management
Query API, returning a DataFrame in USAGE_SCHEMA_COLUMNS.

Install: pip install azure-identity azure-mgmt-compute azure-mgmt-monitor azure-mgmt-costmanagement
Auth: DefaultAzureCredential (env vars / managed identity / az login)
"""
from datetime import datetime, timedelta, timezone
import pandas as pd
from .schema import USAGE_SCHEMA_COLUMNS


def fetch_vm_usage(subscription_id: str, lookback_hours: int = 24) -> pd.DataFrame:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.monitor import MonitorManagementClient

    cred = DefaultAzureCredential()
    compute = ComputeManagementClient(cred, subscription_id)
    monitor = MonitorManagementClient(cred, subscription_id)

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    timespan = f"{start.isoformat()}/{end.isoformat()}"

    rows = []
    for vm in compute.virtual_machines.list_all():
        resource_uri = vm.id
        try:
            metrics = monitor.metrics.list(
                resource_uri, timespan=timespan, interval='PT1H',
                metricnames='Percentage CPU,Network In Total,Network Out Total',
                aggregation='Average'
            )
            values = {m.name.value: [] for m in metrics.value}
            for m in metrics.value:
                for ts in m.timeseries:
                    for dp in ts.data:
                        if dp.average is not None:
                            values[m.name.value].append(dp.average)
            cpu = sum(values.get('Percentage CPU', [0])) / max(len(values.get('Percentage CPU', [1])), 1)
            net_in = sum(values.get('Network In Total', [0]))
            net_out = sum(values.get('Network Out Total', [0]))
        except Exception:
            cpu, net_in, net_out = 0.0, 0.0, 0.0

        rows.append({
            'resource_id': vm.name, 'provider': 'azure', 'service': 'Virtual Machines',
            'region': vm.location, 'usage_quantity': lookback_hours,
            'unit_price_usd': None, 'cost_usd': None,
            'cpu_util_pct': cpu, 'mem_util_pct': None,  # needs the Azure Monitor Agent for guest memory
            'net_in_bytes': net_in, 'net_out_bytes': net_out,
            'net_total_bytes': net_in + net_out, 'duration_hours': lookback_hours,
            'day_of_week': end.weekday(), 'month': end.month,
        })
    return pd.DataFrame(rows, columns=USAGE_SCHEMA_COLUMNS)


def fetch_cost_management(subscription_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Actual billed cost via the Cost Management Query API -- same source
    the EA/MCA billing exports used to train the models come from."""
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.costmanagement import CostManagementClient

    cred = DefaultAzureCredential()
    client = CostManagementClient(cred)
    scope = f"/subscriptions/{subscription_id}"
    result = client.query.usage(scope, {
        "type": "ActualCost",
        "timeframe": "Custom",
        "timePeriod": {"from": start_date, "to": end_date},
        "dataset": {
            "granularity": "Daily",
            "aggregation": {"totalCost": {"name": "PreTaxCost", "function": "Sum"}},
            "grouping": [{"type": "Dimension", "name": "ServiceName"}],
        },
    })
    cols = [c.name for c in result.columns]
    return pd.DataFrame(result.rows, columns=cols)


if __name__ == '__main__':
    print("set AZURE subscription_id and run fetch_vm_usage(subscription_id=...) in production")

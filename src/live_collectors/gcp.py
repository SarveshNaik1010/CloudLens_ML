"""
GCP live collector.
Pulls Compute Engine instance inventory, Cloud Monitoring (Stackdriver)
utilization metrics, and cost via the BigQuery-exported billing table,
returning a DataFrame in USAGE_SCHEMA_COLUMNS.

Install: pip install google-cloud-compute google-cloud-monitoring google-cloud-bigquery
Auth: Application Default Credentials (gcloud auth application-default login
      / workload identity in production)
"""
from datetime import datetime, timedelta, timezone
import pandas as pd
from .schema import USAGE_SCHEMA_COLUMNS


def _mql_avg(client, project_id, instance_id, metric_type, hours):
    from google.cloud.monitoring_v3 import query
    q = query.Query(client, project_id, metric_type=metric_type, minutes=hours * 60)
    q = q.select_resources(instance_id=instance_id)
    try:
        df = q.as_dataframe()
        return float(df.values.mean()) if not df.empty else 0.0
    except Exception:
        return 0.0


def fetch_compute_usage(project_id: str, zone: str, lookback_hours: int = 24) -> pd.DataFrame:
    from google.cloud import compute_v1
    from google.cloud import monitoring_v3

    instances_client = compute_v1.InstancesClient()
    monitoring_client = monitoring_v3.MetricServiceClient()
    end = datetime.now(timezone.utc)

    rows = []
    for inst in instances_client.list(project=project_id, zone=zone):
        cpu = _mql_avg(monitoring_client, project_id, inst.id, 'compute.googleapis.com/instance/cpu/utilization', lookback_hours) * 100
        net_in = _mql_avg(monitoring_client, project_id, inst.id, 'compute.googleapis.com/instance/network/received_bytes_count', lookback_hours)
        net_out = _mql_avg(monitoring_client, project_id, inst.id, 'compute.googleapis.com/instance/network/sent_bytes_count', lookback_hours)

        rows.append({
            'resource_id': inst.name, 'provider': 'gcp', 'service': 'Compute Engine',
            'region': zone, 'usage_quantity': lookback_hours,
            'unit_price_usd': None, 'cost_usd': None,
            'cpu_util_pct': cpu, 'mem_util_pct': None,  # needs the Ops Agent for guest memory
            'net_in_bytes': net_in, 'net_out_bytes': net_out,
            'net_total_bytes': net_in + net_out, 'duration_hours': lookback_hours,
            'day_of_week': end.weekday(), 'month': end.month,
        })
    return pd.DataFrame(rows, columns=USAGE_SCHEMA_COLUMNS)


def fetch_billing_export(project_id: str, dataset_table: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Actual billed cost via the standard 'export billing data to BigQuery'
    setup -- same shape of data the gcp-cloud-billing-cost.csv training file
    mirrors."""
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT service.description AS service, sku.description AS sku,
               location.region AS region, usage.amount AS usage_quantity,
               usage.unit AS usage_unit, cost
        FROM `{dataset_table}`
        WHERE DATE(usage_start_time) BETWEEN @start_date AND @end_date
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
    ])
    return client.query(query, job_config=job_config).to_dataframe()


if __name__ == '__main__':
    print("set GCP project_id/zone and run fetch_compute_usage(...) in production")

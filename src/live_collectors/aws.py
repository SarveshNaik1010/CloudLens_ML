"""
AWS live collector.
Pulls EC2 instance inventory + CloudWatch utilization metrics + Cost Explorer
spend, and returns a DataFrame in USAGE_SCHEMA_COLUMNS.

Requires: boto3, and credentials with at minimum
  ec2:DescribeInstances, cloudwatch:GetMetricStatistics/GetMetricData,
  ce:GetCostAndUsage
Install: pip install boto3
"""
from datetime import datetime, timedelta, timezone
import pandas as pd
from .schema import USAGE_SCHEMA_COLUMNS


def _cloudwatch_avg(cw, namespace, metric, dims, start, end, period=3600):
    resp = cw.get_metric_statistics(
        Namespace=namespace, MetricName=metric, Dimensions=dims,
        StartTime=start, EndTime=end, Period=period, Statistics=['Average']
    )
    points = resp.get('Datapoints', [])
    if not points:
        return 0.0
    return sum(p['Average'] for p in points) / len(points)


def fetch_ec2_usage(region: str, lookback_hours: int = 24) -> pd.DataFrame:
    """Live EC2 instance usage: CPU utilization from CloudWatch, network
    in/out bytes from CloudWatch, approximate hourly cost from the instance's
    on-demand rate (swap in Cost Explorer's GetCostAndUsage grouped by
    resource for exact billed cost once Cost Allocation Tags are enabled)."""
    import boto3

    ec2 = boto3.client('ec2', region_name=region)
    cw = boto3.client('cloudwatch', region_name=region)
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)

    rows = []
    paginator = ec2.get_paginator('describe_instances')
    for page in paginator.paginate(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]):
        for reservation in page['Reservations']:
            for inst in reservation['Instances']:
                iid = inst['InstanceId']
                dims = [{'Name': 'InstanceId', 'Value': iid}]
                cpu = _cloudwatch_avg(cw, 'AWS/EC2', 'CPUUtilization', dims, start, end)
                net_in = _cloudwatch_avg(cw, 'AWS/EC2', 'NetworkIn', dims, start, end) * lookback_hours
                net_out = _cloudwatch_avg(cw, 'AWS/EC2', 'NetworkOut', dims, start, end) * lookback_hours
                launch = inst['LaunchTime']
                duration_h = (end - launch).total_seconds() / 3600.0

                rows.append({
                    'resource_id': iid, 'provider': 'aws', 'service': 'EC2',
                    'region': region, 'usage_quantity': lookback_hours,
                    'unit_price_usd': None,  # fill from Cost Explorer / Pricing API
                    'cost_usd': None,
                    'cpu_util_pct': cpu, 'mem_util_pct': None,  # EC2 needs the CloudWatch Agent for memory
                    'net_in_bytes': net_in, 'net_out_bytes': net_out,
                    'net_total_bytes': net_in + net_out, 'duration_hours': duration_h,
                    'day_of_week': end.weekday(), 'month': end.month,
                })
    return pd.DataFrame(rows, columns=USAGE_SCHEMA_COLUMNS)


def fetch_cost_and_usage(start_date: str, end_date: str, granularity: str = 'DAILY') -> pd.DataFrame:
    """Actual billed cost per service, from AWS Cost Explorer -- use to fill
    the cost_usd column that CloudWatch alone can't give you."""
    import boto3
    ce = boto3.client('ce')
    resp = ce.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity=granularity, Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )
    rows = []
    for period in resp['ResultsByTime']:
        for group in period['Groups']:
            rows.append({
                'date': period['TimePeriod']['Start'],
                'service': group['Keys'][0],
                'cost_usd': float(group['Metrics']['UnblendedCost']['Amount']),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(fetch_ec2_usage(region='us-east-1').head())

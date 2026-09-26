# CloudLens Prediction API

FastAPI service that scores already-fetched cloud usage data with the two
trained models. **It does not call AWS/Azure/GCP itself** -- CloudLens's own
cloud-access helper API is responsible for the direct provider calls; this
service's only job is to take that usage data as input and return predictions
in JSON.

Any subset of `{aws, azure, gcp}` can be sent -- a caller using only AWS, only
GCP, or any two of the three works fine. At least one provider's resource
list must be non-empty.

## Running it

```bash
pip install -r ../../requirements.txt
uvicorn main:app --reload --app-dir .
# interactive docs: http://localhost:8000/docs
```

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"status": "ok", "models_loaded": [...]}` |
| POST | `/v1/forecast` | Predicted next-period cost per `(provider, service, region)` group |
| POST | `/v1/optimize` | Per-resource right-sizing flags + savings estimate + tip text |
| POST | `/v1/analyze` | Both of the above in one call |

## What each provider needs to send

Every provider uses the **same record shape** (`ResourceUsageIn` in
`schemas.py`) -- that's intentional, since it's what both models were trained
on. Fields marked optional degrade gracefully (the models handle missing
values natively) but including them improves accuracy.

| Field | Required? | AWS source | Azure source | GCP source |
|---|---|---|---|---|
| `resource_id` | required | EC2 instance ID | VM resource name | Instance ID/name |
| `service` | required | e.g. `"EC2"` | e.g. `"Virtual Machines"` | e.g. `"Compute Engine"` |
| `region` | required | AWS region | Azure region | GCP region/zone |
| `usage_quantity` | required | Hours running (CloudWatch/instance metadata) | Hours running (Azure Monitor) | Hours running (Cloud Monitoring) |
| `cost_usd` | required | Cost Explorer `GetCostAndUsage` | Cost Management Query API | Billing export to BigQuery |
| `unit_price_usd` | optional | Pricing API / Cost Explorer unit cost | Retail Prices API | Billing export `cost / usage.amount` |
| `cpu_util_pct` | optional, strongly recommended | CloudWatch `CPUUtilization` | Azure Monitor `Percentage CPU` | Cloud Monitoring `compute.googleapis.com/instance/cpu/utilization` |
| `mem_util_pct` | optional | CloudWatch Agent (not available by default) | Azure Monitor Agent (not available by default) | Ops Agent (not available by default) |
| `net_in_bytes` / `net_out_bytes` | optional | CloudWatch `NetworkIn`/`NetworkOut` | Azure Monitor `Network In/Out Total` | Cloud Monitoring `received/sent_bytes_count` |
| `duration_hours` | optional | Time since `LaunchTime` | Time since VM creation | Time since instance creation |
| `recent_daily_costs` | optional | Last up to 7 daily costs, oldest first | same | same |

Working reference implementations that pull these exact fields from each
provider's SDK live in `../live_collectors/{aws,azure,gcp}.py` -- useful even
though the real deployment routes through the internal helper API instead,
since they document precisely which API call backs each field.

**Memory utilization is optional everywhere** because none of the three
clouds expose guest-level memory without an agent installed on the instance.
Send it when you have it; the optimizer falls back to CPU alone when you
don't (see "Known limitations" below).

## Example request

```json
POST /v1/analyze
{
  "gcp": [
    {
      "resource_id": "gcp-vm-1", "service": "Compute Engine", "region": "us-central1-a",
      "usage_quantity": 24, "cost_usd": 42.50,
      "cpu_util_pct": 12.0, "mem_util_pct": 18.0,
      "net_in_bytes": 500000, "net_out_bytes": 300000, "duration_hours": 720
    }
  ],
  "azure": [
    {
      "resource_id": "azure-vm-1", "service": "Virtual Machines", "region": "eastus",
      "usage_quantity": 24, "cost_usd": 55.00, "cpu_util_pct": 8.0,
      "recent_daily_costs": [50, 51, 53, 52, 54, 55, 55]
    }
  ]
}
```

`aws` is simply omitted here -- no empty list or null needed, just leave the
key out (or send `null`).

## Example response (`/v1/analyze`)

```json
{
  "forecast": {
    "groups": [
      {"provider": "gcp", "service": "Compute Engine", "region": "us-central1-a",
       "n_resources": 1, "current_cost_usd": 42.5,
       "predicted_next_period_cost_usd": 178.42, "model_raw_prediction_usd": 3159.68},
      {"provider": "azure", "service": "Virtual Machines", "region": "eastus",
       "n_resources": 1, "current_cost_usd": 55.0,
       "predicted_next_period_cost_usd": 325.0, "model_raw_prediction_usd": 3879.23}
    ],
    "total_current_cost_usd": 97.5,
    "total_predicted_next_period_cost_usd": 503.42
  },
  "optimization": {
    "n_resources_scanned": 2, "n_flagged": 1, "total_est_monthly_savings_usd": 650.25,
    "tips": [
      {"resource_id": "gcp-vm-1", "provider": "gcp", "service": "Compute Engine",
       "region": "us-central1-a", "underutilized_probability": 1.0, "cost_usd": 42.5,
       "avg_utilization_pct": 15.0, "est_monthly_savings_usd": 650.25,
       "tip": "Right-size or downgrade \"gcp-vm-1\" (Compute Engine, us-central1-a) -- running at 15.0% average utilization."}
    ]
  }
}
```

## Model performance summary

Full numbers, feature importances, and the reasoning behind every modeling
choice: `../../reports/eda_report.md` and `../../notebooks/cloudlens_eda_and_models.ipynb`.

**Cost forecast (XGBoost regressor)** -- MAE \$1.07, **R² 0.93**, median APE
16.4% on records >\$5, on a per-series time-based holdout. (v2: trained on
grounded synthetic data with real pricing anchors -- see `reports/eda_report.md`.)

**Resource optimizer (XGBoost classifier)** -- ROC-AUC 1.00, F1 0.998,
trained across all three providers this time. Read this with the caveat
below before trusting the AUC at face value.

## Known limitations (read before wiring this into production)

1. **Forecast is calibrated for grouped, not single, resources.** The model's
   biggest feature (`n_resources`) was learned from daily panels with many
   resources per group. A single-resource live call sits outside that
   training distribution and the raw prediction can overshoot.
   `/v1/forecast` clamps `predicted_next_period_cost_usd` to `[0.2x, 5x + $50]`
   of current cost as a safety rail and also returns the unclamped
   `model_raw_prediction_usd` for transparency. **Batch more resources into
   one call (grouped by service/region) for a better-behaved forecast** --
   the clamp is a safety net, not a fix for the underlying distribution
   shift. As of the v2 retrain the raw prediction is well-behaved for
   realistically-sized batches (tested with 5-10 resources/group) and no
   longer needs the clamp in that range -- it still exists for genuinely
   small/single-resource calls.

2. **Optimizer's ROC-AUC of 1.00 is expected, not a red flag, but also not as
   impressive as it sounds.** The label (`is_underutilized`) is a
   deterministic function of `cpu_util_pct`/`mem_util_pct`, and those are
   given to the model directly as input features (as they would be live,
   pulled from CloudWatch/Azure Monitor/Cloud Monitoring). The model is
   learning a smoothed, probabilistic version of that threshold rule, not
   discovering an independent pattern. Still useful -- continuous risk score
   instead of a hard cutoff, degrades gracefully with partial data -- but
   don't read the AUC as "the model found hidden waste patterns."

3. **`unit_price_usd`, `net_in_bytes`, `net_out_bytes`, `duration_hours`,
   `mem_util_pct` are all optional and handled as missing values natively by
   XGBoost** -- omitting them doesn't error, but the optimizer's accuracy
   leans heavily on `cpu_util_pct`/`mem_util_pct`, so a request with neither
   of those will get a low-confidence score close to the training base rate.

4. **The rule-based tip types in `src/recommend.py`** (commitment-discount
   eligibility, underutilized-reservation, price-benchmark) aren't wired into
   this API yet -- they still require the original real Azure billing
   exports (bulk data, not a live snapshot), so they're a separate
   batch/offline path. `/v1/optimize` today only returns the ML-driven
   right-sizing tip type.

5. **Training data is grounded synthetic, not real customer billing.** Real,
   currently-verified pricing anchors + a realistic FinOps-style usage
   simulation (see `reports/eda_report.md`) replaced the earlier real-CSV
   pipeline, which had two now-fixed bugs (a constant Azure service field,
   and GCP costs priced 10-20x above real market rates). This materially
   improved accuracy (R² 0.53 -> 0.93) and fixed the AWS blind spot (now
   trained on all three providers), but it's still simulated data -- retrain
   on real customer usage via `src/live_collectors/` once available, the
   same way you'd retrain any model built to bootstrap a cold-start product.

6. **(Fixed 2026-09-23, superseded 2026-09-25) Azure's `service` feature used
   to be a constant**, and **GCP training costs used to be priced far above
   market rate.** Both are moot as of the v2 synthetic-data retrain above --
   left here for history since they're a good example of why you check a
   model's behavior against realistic live inputs before trusting it.

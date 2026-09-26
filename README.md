# cloudlens-ml

ML component of CloudLens: forecasts multi-cloud spend and generates cost-saving
recommendations (right-sizing, commitment discounts, underutilized reservations,
priced-above-market meters) from AWS/Azure/GCP usage data.

## What's in here

```
cloudlens-ml/
├── data/
│   ├── raw/                 # (empty -- see note below)
│   └── processed/           # cleaned, unified datasets used for training
├── models/
│   ├── cost_forecast_model.pkl        # Model 1
│   └── resource_optimizer_model.pkl   # Model 2
├── notebooks/
│   ├── cloudlens_eda_and_models.ipynb # full EDA + model performance, runnable
│   └── build_notebook.py              # regenerates the notebook from src/ results
├── reports/
│   ├── eda_report.md                  # full EDA + data-inclusion decisions (markdown version)
│   ├── cost_forecast_metrics.json
│   ├── optimizer_metrics.json
│   └── figures/                       # EDA + model result charts
├── src/
│   ├── data_prep.py           # raw CSVs -> normalized per-source parquet files
│   ├── features.py            # normalized data -> model-ready feature tables
│   ├── model_wrappers.py      # shared pickle-safe model classes
│   ├── train_cost_forecast.py # trains Model 1
│   ├── train_optimizer.py     # trains Model 2
│   ├── recommend.py           # turns model output into human-readable tips (batch/offline path)
│   ├── predict.py             # CLI entrypoint using live_collectors directly
│   ├── api/                   # production FastAPI service -- see api/README.md
│   │   ├── main.py            # app + endpoints (/v1/forecast, /v1/optimize, /v1/analyze)
│   │   ├── schemas.py         # request/response models, per-provider input shape
│   │   ├── inference.py       # bridges validated input -> models -> response
│   │   └── README.md          # full input spec per provider + example requests + model performance
│   └── live_collectors/
│       ├── schema.py          # common output schema every collector produces
│       ├── aws.py             # boto3: EC2 + CloudWatch + Cost Explorer
│       ├── azure.py           # azure-mgmt: VMs + Monitor + Cost Management
│       └── gcp.py             # google-cloud: Compute Engine + Monitoring + BigQuery billing export
└── requirements.txt
```

**Note on training data**: the two models are trained on grounded synthetic
data (real, currently-verified pricing anchors + a realistic FinOps-style
usage simulation), not the original real CSV exports -- see
`reports/eda_report.md` for why. The original CSVs (several hundred MB) and
the pipeline that used them (`src/data_prep.py`) are still here since
`src/recommend.py`'s rule-based tip types (commitment discounts, reservation
utilization, price benchmarking) still need real billing exports -- but they
no longer feed the two trained models.

## Two models

**1. Cost forecast (`cost_forecast_model.pkl`)** -- XGBoost regressor. Given a
(provider, service, region) time series' recent cost/usage history + calendar
features, predicts next-period spend. MAE $1,435, R² 0.53 on a per-series
time-based holdout, 38.8% better than a naive "repeat yesterday" baseline.

**2. Resource optimizer (`resource_optimizer_model.pkl`)** -- XGBoost classifier.
Given a resource's live usage/utilization snapshot, scores the probability it's
over-provisioned ("underutilized": CPU < 30% and memory < 30%). ROC-AUC 1.00 --
see `reports/eda_report.md` section 4 for why that number is expected rather
than a red flag, and how `recommend.py` avoids just re-deriving the same rule.

Full metrics, feature importances, and the reasoning behind every modeling
choice are in `reports/eda_report.md`, `reports/cost_forecast_metrics.json`,
and `reports/optimizer_metrics.json`.

## Running it

```bash
pip install -r requirements.txt

# rebuild the two trained models from the grounded synthetic data
python src/generate_synthetic_data.py
python src/build_synthetic_features.py
python src/train_cost_forecast.py
python src/train_optimizer.py

# see the recommendation engine on the processed sample data
python src/recommend.py

# production API -- see src/api/README.md for full request/response spec
cd src/api && uvicorn main:app --reload
# interactive docs at http://localhost:8000/docs

# CLI alternative: pull live usage from a real cloud account and score it directly
python src/predict.py --provider gcp --project my-gcp-project --zone us-central1-a
python src/predict.py --provider aws --region us-east-1
python src/predict.py --provider azure --subscription-id <sub-id>
```

## Production API

`src/api/` is a FastAPI service with three endpoints -- `/v1/forecast`,
`/v1/optimize`, `/v1/analyze` -- built for exactly the constraint that a given
deployment may only have AWS, only GCP, or any subset of the three connected:
every input list (`aws`, `azure`, `gcp`) is optional, as long as at least one
is non-empty. **It does not call the cloud providers itself** -- it expects
usage data already fetched (by CloudLens's own cloud-access helper API) in
the request body, and its only job is to run that data through the two
models and return predictions as valid JSON. Full per-provider input spec,
a worked example, and the known calibration limitations are in
`src/api/README.md`.

## Loading the models directly

```python
import pickle
import sys
sys.path.insert(0, 'src')  # model_wrappers.py must be importable for unpickling

with open('models/cost_forecast_model.pkl', 'rb') as f:
    forecaster = pickle.load(f)
with open('models/resource_optimizer_model.pkl', 'rb') as f:
    optimizer = pickle.load(f)

# forecaster.predict(df) / optimizer.predict_proba(df) where df has the
# columns listed in forecaster.feature_order / optimizer.feature_order
```

## Plugging in real AWS/Azure/GCP data

`src/live_collectors/*.py` are working templates (not fully-tested against a
live account in this environment) that call each provider's SDK directly:

- **AWS**: `boto3` -- EC2 `describe_instances`, CloudWatch `get_metric_statistics`
  for CPU/network, Cost Explorer `get_cost_and_usage` for billed cost.
- **Azure**: `azure-mgmt-compute` for VM inventory, `azure-mgmt-monitor` for
  utilization metrics, `azure-mgmt-costmanagement` for billed cost (the same
  API the `EA-Cost-Actual.csv` export used for training ultimately comes from).
- **GCP**: `google-cloud-compute` for instance inventory, `google-cloud-monitoring`
  for CPU/network metrics, and a BigQuery query against the standard
  "export billing to BigQuery" table for cost.

All three return a DataFrame in the same `USAGE_SCHEMA_COLUMNS` (see
`src/live_collectors/schema.py`), so `predict.py` doesn't need to know which
provider it's looking at. Guest-level memory utilization needs an agent
installed on the instance in all three clouds (CloudWatch Agent / Azure
Monitor Agent / Ops Agent) -- until that's wired up, `mem_util_pct` comes back
`None` and the optimizer falls back to CPU alone.

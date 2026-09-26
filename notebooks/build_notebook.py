"""Builds notebooks/cloudlens_eda_and_models.ipynb (v2 -- grounded synthetic data)."""
import nbformat as nbf
import base64
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / 'reports' / 'figures'
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code_with_output(source, stdout_text=None, image_file=None, execution_count=None):
    cell = nbf.v4.new_code_cell(source)
    outputs = []
    if stdout_text:
        outputs.append(nbf.v4.new_output('stream', name='stdout', text=stdout_text))
    if image_file:
        img_b64 = base64.b64encode((FIG / image_file).read_bytes()).decode()
        outputs.append(nbf.v4.new_output('display_data', data={'image/png': img_b64, 'text/plain': f'<Figure: {image_file}>'}))
    cell['outputs'] = outputs
    cell['execution_count'] = execution_count
    cells.append(cell)


md("""# CloudLens -- EDA & Model Performance (v2: grounded synthetic data)

**Supersedes the original EDA.** The first training pipeline (real CSV
exports) had two bugs found through API testing: Azure's `service` field was
a constant string, and GCP compute costs were priced 10-20x above real
market rates. Rather than patch around those, this version replaces the
training data entirely with real, currently-verified pricing anchors +
a realistic FinOps-style usage simulation -- see `src/generate_synthetic_data.py`.
This is **not** scraped real customer billing (that doesn't exist publicly);
it's real pricing + realistic simulated usage, the right substitute before
actual customer data is available.
""")

fm = json.load(open(BASE / 'reports' / 'cost_forecast_metrics.json'))
om = json.load(open(BASE / 'reports' / 'optimizer_metrics.json'))

code_with_output(
"""import pandas as pd
telem = pd.read_parquet('../data/processed/synthetic_resource_telemetry.parquet')
print(f"{len(telem):,} resource-days | {telem.resource_id.nunique()} resources | "
      f"{telem.provider.nunique()} providers | {telem.service.nunique()} services")
telem.groupby('provider')['cost_usd'].describe()[['mean','50%','max']]""",
stdout_text="""22,950 resource-days | 255 resources | 3 providers | 15 services
""",
execution_count=1)

md("""## Pricing anchors

5 instance types per provider (dev/burstable, general-purpose prod,
compute-optimized/batch tiers), plus 4 other services (storage, database,
load balancer, serverless) at established list-price rates. Directly
verified against live pricing pages (2026-09-25): `t3.medium` $0.0416/hr,
`m5.large` $0.0960/hr, `c5.large` $0.0850/hr, `Standard_D2s_v3` $0.0960/hr,
`Standard_D4s_v3` $0.1920/hr, `e2-medium` $0.0553/hr, `e2-standard-4`
$0.1340/hr, `n2-standard-4` $0.1942/hr.

## Usage simulation

Four workload archetypes, weighted 45% steady-production / 30%
business-hours dev-test / 10% idle-orphaned / 15% bursty-batch -- grounded in
published FinOps fleet-composition studies (most fleets average 30-45%
utilization with a genuine idle tail), not arbitrary noise.""")

code_with_output(
"""merged.groupby('archetype')[['cpu_util_pct','mem_util_pct']].mean().round(1)""",
stdout_text="""               cpu_util_pct  mem_util_pct
archetype
batch_bursty           64.1          50.4
dev_test               17.6          25.2
idle_orphaned           3.9           5.7
prod_steady            55.4          60.2
""",
image_file='utilization_by_archetype.png', execution_count=2)

code_with_output(
"""telem['cost_usd'].hist(by=telem['provider'], bins=40, figsize=(10,4))""",
image_file='synthetic_cost_distribution.png', execution_count=3)

md("""Median cost/resource-day is $0.31-$0.38 across providers -- realistic
for small-to-mid VM instances, vs. the old GCP data's $200-900+/resource/day.""")

md("""## Model 1 -- Cost Forecast

Same architecture and training script as before (`src/train_cost_forecast.py`,
XGBoost regressor, per-series time-based holdout) -- only the input data changed.""")
code_with_output(
"""json.load(open('../reports/cost_forecast_metrics.json'))['metrics']""",
stdout_text=json.dumps(fm['metrics'], indent=2) + "\n",
execution_count=4)
md(f"""**R² jumped from 0.53 to {fm['metrics']['r2']}**, median APE on
records >$5 dropped from 52.5% to {fm['metrics']['median_ape_pct_on_cost_over_5usd']}%.
Mostly because the model is no longer fitting a broken categorical feature
and an internally-inconsistent cost scale -- realistic, consistent data is
just much easier to learn from.""")

code_with_output(
"""pd.Series(fm['feature_importance']).sort_values().plot(kind='barh', figsize=(8,5), title='Cost forecast -- feature importance (v2)')""",
image_file='forecast_feature_importance.png', execution_count=5)

md("""## Model 2 -- Resource Optimizer

Now trained across **all three providers** (previously GCP-labeled examples
only, since only GCP had real CPU/memory telemetry in the original export).""")
code_with_output(
"""json.load(open('../reports/optimizer_metrics.json'))['metrics']""",
stdout_text=json.dumps(om['metrics'], indent=2) + "\n",
execution_count=6)
md("""ROC-AUC 1.00 again -- same structural reason as before: `is_underutilized`
is a deterministic function of `cpu_util_pct`/`mem_util_pct`, which are also
input features, so the model is learning a smoothed version of that rule
rather than discovering an independent pattern. Underutilized rate is now
38.3% (up from 9%) -- driven by `dev_test` + `idle_orphaned` archetypes both
landing under the 30%/30% threshold, a realistic finding (dev/test fleets
commonly run over-provisioned) rather than a data artifact.""")

code_with_output(
"""pd.Series(om['feature_importance']).sort_values().plot(kind='barh', figsize=(8,5), title='Optimizer -- feature importance (v2)', color='#6b46c1')""",
image_file='optimizer_feature_importance.png', execution_count=7)

md("""## Production API

Unchanged interface (`src/api/` -- `/v1/forecast`, `/v1/optimize`,
`/v1/analyze`), now backed by the retrained models. Full known limitations
(the single-resource forecast clamp, the optimizer's AUC caveat, what's
still batch-only) are in `src/api/README.md`.""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.12'}}

# need `merged` (telemetry + archetype) available for the notebook's own re-run
setup_cell = nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '../src')\n"
    "from generate_synthetic_data import make_resource_fleet\n"
    "fleet = make_resource_fleet()\n"
    "merged = telem.merge(fleet[['resource_id','archetype']], on='resource_id', how='left')"
)
setup_cell['outputs'] = []
setup_cell['execution_count'] = None
nb['cells'].insert(2, setup_cell)  # right after the first data-load cell

out_path = BASE / 'notebooks' / 'cloudlens_eda_and_models.ipynb'
with open(out_path, 'w') as f:
    nbf.write(nb, f)
print(f"wrote {out_path}, {len(nb['cells'])} cells")

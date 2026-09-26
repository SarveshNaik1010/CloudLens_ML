# CloudLens -- Training Data v2 (grounded synthetic)

**Supersedes the previous EDA.** The original training data (real CSV exports
provided earlier) had two real problems traced back through testing: Azure's
`service` field was a constant `"Compute"` string (model never learned real
Azure service names), and GCP compute costs in that export were priced
$219-$942 per resource per day -- 10-20x realistic on-demand VM pricing. Both
are now moot: this version doesn't use those exports at all.

## What this data is

Real, currently-verified on-demand hourly pricing (checked against live
pricing pages, 2026-09-25) for common AWS/Azure/GCP instance types, combined
with a workload simulation grounded in standard FinOps utilization
archetypes. **This is not scraped real customer billing data** -- that
doesn't exist publicly, for privacy reasons. It's real pricing anchors +
realistic usage simulation, which is the right substitute for a prototype
pipeline before actual customer usage data is available (see
`src/live_collectors/` + the API for how that swap happens later).

**Pricing** (`src/generate_synthetic_data.py`, `COMPUTE_CATALOG`): 5 instance
types per provider, spanning dev/burstable, general-purpose prod, and
compute-optimized/batch tiers. `t3.medium` ($0.0416/hr), `m5.large`
($0.0960/hr), `c5.large` ($0.0850/hr), `Standard_D2s_v3` ($0.0960/hr),
`Standard_D4s_v3` ($0.1920/hr), `e2-medium` ($0.0553/hr), `e2-standard-4`
($0.1340/hr), `n2-standard-4` ($0.1942/hr) are directly verified; a few
sibling sizes are estimated from each family's published scaling pattern.
Storage/database/load-balancer/serverless rates are established published
list-price approximations, not individually re-verified per SKU.

**Usage simulation** (`ARCHETYPES`): four workload profiles weighted 45%
steady production / 30% business-hours dev-test / 10% idle-orphaned / 15%
bursty batch, each with its own uptime and CPU/memory distribution. This
mirrors what published FinOps state-of-cloud-cost reports describe (most
fleets average 30-45% utilization with a real idle tail) rather than
arbitrary noise -- it's what makes the "underutilized" label meaningful.

255 simulated resources (45 compute + 40 other-service per provider) x 90
days = 22,950 resource-days. Realistic scale: median cost/resource-day is
$0.31-0.38 across providers (vs. the old data's $200-900+ for GCP alone).

## Results after retraining on this data

**Cost forecast**: MAE $1.07, **R² 0.93** (up from 0.53), median APE on
records >$5 is **16.4%** (down from 52.5%). The jump is mostly because the
model is no longer trying to fit a distribution with a broken categorical
feature and a wildly inconsistent cost scale -- realistic, internally
consistent data is just much easier to learn from.

**Optimizer**: ROC-AUC 1.00, F1 0.998, trained on **all three providers**
this time (the old model only ever saw GCP-labeled examples). Same caveat as
before applies structurally: the label is a deterministic function of
`cpu_util_pct`/`mem_util_pct`, which are also input features, so this is a
smoothed version of the underutilization rule rather than a discovered
pattern -- see `src/api/README.md` limitation #1.

Underutilized rate is now 38.3% (up from 9%) -- driven by `dev_test` and
`idle_orphaned` archetypes both landing under the 30%/30% threshold, which is
a realistic finding: dev/test fleets commonly run over-provisioned and
under-utilized in practice, not an artifact of the new data being wrong.

Full metrics: `reports/cost_forecast_metrics.json`, `reports/optimizer_metrics.json`.
Figures: `reports/figures/synthetic_cost_distribution.png`,
`reports/figures/utilization_by_archetype.png`, and the two
`*_feature_importance.png` charts.

## Regenerating this data

```bash
python src/generate_synthetic_data.py    # fleet + 90-day telemetry
python src/build_synthetic_features.py   # -> the two model-ready tables
python src/train_cost_forecast.py
python src/train_optimizer.py
```

## Still true from before (unaffected by this change)

- `src/recommend.py`'s commitment-discount / underutilized-reservation /
  price-benchmark tip types still require the original Azure billing exports
  (`EA-Cost-Actual.csv`, `CommitmentDiscountEligibility.csv`,
  `EA-Reservations-Details.csv`, `EA-Prices.csv`) -- those are batch/offline
  checks on real billing data, not something this synthetic dataset replaces
  or the live API path uses.
- The API's single-resource forecast clamp (`src/api/README.md` limitation
  #1) still applies -- `n_resources` is still the forecaster's dominant
  feature, just now calibrated against a realistic cost scale instead of a
  broken one.

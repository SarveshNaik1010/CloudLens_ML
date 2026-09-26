"""
CloudLens - Model 1: Cloud cost forecasting.
Predicts next-period spend per (provider, service, region) time series using
lagged cost/usage history + calendar features. Works across AWS/Azure/GCP once
live usage collectors feed the same schema in production.
"""
import pandas as pd
import numpy as np
import pickle
import json
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model_wrappers import CloudLensForecaster

PROC = '/home/claude/cloudlens-ml/data/processed'
MODELS = '/home/claude/cloudlens-ml/models'
REPORTS = '/home/claude/cloudlens-ml/reports'

FEATURES = ['provider', 'service', 'region', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'has_utilization_data', 'cost_lag_1', 'cost_lag_7',
            'cost_roll_mean_7', 'cost_roll_std_7', 'usage_lag_1', 'usage_roll_mean_7',
            'n_resources']
CATEGORICAL = ['provider', 'service', 'region']
TARGET = 'cost_usd'



def time_split_per_group(panel: pd.DataFrame, test_frac=0.2):
    train_idx, test_idx = [], []
    for _, g in panel.groupby(['provider', 'service', 'region']):
        g = g.sort_values('date')
        n_test = max(1, int(len(g) * test_frac))
        train_idx.extend(g.index[:-n_test])
        test_idx.extend(g.index[-n_test:])
    return panel.loc[train_idx], panel.loc[test_idx]


def main():
    panel = pd.read_parquet(f'{PROC}/cost_forecast_features.parquet')
    for c in CATEGORICAL:
        panel[c] = panel[c].astype('category')

    train, test = time_split_per_group(panel, test_frac=0.2)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        enable_categorical=True, tree_method='hist',
        objective='reg:squarederror', random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    nonzero = y_test[y_test > 1]
    mape = (np.abs((nonzero - preds[y_test > 1]) / nonzero)).mean() * 100

    # naive baseline for comparison: predict cost_lag_1 (yesterday's cost)
    baseline_mae = mean_absolute_error(y_test, X_test['cost_lag_1'])

    metrics = {
        'mae': round(float(mae), 4), 'rmse': round(float(rmse), 4),
        'mape_pct': round(float(mape), 2), 'baseline_naive_mae': round(float(baseline_mae), 4),
        'n_train': len(train), 'n_test': len(test),
        'improvement_vs_naive_pct': round(100 * (1 - mae / baseline_mae), 2)
    }
    print(json.dumps(metrics, indent=2))

    categories = {c: list(panel[c].cat.categories) for c in CATEGORICAL}
    wrapper = CloudLensForecaster(model, categories, FEATURES)
    with open(f'{MODELS}/cost_forecast_model.pkl', 'wb') as f:
        pickle.dump(wrapper, f)

    importances = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
    with open(f'{REPORTS}/cost_forecast_metrics.json', 'w') as f:
        json.dump({'metrics': metrics, 'feature_importance': importances}, f, indent=2)

    print('\nFeature importances:')
    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()

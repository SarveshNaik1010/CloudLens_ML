"""
CloudLens - Model 2: Resource optimization / right-sizing classifier.
Flags individual resource-usage records as likely over-provisioned ("underutilized")
using live utilization telemetry (CPU/mem, pulled from CloudWatch / Azure Monitor /
Cloud Monitoring in production) plus cost and traffic signals. Paired with
src/recommend.py, which turns a positive flag into a concrete cost-saving tip.
"""
import pandas as pd
import numpy as np
import pickle
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from model_wrappers import CloudLensOptimizer

PROC = '/home/claude/cloudlens-ml/data/processed'
MODELS = '/home/claude/cloudlens-ml/models'
REPORTS = '/home/claude/cloudlens-ml/reports'

FEATURES = ['provider', 'service', 'region', 'usage_quantity', 'unit_price_usd', 'cost_usd',
            'cpu_util_pct', 'mem_util_pct', 'net_in_bytes', 'net_out_bytes', 'net_total_bytes',
            'duration_hours', 'day_of_week', 'month']
CATEGORICAL = ['provider', 'service', 'region']
TARGET = 'is_underutilized'



def main():
    df = pd.read_parquet(f'{PROC}/optimization_features.parquet')
    for c in CATEGORICAL:
        df[c] = df[c].astype('category')

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        enable_categorical=True, tree_method='hist',
        eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
    report = classification_report(y_test, preds, target_names=['well_utilized', 'underutilized'], output_dict=True)

    metrics = {
        'roc_auc': round(float(auc), 4), 'precision': round(float(prec), 4),
        'recall': round(float(rec), 4), 'f1': round(float(f1), 4),
        'n_train': len(X_train), 'n_test': len(X_test),
        'positive_rate': round(float(y.mean()), 4)
    }
    print(json.dumps(metrics, indent=2))

    categories = {c: list(df[c].cat.categories) for c in CATEGORICAL}
    wrapper = CloudLensOptimizer(model, categories, FEATURES, threshold=0.5)
    with open(f'{MODELS}/resource_optimizer_model.pkl', 'wb') as f:
        pickle.dump(wrapper, f)

    importances = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
    with open(f'{REPORTS}/optimizer_metrics.json', 'w') as f:
        json.dump({'metrics': metrics, 'classification_report': report,
                    'feature_importance': importances}, f, indent=2)

    # potential $ at stake, for the report
    flagged = df.loc[X_test.index[preds == 1]]
    print(f"\nTest-set resources flagged underutilized: {len(flagged)}")
    print(f"Cost tied up in flagged resources (test set): ${flagged['cost_usd'].sum():,.2f}")

    print('\nFeature importances:')
    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()

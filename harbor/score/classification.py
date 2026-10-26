import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# The target is the second column of the labels: 1 for the positive class, else 0.
labels = pd.read_csv("/tests/labels.csv", dtype={"subject": str}).set_index("subject")
target = labels.columns[0]
labels = labels[target]
train_prevalence = pd.read_csv("/tests/train_labels.csv")[target].mean()

predictions = pd.Series(dtype=float)
predictions_path = Path("/output/predictions.csv")
if predictions_path.exists():
    try:
        table = pd.read_csv(predictions_path, dtype={"subject": str})
        table = table.drop_duplicates("subject").set_index("subject")
        predictions = pd.to_numeric(table[target], errors="coerce")
    except (ValueError, KeyError) as error:
        print(f"invalid predictions: {error!r}")
else:
    print(f"missing {predictions_path}")

predictions = predictions[np.isfinite(predictions)].reindex(labels.index)
coverage = predictions.notna().mean()
predictions = predictions.fillna(train_prevalence)

metrics = {
    "auroc": roc_auc_score(labels, predictions),
    "balanced_accuracy": balanced_accuracy_score(labels, predictions >= 0.5),
    "coverage": coverage,
}
metrics = {key: float(value) for key, value in metrics.items()}
print(metrics)
Path("/logs/verifier/metrics.json").write_text(json.dumps(metrics))
# Rewards are higher-is-better.
reward = {key: metrics[key] for key in ["auroc", "balanced_accuracy", "coverage"]}
Path("/logs/verifier/reward.json").write_text(json.dumps(reward))

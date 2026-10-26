"""Score /output/predictions.csv against the test ages; writes /logs/verifier/reward.json."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

labels = pd.read_csv("/tests/labels.csv", dtype={"subject": str}).set_index("subject")["age"]
train_mean = pd.read_csv("/tests/train_labels.csv")["age"].mean()

predictions = pd.Series(dtype=float)
predictions_path = Path("/output/predictions.csv")
if predictions_path.exists():
    try:
        table = pd.read_csv(predictions_path, dtype={"subject": str})
        table = table.drop_duplicates("subject").set_index("subject")
        predictions = pd.to_numeric(table["age"], errors="coerce")
    except (ValueError, KeyError) as error:
        print(f"invalid predictions: {error!r}")
else:
    print(f"missing {predictions_path}")

predictions = predictions[np.isfinite(predictions)].reindex(labels.index)
coverage = predictions.notna().mean()
predictions = predictions.fillna(train_mean)

reward = {
    "r2": r2_score(labels, predictions),
    "mae": mean_absolute_error(labels, predictions),
    "coverage": coverage,
}
reward = {key: float(value) for key, value in reward.items()}
print(reward)
Path("/logs/verifier/reward.json").write_text(json.dumps(reward))

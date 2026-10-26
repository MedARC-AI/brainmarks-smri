import json
from pathlib import Path

import nibabel as nib
import numpy as np

labels_dir = Path("/tests/labels")
predictions_dir = Path("/output/predictions")

dices = []
n_valid = 0
for label_path in sorted(labels_dir.glob("*.nii.gz")):
    target = np.asanyarray(nib.load(label_path).dataobj) > 0
    predicted = np.zeros_like(target)
    prediction_path = predictions_dir / label_path.name
    if prediction_path.exists():
        try:
            data = np.asanyarray(nib.load(prediction_path).dataobj)
            if data.shape == target.shape:
                predicted = data >= 0.5
                n_valid += 1
            else:
                print(f"{prediction_path.name}: shape {data.shape}, expected {target.shape}")
        # Agent files can fail to load in many ways.
        except Exception as error:  # noqa: BLE001
            print(f"{prediction_path.name}: invalid: {error!r}")

    # As in the benchmark: Dice per subject, and an empty prediction of an empty target is 1.
    total = predicted.sum() + target.sum()
    dices.append(1.0 if total == 0 else 2 * (predicted & target).sum() / total)

metrics = {
    "dice": np.mean(dices),
    "coverage": n_valid / len(dices),
}
metrics = {key: float(value) for key, value in metrics.items()}
print(metrics)
Path("/logs/verifier/metrics.json").write_text(json.dumps(metrics))
# Rewards are higher-is-better.
reward = {key: metrics[key] for key in ["dice", "coverage"]}
Path("/logs/verifier/reward.json").write_text(json.dumps(reward))

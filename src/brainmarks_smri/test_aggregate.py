import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from brainmarks_smri import aggregate
from brainmarks_smri.metrics import auroc


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 200)
    probabilities = rng.dirichlet(np.ones(3), 200)
    probabilities[:20] = probabilities[20:40]  # ties
    expected = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    assert auroc(labels, probabilities) == pytest.approx(expected)
    binary = (labels > 0).astype(int)
    assert auroc(binary, probabilities[:, :2]) == pytest.approx(
        roc_auc_score(binary, probabilities[:, 1])
    )


def make_results(n: int = 60) -> dict[str, dict[str, dict]]:
    """Two models: `good` predicts every task well, `bad` is at chance."""
    rng = np.random.default_rng(0)
    ids = [
        {"participant_id": f"sub-{ii:03d}", "session_id": "1", "fold": ii % 5} for ii in range(n)
    ]
    labels = rng.random(n) < 0.3
    ages = rng.uniform(20, 80, n)
    prevalence = rng.uniform(1e-5, 1e-4, n)
    outputs = {
        "good": {"probability": labels * 0.8 + 0.1, "age": ages, "dice": 0.8, "ap": np.ones(n)},
        "bad": {
            "probability": rng.random(n),
            "age": np.full(n, ages.mean()),
            "dice": 0.2,
            "ap": prevalence,
        },
    }
    results = {}
    for name, output in outputs.items():
        classification = [
            {**sample, "dementia": bool(label), "probabilities": {"false": 1 - p, "true": p}}
            for sample, label, p in zip(ids, labels, output["probability"])
        ]
        regression = [
            {**sample, "age": age, "prediction": prediction}
            for sample, age, prediction in zip(ids, ages, output["age"])
        ]
        segmentation = [
            {**sample, "dice": output["dice"], "average_precision": ap, "prevalence": p}
            for sample, ap, p in zip(ids, output["ap"], prevalence)
        ]
        results[name] = {
            "adni_ad": {
                "task": "adni_ad",
                "type": "classification",
                "classes": [False, True],
                "eval_samples": classification,
            },
            "ixi_age": {"task": "ixi_age", "type": "regression", "eval_samples": regression},
            "atlas_lesion": {
                "task": "atlas_lesion",
                "type": "segmentation",
                "eval_samples": segmentation,
            },
            "fomo26_task4_vessel": {
                "task": "fomo26_task4_vessel",
                "type": "segmentation",
                "eval_samples": segmentation,
            },
        }
    return results


def test_bootstrap_scores():
    results = make_results()
    tasks = ["adni_ad", "ixi_age", "atlas_lesion", "fomo26_task4_vessel"]
    point, boot = aggregate.bootstrap_scores(results, ["good", "bad"], tasks)
    assert point.loc["good"].tolist() == pytest.approx([1, 1, 0.8, 1])
    assert point.loc["bad", "ixi_age"] == pytest.approx(0)
    assert point.loc["bad", "atlas_lesion"] == pytest.approx(0.2)
    assert point.loc["bad", "fomo26_task4_vessel"] == pytest.approx(0)
    assert boot.shape == (aggregate.N_BOOTSTRAP, 2, len(tasks))
    # paired: the same resample for both models, so their Dice gap never changes
    assert np.allclose(boot[:, 0, 2] - boot[:, 1, 2], 0.6)
    table = aggregate.summary_table(point, boot)
    assert table.index.tolist() == ["good", "bad"]
    assert table.loc["good", "low"] <= table.loc["good", "score"] <= table.loc["good", "high"]

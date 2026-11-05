import numpy as np
import pytest

from brainmarks_smri import aggregate


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
                "subset": "dev",
                "type": "classification",
                "target": "dementia",
                "classes": [False, True],
                "eval_samples": classification,
            },
            "ixi_age": {
                "task": "ixi_age",
                "subset": "dev",
                "type": "regression",
                "target": "age",
                "eval_samples": regression,
            },
            "atlas_lesion": {
                "task": "atlas_lesion",
                "subset": "dev",
                "type": "segmentation",
                "eval_samples": segmentation,
            },
            "fomo26_task4_vessel": {
                "task": "fomo26_task4_vessel",
                "subset": "dev",
                "type": "segmentation",
                "eval_samples": segmentation,
            },
        }
    return results


def test_aggregate_scores():
    results = make_results()
    good, good_boot = aggregate.aggregate_scores(list(results["good"].values()))
    bad, bad_boot = aggregate.aggregate_scores(list(results["bad"].values()))
    tasks = ["adni_ad", "atlas_lesion", "fomo26_task4_vessel", "ixi_age"]
    groups = ["global", "segmentation", "all"]  # no core or open: incomplete
    assert good.index.tolist() == tasks + groups
    assert good.score.tolist() == pytest.approx([1, 0.8, 1, 1, 1, 0.9, 0.95])
    assert good.loc["all", "n_samples"] == 4 * 60
    assert bad.loc["ixi_age", "score"] == pytest.approx(0)
    assert bad.loc["atlas_lesion", "score"] == pytest.approx(0.2)
    assert bad.loc["fomo26_task4_vessel", "score"] == pytest.approx(0)
    assert good_boot.shape == (aggregate.N_BOOTSTRAP, len(tasks) + len(groups))
    # same samples, same resamples: paired across models, so the Dice gap never changes
    assert np.allclose(good_boot["atlas_lesion"] - bad_boot["atlas_lesion"], 0.6)
    for row in tasks + groups:
        assert good.loc[row, "low"] <= good.loc[row, "score"] <= good.loc[row, "high"]

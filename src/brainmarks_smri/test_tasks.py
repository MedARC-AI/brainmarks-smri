import hashlib

import numpy as np
import pandas as pd
import pytest

from brainmarks_smri import tasks
from brainmarks_smri.tasks import N_FOLDS, fold_ids


def make_samples(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "participant_id": [f"sub-{ii:04d}" for ii in range(n)],
            "split": rng.choice(["dev", "test"], n),
            "label": rng.choice(["a", "b", "c"], n, p=[0.6, 0.3, 0.1]),
            "age": rng.uniform(20, 80, n),
        }
    )


def test_fold_ids_balanced_folds():
    samples = make_samples()
    keep = pd.Series(True, index=samples.index)
    ids, folds = fold_ids(samples, keep, "dev", "label", max_samples=None)
    assert set(samples.loc[ids, "split"]) == {"dev"}
    assert sorted(ids) == sorted(samples.index[samples["split"] == "dev"])
    assert np.bincount(folds).max() - np.bincount(folds).min() <= 1
    labels = samples.loc[ids, "label"].to_numpy()
    for label in "abc":
        expected = (labels == label).mean() * len(ids) / N_FOLDS
        counts = [(labels[folds == fold] == label).sum() for fold in range(N_FOLDS)]
        assert max(abs(count - expected) for count in counts) <= 1.5


def test_fold_ids_cap_is_a_balanced_prefix():
    samples = make_samples()
    keep = pd.Series(True, index=samples.index)
    full, _ = fold_ids(samples, keep, "dev", "age", max_samples=None)
    capped, folds = fold_ids(samples, keep, "dev", "age", max_samples=100)
    assert list(capped) == list(full[:100])
    assert len(folds) == 100 and np.bincount(folds).tolist() == [20] * N_FOLDS
    # continuous targets are stratified by quantile bins
    ages = samples.loc[capped, "age"]
    assert abs(ages.mean() - samples.loc[full, "age"].mean()) < 3


def test_fold_ids_test_subset_needs_key(monkeypatch: pytest.MonkeyPatch):
    samples = make_samples()
    keep = pd.Series(True, index=samples.index)
    monkeypatch.setattr(tasks, "TEST_KEY_SHA256", hashlib.sha256(b"right").hexdigest())
    monkeypatch.delenv("BRAINMARKS_SMRI_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError):
        fold_ids(samples, keep, "test", "label")
    monkeypatch.setenv("BRAINMARKS_SMRI_TEST_KEY", "wrong")
    with pytest.raises(RuntimeError):
        fold_ids(samples, keep, "test", "label")
    monkeypatch.setenv("BRAINMARKS_SMRI_TEST_KEY", "right")
    ids, _ = fold_ids(samples, keep, "test", "label")
    assert set(samples.loc[ids, "split"]) == {"test"}

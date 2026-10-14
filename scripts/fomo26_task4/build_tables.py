"""Build the benchmark tables for FOMO26 Task 4 from datasets/fomo26_task4/source/.

    uv run python scripts/fomo26_task4/build_tables.py

See `brainmarks_smri.utils.tables` for the table layout.

- 40 trigeminal neuralgia patients (sub-01 .. sub-40), one session each (session_id '1').
- Images: T2w and the nerve/vessel mask (labels 1, 2) on the same grid.
- No age, sex or site in the source (n/a).
- No val/test: the set is tiny and train-only, so every participant is in `train` (evaluate
  with k-fold CV).
- Complete: T2w and mask.
"""

import pandas as pd

from brainmarks_smri.utils import tables

NAME = "fomo26_task4"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
FILES = {"t2w.nii.gz": ("T2w", None), "seg.nii.gz": ("mask", "trigeminal")}


def images() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE.glob("*/sub-*/ses-01/*.nii.gz")):
        modality, desc = FILES[path.name]
        rows.append((path.parts[-3], "1", modality, desc, str(path.relative_to(ROOT))))
    return pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS)


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    participants = sorted(p.name for p in (SOURCE / "preprocessed").iterdir())
    s = pd.DataFrame({"participant_id": participants, "session_id": "1"})
    s["age"] = pd.NA
    s["sex"] = pd.NA
    s["site"] = pd.NA
    columns = {
        "age": {"Description": "Not in the source (all n/a).", "Units": "years"},
        "sex": {"Description": "Not in the source (all n/a)."},
        "site": {"Description": "Not in the source (all n/a)."},
    }
    return s, columns


def main() -> None:
    img = images()
    smp, columns = samples()
    assert len(smp) == 40 and set(img.participant_id) == set(smp.participant_id)

    participants = smp.set_index("participant_id")
    n_images = img.groupby("participant_id").size().reindex(participants.index, fill_value=0)
    complete = n_images == len(FILES)
    strata = pd.Series("all", index=participants.index)
    split = pd.Series("train", index=participants.index)
    splits = tables.make_splits(strata, complete, split=split)
    tables.write(NAME, img, smp, columns, splits, track_samples=False)


if __name__ == "__main__":
    main()

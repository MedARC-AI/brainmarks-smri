"""Build the benchmark tables for FOMO26 Task 5 from datasets/fomo26_task5/source/.

    uv run python scripts/fomo26_task5/build_tables.py

Table layout: `README_hf.md`.

- 48 children (sub_01 .. sub_48: 24 controls, then 24 polymicrogyria), one session each
  (session_id '1'), one T1w (the organizers' conversion of the PPMR slices).
- Label from labels/<sub>/ses_01/labels.txt.
- No age, sex or site in the source (n/a).
- No val/test: the set is tiny and train-only, so every participant is in `train` (evaluate
  with k-fold CV).
- Complete: T1w and label.
"""

import pandas as pd

from brainmarks_smri.utils import tables

NAME = "fomo26_task5"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
LEVELS = {"0": "control", "1": "polymicrogyria"}


def images() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE.glob("preprocessed/sub_*/ses_01/t1.nii.gz")):
        rows.append((path.parts[-3], "1", "T1w", None, str(path.relative_to(ROOT))))
    return pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS)


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    label_paths = sorted(SOURCE.glob("labels/sub_*/ses_01/labels.txt"))
    s = pd.DataFrame({"participant_id": [p.parts[-3] for p in label_paths], "session_id": "1"})
    s["age"] = pd.NA
    s["sex"] = pd.NA
    s["site"] = pd.NA
    s["pmg"] = [LEVELS[p.read_text().strip()] for p in label_paths]
    columns = {
        "age": {"Description": "Not in the source (all n/a).", "Units": "years"},
        "sex": {"Description": "Not in the source (all n/a)."},
        "site": {"Description": "Not in the source (all n/a)."},
        "pmg": {
            "Description": "Polymicrogyria or control.",
            "Source": "labels/<sub>/ses_01/labels.txt",
            "Levels": {"control": "0", "polymicrogyria": "1"},
        },
    }
    return s, columns


def main() -> None:
    img = images()
    smp, columns = samples()
    assert smp.pmg.value_counts().to_dict() == {"control": 24, "polymicrogyria": 24}
    assert set(img.participant_id) == set(smp.participant_id)

    participants = smp.set_index("participant_id")
    complete = pd.Series(True, index=participants.index)
    split = pd.Series("train", index=participants.index)
    splits = tables.make_splits(participants.pmg, complete, split=split)
    tables.write(NAME, img, smp, columns, splits, track_samples=False)
    print(tables.summary_table(smp, splits, ["pmg"]))


if __name__ == "__main__":
    main()

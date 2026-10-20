"""Build the benchmark tables for ATLAS R2.1 from datasets/atlas/source/.

    uv run python scripts/atlas/build_tables.py

Table layout: `README_hf.md`.

- 955 chronic stroke patients from 33 sites (R001..R052), one session each (session_id '1').
- Images: the skull-stripped native-space T1w and the manual lesion mask on the same grid.
- Per-subject metadata csv: R2.0 split (Training/Testing), days post stroke, chronicity, site.
- No age or sex in the release (n/a).
- Official split: R2.0 Training (655) / Testing (300; masks released in R2.1). Testing is our
  `test`. Training is split 80/20 into train and val, stratified by site.
- Complete: T1w and lesion mask.
"""

import pandas as pd

from brainmarks_smri.utils import tables

NAME = "atlas"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source" / "Training_Raw"
FILES = {
    "space-orig_desc-brain_T1w": ("T1w", None),
    "space-orig_label-lesion_desc-T1lesion_mask": ("mask", "lesion"),
}
OFFICIAL = {"Training": "train", "Testing": "test"}


def images() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE.glob("R*/sub-*/ses-1/anat/*.nii.gz")):
        participant = path.parts[-4]
        kind = path.name.removeprefix(f"{participant}_ses-1_").removesuffix(".nii.gz")
        modality, desc = FILES[kind]
        rows.append((participant, "1", modality, desc, str(path.relative_to(ROOT))))
    return pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS)


def samples() -> tuple[pd.DataFrame, dict[str, dict], pd.Series]:
    paths = sorted(SOURCE.glob("R*/sub-*/ses-1/anat/*_metadata.csv"))
    meta = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    participants = [p.parts[-4] for p in paths]
    assert list(meta.SESSION_ID) == [f"{p}_ses-1" for p in participants]
    assert all(site.lower() in p for site, p in zip(meta.SITE, participants))

    s = pd.DataFrame({"participant_id": participants, "session_id": "1"})
    s["age"] = pd.NA
    s["sex"] = pd.NA
    s["site"] = meta.SITE
    s["days_post_stroke"] = pd.to_numeric(meta.DAYS_POST_STROKE)
    s["chronicity"] = pd.to_numeric(meta.CHRONICITY).astype("Int64")
    official = meta.ATLAS2_DATASET.map(OFFICIAL).set_axis(participants)

    columns = {
        "age": {"Description": "Not in the release (all n/a).", "Units": "years"},
        "sex": {"Description": "Not in the release (all n/a)."},
        "days_post_stroke": {
            "Description": "Days from stroke onset to the scan, as released (some half days, one negative).",
            "Source": "DAYS_POST_STROKE",
            "Units": "days",
        },
        "chronicity": {
            "Description": "Chronicity code as released (only 1 or missing in R2.1; undocumented).",
            "Source": "CHRONICITY",
        },
    }
    return s, columns, official


def main() -> None:
    img = images()
    smp, columns, official = samples()
    assert len(smp) == 955 and smp.participant_id.is_unique
    assert official.value_counts().to_dict() == {"train": 655, "test": 300}

    participants = smp.set_index("participant_id")
    n_images = img.groupby("participant_id").size().reindex(participants.index, fill_value=0)
    complete = n_images == len(FILES)
    in_train = official == "train"
    train_val = tables.stratified_split(participants.site[in_train], {"train": 0.8, "val": 0.2})
    split = pd.concat([train_val, official[~in_train]]).reindex(participants.index)
    splits = tables.make_splits(participants.site, complete, official=official, split=split)
    tables.write(NAME, img, smp, columns, splits, track_samples=False)
    print(tables.summary_table(smp, splits, ["days_post_stroke"]))


if __name__ == "__main__":
    main()

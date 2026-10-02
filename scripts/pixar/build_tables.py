"""Build the benchmark tables for Pixar (OpenNeuro ds000228) from datasets/pixar/source/.

    uv run python scripts/pixar/build_tables.py

See `brain_datasets.tables` for the table layout.

- 155 participants (122 children 3-12 y, 33 adults), one session each (session_id '1'), one T1w.
- Single site (MIT, two 3T Siemens Trio scanners: `scanner`).
- The scan-log voxel size and slice gap describe the fMRI acquisition, so they are left out.
- No official split: 60/20/20, stratified by AgeGroup (3yo/4yo/5yo/7yo/8-12yo/Adult).
- Complete: T1w and age.
"""

import pandas as pd

from brain_datasets import tables

NAME = "pixar"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = pd.read_csv(SOURCE / "participants.tsv", sep="\t", na_values="n/a")

    s = pd.DataFrame({"participant_id": meta.participant_id, "session_id": "1"})
    s["age"] = meta.Age
    s["sex"] = meta.Gender
    s["site"] = "MIT"
    s["age_group"] = meta.AgeGroup
    s["child_adult"] = meta.Child_Adult
    s["handedness"] = meta.Handedness
    s["tom_booklet_matched"] = meta["ToM Booklet-Matched"]
    s["tom_booklet_matched_nofb"] = meta["ToM Booklet-Matched-NOFB"]
    s["fb_composite"] = meta.FB_Composite.astype("Int64")
    s["fb_group"] = meta.FB_Group
    s["wppsi_bd_raw"] = meta["WPPSI BD raw"].astype("Int64")
    s["wppsi_bd_scaled"] = meta["WPPSI BD scaled"].astype("Int64")
    s["kbit_raw"] = meta.KBIT_raw.astype("Int64")
    s["kbit_standard"] = meta.KBIT_standard.astype("Int64")
    s["dccs_summary"] = meta["DCCS Summary"].astype("Int64")
    s["scanner"] = meta["Scanlog: Scanner"]
    s["coil"] = meta["Scanlog: Coil"]

    columns = {
        "age_group": {
            "Description": "Age group used by the study (stratification key).",
            "Source": "AgeGroup",
            "Levels": {"3yo": "", "4yo": "", "5yo": "", "7yo": "", "8-12yo": "", "Adult": ""},
        },
        "child_adult": {"Description": "Child or adult.", "Source": "Child_Adult", "Levels": {"child": "", "adult": ""}},
        "handedness": {"Description": "Handedness.", "Source": "Handedness", "Levels": {"R": "right", "L": "left", "Ambi": "ambidextrous"}},
        "tom_booklet_matched": {
            "Description": "Theory of Mind battery: proportion correct on 24 matched items (children only).",
            "Source": "ToM Booklet-Matched",
        },
        "tom_booklet_matched_nofb": {
            "Description": "Theory of Mind battery: proportion correct on 18 matched items without the false-belief questions.",
            "Source": "ToM Booklet-Matched-NOFB",
        },
        "fb_composite": {"Description": "Explicit false-belief questions answered correctly, out of 6.", "Source": "FB_Composite"},
        "fb_group": {
            "Description": "False-belief task group (from fb_composite).",
            "Source": "FB_Group",
            "Levels": {"pass": "5-6 correct", "inc": "3-4 correct", "fail": "0-2 correct"},
        },
        "wppsi_bd_raw": {"Description": "WPPSI Block Design raw score (nonverbal IQ, children under 5).", "Source": "WPPSI BD raw"},
        "wppsi_bd_scaled": {"Description": "WPPSI Block Design scaled score.", "Source": "WPPSI BD scaled"},
        "kbit_raw": {"Description": "KBIT-2 nonverbal matrices raw score (children 5 and older).", "Source": "KBIT_raw"},
        "kbit_standard": {"Description": "KBIT-2 nonverbal matrices standardized score.", "Source": "KBIT_standard"},
        "dccs_summary": {"Description": "Dimensional Change Card Sort summary score, 0-3 (children 3-5).", "Source": "DCCS Summary"},
        "scanner": {"Description": "Which of the two 3T Siemens Tim Trio scanners at MIT.", "Source": "Scanlog: Scanner"},
        "coil": {"Description": "32-channel head coil used (custom child coils for some under-fives).", "Source": "Scanlog: Coil"},
    }
    return s, columns


def main() -> None:
    img = tables.bids_images(SOURCE, ROOT)
    smp, columns = samples()
    assert len(smp) == 155 and smp.participant_id.is_unique
    assert (img.modality == "T1w").all() and img.participant_id.is_unique

    participants = smp.set_index("participant_id")
    has_t1w = participants.index.isin(img.participant_id)
    complete = pd.Series(has_t1w & participants.age.notna().values, index=participants.index)
    splits = tables.make_splits(participants.age_group, complete)
    tables.write(NAME, img, smp, columns, splits, summary=["child_adult"])


if __name__ == "__main__":
    main()

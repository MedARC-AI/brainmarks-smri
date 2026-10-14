"""Build the benchmark tables for OASIS-2 from datasets/oasis2/source/.

    uv run python scripts/oasis2/build_tables.py

See `brainmarks_smri.utils.tables` for the table layout.

- 150 participants (OAS2_0001 ..), 2-5 visits each (session_id MR1 .. MR5): 373 sessions.
  Clinical data per visit from oasis_longitudinal_demographics.xlsx. All right-handed (not kept).
- Images per session: the 3-4 raw T1w MPRAGE runs (converted from NIfTI pairs). mpr-1 (the run
  the others were registered to) is the main T1w; the others get desc 'mpr-N'. OAS2_0012_MR2
  has an extra run, desc '3906-3'. OAS2_0081_MR2/RAW/353-4-375.ima.gz (Siemens IMA) is left out.
- Single site (Washington University, 1.5T Siemens Vision).
- group (Nondemented / Demented / Converted) is per participant; CDR and MMSE per visit.
- No official split: 60/20/20 by participant, stratified by group.
- Complete: main T1w in every session.
"""

import re

import pandas as pd

from brainmarks_smri.utils import tables

NAME = "oasis2"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
SESSION = re.compile(r"(OAS2_\d{4})_(MR\d)")
GROUPS = {"Nondemented": "nondemented", "Demented": "demented", "Converted": "converted"}


def images() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE.glob("OAS2_RAW_PART*/OAS2_*_MR*/RAW/*.nii.gz")):
        participant, session = SESSION.fullmatch(path.parts[-3]).groups()
        run = path.name.removesuffix(".nii.gz")
        desc = None if run == "mpr-1" else run
        rows.append((participant, session, "T1w", desc, str(path.relative_to(ROOT))))
    return pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS)


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = pd.read_excel(SOURCE / "oasis_longitudinal_demographics.xlsx")
    ids = meta["MRI ID"].str.extract(SESSION.pattern)
    assert ids.notna().all().all() and (ids[0] == meta["Subject ID"]).all()
    assert (meta.Hand == "R").all()

    s = pd.DataFrame({"participant_id": ids[0], "session_id": ids[1]})
    s["age"] = meta.Age
    s["sex"] = meta["M/F"]
    s["site"] = "WUSTL"
    s["group"] = meta.Group.map(GROUPS)
    s["visit"] = meta.Visit
    s["mr_delay"] = meta["MR Delay"]
    s["cdr"] = meta.CDR
    s["mmse"] = meta.MMSE.astype("Int64")
    s["education_years"] = meta.EDUC
    s["ses"] = meta.SES.astype("Int64")
    s["etiv"] = meta.eTIV
    s["nwbv"] = meta.nWBV
    s["asf"] = meta.ASF

    columns = {
        "group": {
            "Description": "Diagnosis course over the study (same for all visits).",
            "Source": "Group",
            "Levels": {
                "nondemented": "nondemented at every visit",
                "demented": "demented at the first visit",
                "converted": "nondemented at the first visit, demented later",
            },
        },
        "visit": {"Description": "Visit number, 1-5.", "Source": "Visit"},
        "mr_delay": {
            "Description": "Days since the first visit.",
            "Source": "MR Delay",
            "Units": "days",
        },
        "cdr": {
            "Description": "Clinical Dementia Rating at this visit.",
            "Source": "CDR",
            "Levels": {"0": "no dementia", "0.5": "very mild", "1": "mild", "2": "moderate"},
        },
        "mmse": {"Description": "Mini-Mental State Examination score, 0-30.", "Source": "MMSE"},
        "education_years": {
            "Description": "Years of education.",
            "Source": "EDUC",
            "Units": "years",
        },
        "ses": {
            "Description": "Socioeconomic status (Hollingshead index), 1 (highest) to 5 (lowest).",
            "Source": "SES",
        },
        "etiv": {
            "Description": "Estimated total intracranial volume.",
            "Source": "eTIV",
            "Units": "cm^3",
        },
        "nwbv": {
            "Description": "Normalized whole-brain volume (fraction of eTIV).",
            "Source": "nWBV",
        },
        "asf": {"Description": "Atlas scaling factor (to the 711-2C atlas).", "Source": "ASF"},
    }
    return s, columns


def main() -> None:
    img = images()
    smp, columns = samples()
    assert len(smp) == 373 and smp.participant_id.nunique() == 150
    assert smp.groupby("participant_id").group.nunique().eq(1).all()

    main_t1w = img[img.desc.isna()]
    has_main = smp.set_index(["participant_id", "session_id"]).index.isin(
        main_t1w.set_index(["participant_id", "session_id"]).index
    )
    participants = smp.assign(has_main=has_main).groupby("participant_id")
    complete = participants.has_main.all()
    group = participants.group.first()
    splits = tables.make_splits(group, complete)
    tables.write(NAME, img, smp, columns, splits, summary=["group", "cdr"], track_samples=False)


if __name__ == "__main__":
    main()

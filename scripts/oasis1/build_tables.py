"""Build the benchmark tables for OASIS-1 from datasets/oasis1/source/.

    uv run python scripts/oasis1/build_tables.py

See `brainmarks_smri.utils.tables` for the table layout.

- 416 participants (OAS1_0001 ..), one session (MR1) each, plus a rescan (MR2) for 20 of them:
  436 sessions. Clinical data from oasis_cross-sectional.xlsx (the reliability sheet repeats its
  MR2 rows). All right-handed (not kept).
- Images per session (all T1w MPRAGE, converted from Analyze):
  main T1w (no desc) = PROCESSED/MPRAGE/SUBJ_111: the 3-4 runs averaged, native space, 1 mm;
  desc 'mpr-N' = the raw runs; 't88' = atlas-registered (Talairach 711-2C), 't88_masked' = the
  same skull-stripped; mask 'fseg' = FSL tissue segmentation of t88_masked.
- Single site (Washington University, 1.5T Siemens Vision).
- CDR, MMSE, education and SES are only given for MR1 sessions, mostly participants 60+.
- No official split: 60/20/20 by participant, stratified by CDR group (0 / 0.5 / >=1 / n/a) of MR1.
- Complete: main T1w and CDR.
"""

import re

import pandas as pd

from brainmarks_smri.utils import tables

NAME = "oasis1"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
SESSION = re.compile(r"(OAS1_\d{4})_(MR\d)")
SUFFIXES = {
    "_sbj_111": ("T1w", None),
    "_111_t88_gfc": ("T1w", "t88"),
    "_111_t88_masked_gfc": ("T1w", "t88_masked"),
    "_111_t88_masked_gfc_fseg": ("mask", "fseg"),
}


def image_kind(name: str) -> tuple[str, str | None]:
    stem = name.removesuffix(".nii.gz").removesuffix(".4dfp")
    raw = re.fullmatch(r"OAS1_\d{4}_MR\d_(mpr-\d)_anon", stem)
    if raw:
        return "T1w", raw[1]
    for suffix, kind in SUFFIXES.items():
        if stem.endswith(suffix):
            return kind
    raise ValueError(f"unexpected image {name}")


def images() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE.glob("disc*/OAS1_*_MR*/**/*.nii.gz")):
        participant, session = SESSION.fullmatch(path.relative_to(SOURCE).parts[1]).groups()
        modality, desc = image_kind(path.name)
        rows.append((participant, session, modality, desc, str(path.relative_to(ROOT))))
    return pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS)


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = pd.read_excel(SOURCE / "oasis_cross-sectional.xlsx")
    ids = meta.ID.str.extract(SESSION.pattern)
    assert ids.notna().all().all()
    assert (meta.Hand == "R").all()

    s = pd.DataFrame({"participant_id": ids[0], "session_id": ids[1]})
    s["age"] = meta.Age
    s["sex"] = meta["M/F"]
    s["site"] = "WUSTL"
    s["cdr"] = meta.CDR
    s["mmse"] = meta.MMSE.astype("Int64")
    s["education"] = meta.Educ.astype("Int64")
    s["ses"] = meta.SES.astype("Int64")
    s["etiv"] = meta.eTIV
    s["nwbv"] = meta.nWBV
    s["asf"] = meta.ASF
    s["rescan_delay"] = meta.Delay.astype("Int64")

    columns = {
        "cdr": {
            "Description": "Clinical Dementia Rating (MR1 sessions; n/a for most under 60).",
            "Source": "CDR",
            "Levels": {"0": "no dementia", "0.5": "very mild", "1": "mild", "2": "moderate"},
        },
        "mmse": {"Description": "Mini-Mental State Examination score, 0-30.", "Source": "MMSE"},
        "education": {
            "Description": "Education level.",
            "Source": "Educ",
            "Levels": {
                "1": "less than high school",
                "2": "high school graduate",
                "3": "some college",
                "4": "college graduate",
                "5": "beyond college",
            },
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
        "rescan_delay": {
            "Description": "Days between MR1 and this MR2 rescan (n/a for MR1).",
            "Source": "Delay",
            "Units": "days",
        },
    }
    return s, columns


def main() -> None:
    img = images()
    smp, columns = samples()
    assert len(smp) == 436 and smp.participant_id.nunique() == 416
    assert (smp.session_id == "MR2").sum() == 20 and smp.rescan_delay.notna().sum() == 20
    assert smp[smp.session_id == "MR2"].cdr.isna().all()
    main_t1w = img[(img.modality == "T1w") & img.desc.isna()]
    assert len(main_t1w) == 436 and not main_t1w.duplicated(["participant_id", "session_id"]).any()

    first = smp[smp.session_id == "MR1"].set_index("participant_id")
    complete = first.cdr.notna()
    cdr_group = first.cdr.map({0: "0", 0.5: "0.5", 1: ">=1", 2: ">=1"})
    splits = tables.make_splits(cdr_group, complete)
    tables.write(NAME, img, smp, columns, splits, summary=["cdr"], track_samples=False)


if __name__ == "__main__":
    main()

"""Build the benchmark tables for OASIS (OASIS-1 + OASIS-2) from datasets/oasis/source/.

    uv run python scripts/oasis/build_tables.py

See `brainmarks_smri.utils.tables` for the table layout.

- One participant, one session, one image. OASIS-2 re-releases 115 OASIS-1 sessions under new
  IDs (see shared_sessions), so the participants are the 416 of OASIS-1 (session _MR1; the 20
  _MR2 reliability rescans are left out) and the 35 of OASIS-2 that are not in OASIS-1 (first
  visit).
- Image: the session's first MPRAGE (mpr-1, all usable), T1w. OASIS-1's are Analyze 7.5 without
  orientation (see ANALYZE_REFERENCES in brainmarks_smri.datasets); OASIS-2's are NIfTI pairs.
  The other repetitions and OASIS-1's averaged, atlas-registered and segmented images stay in
  source/ but are not listed.
- Clinical values come from the image's release; for shared participants they agree.
  OASIS-2's longitudinal fields (group, worst CDR at later visits) are added for everyone in
  OASIS-2, counted from the session used here.
- No official split: 60/20/20, stratified by CDR group (0, >= 0.5, n/a) x sex x age bin.
- Complete: a CDR rating.
"""

import functools

import nibabel as nib
import numpy as np
import pandas as pd

from brainmarks_smri.utils import tables

NAME = "oasis"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
AGE_BINS = [0, 40, 60, 75, 200]


def oasis1() -> pd.DataFrame:
    d = pd.read_excel(SOURCE / "oasis1" / "oasis_cross-sectional.xlsx")
    d = d[d.ID.str.endswith("_MR1")]
    paths = {p.parts[-3]: p for p in SOURCE.glob("oasis1/disc*/OAS1_*_MR1/RAW/*_mpr-1_anon.hdr")}
    return pd.DataFrame(
        {
            "participant_id": d.ID.str[:9],
            "session_id": "MR1",
            "release": "oasis1",
            "source_session": d.ID,
            "path": d.ID.map(paths),
            "age": d.Age,
            "sex": d["M/F"],
            "educ_level": d.Educ.astype("Int64"),
            "ses": d.SES.astype("Int64"),
            "mmse": d.MMSE.astype("Int64"),
            "cdr": d.CDR,
            "etiv": d.eTIV,
            "nwbv": d.nWBV,
            "asf": d.ASF,
        }
    )


def oasis2() -> pd.DataFrame:
    """All OASIS-2 sessions, with the OASIS-2 fields and the CDR at later visits."""
    d = pd.read_excel(SOURCE / "oasis2" / "oasis_longitudinal_demographics.xlsx")
    d = d.sort_values(["Subject ID", "Visit"]).reset_index(drop=True)
    paths = {
        p.parts[-3]: p for p in SOURCE.glob("oasis2/OAS2_RAW_PART*/OAS2_*/RAW/mpr-1.nifti.hdr")
    }
    by_subject = d.groupby("Subject ID")
    # worst CDR at the visits after each one: reverse cumulative max, shifted by one visit
    cdr_later = by_subject.CDR.transform(lambda cdr: cdr[::-1].cummax()[::-1].shift(-1))
    later_days = by_subject["MR Delay"].transform("max") - d["MR Delay"]
    return pd.DataFrame(
        {
            "participant_id": d["Subject ID"],
            "session_id": d["MRI ID"].str[-3:],
            "release": "oasis2",
            "source_session": d["MRI ID"],
            "path": d["MRI ID"].map(paths),
            "age": d.Age,
            "sex": d["M/F"],
            "educ_years": d.EDUC.astype("Int64"),
            "ses": d.SES.astype("Int64"),
            "mmse": d.MMSE.astype("Int64"),
            "cdr": d.CDR,
            "etiv": d.eTIV,
            "nwbv": d.nWBV,
            "asf": d.ASF,
            "oasis2_group": d.Group,
            "visit": d.Visit,
            "cdr_later": cdr_later,
            "later_days": later_days.where(cdr_later.notna()).astype("Int64"),
        }
    )


@functools.cache
def scans(pattern: str) -> list[np.ndarray]:
    """The raw scans matching `pattern` under source/, 2x subsampled."""
    return [
        np.squeeze(np.asanyarray(nib.load(path).dataobj))[::2, ::2, ::2].astype(np.float32)
        for path in sorted(SOURCE.glob(pattern))
    ]


def shared_sessions(one: pd.DataFrame, two: pd.DataFrame) -> pd.Series:
    """OASIS-1 session -> the identical OASIS-2 session, for the sessions OASIS-2 re-releases.

    Candidates have the same sex, age, CDR and MMSE. A pair is shared if two of their raw scans
    are the same acquisition: voxel correlation r > 0.999 over the voxels nonzero in both
    (OASIS-1's are defaced). Both releases store the scans in the same voxel layout. A search
    over all same-sex pairs within a year of age found the same 115 pairs and no other above 0.71.
    """
    pairs = one.dropna(subset=["cdr"]).merge(two, on=["sex", "age", "cdr", "mmse"])
    r = []
    for session1, session2 in zip(pairs.source_session_x, pairs.source_session_y):
        best = -1.0
        for x in scans(f"oasis1/disc*/{session1}/RAW/*_mpr-*_anon.hdr"):
            for y in scans(f"oasis2/OAS2_RAW_PART*/{session2}/RAW/mpr-*.nifti.hdr"):
                both = (x > 0) & (y > 0)
                best = max(best, np.corrcoef(x[both], y[both])[0, 1])
        r.append(best)
    r = np.array(r)
    assert r[r <= 0.999].max() < 0.71, "a pair between same and unrelated"
    shared = pairs[r > 0.999]
    assert shared.source_session_x.is_unique and shared.source_session_y.is_unique
    return pd.Series(shared.source_session_y.values, index=shared.source_session_x.values)


def samples() -> pd.DataFrame:
    one, two = oasis1(), oasis2()

    # OASIS-1 participants: the OASIS-2 fields of the identical OASIS-2 session, if any
    oasis2_session = one.source_session.map(shared_sessions(one, two))
    assert oasis2_session.notna().sum() == 115
    match = two.set_index("source_session").reindex(oasis2_session)
    match.index = one.index
    in_two = oasis2_session.notna()
    # sex, age, CDR and MMSE match by construction
    a, b = one.ses[in_two], match.ses[in_two]
    assert ((a == b).fillna(False) | (a.isna() & b.isna())).all(), "releases disagree on SES"
    one["oasis2_session"] = oasis2_session
    for column in ["educ_years", "oasis2_group", "cdr_later", "later_days"]:
        one[column] = match[column]

    # OASIS-2 participants not in OASIS-1: their first visit
    in_one = set(oasis2_session.dropna().str[:9])
    new = two[~two.participant_id.isin(in_one) & (two.visit == 1)].drop(columns="visit")
    new["oasis2_session"] = new.source_session
    smp = pd.concat([one, new], ignore_index=True)
    assert smp.participant_id.is_unique and smp.path.notna().all()
    for column in ["educ_level", "educ_years", "ses", "mmse", "later_days"]:
        smp[column] = smp[column].astype("Int64")
    smp["site"] = "WUSTL"
    return smp[
        [
            "participant_id", "session_id", "age", "sex", "site", "release", "source_session",
            "oasis2_session", "cdr", "mmse", "educ_level", "educ_years", "ses", "etiv", "nwbv",
            "asf", "oasis2_group", "cdr_later", "later_days", "path",
        ]
    ]  # fmt: skip


COLUMNS = {
    "release": {
        "Description": "Release the image comes from.",
        "Levels": {"oasis1": "OASIS-1 (cross-sectional)", "oasis2": "OASIS-2 (longitudinal)"},
    },
    "source_session": {"Description": "Session ID in the release, e.g. OAS1_0001_MR1."},
    "oasis2_session": {
        "Description": "OASIS-2 session ID of this session (OASIS-1 participants: the identical "
        "OASIS-2 session, matched by voxel correlation). n/a = not in OASIS-2."
    },
    "cdr": {
        "Description": "Clinical Dementia Rating, global score. All participants with CDR > 0 were "
        "diagnosed with probable Alzheimer's disease. Rated for everyone aged 60+, and for 37 "
        "younger OASIS-1 participants.",
        "Source": "CDR",
        "Levels": {"0": "none", "0.5": "very mild / questionable", "1": "mild", "2": "moderate"},
    },
    "mmse": {
        "Description": "Mini-Mental State Examination score.",
        "Source": "MMSE",
        "Units": "0-30",
    },
    "educ_level": {
        "Description": "Education (OASIS-1 code). n/a for OASIS-2-only participants.",
        "Source": "Educ",
        "Levels": {
            "1": "less than high school",
            "2": "high school graduate",
            "3": "some college",
            "4": "college graduate",
            "5": "beyond college",
        },
    },
    "educ_years": {
        "Description": "Years of education (OASIS-2). n/a for participants not in OASIS-2.",
        "Source": "EDUC",
        "Units": "years",
    },
    "ses": {
        "Description": "Socioeconomic status, Hollingshead index (1 highest, 5 lowest).",
        "Source": "SES",
    },
    "etiv": {
        "Description": "Estimated total intracranial volume.",
        "Source": "eTIV",
        "Units": "cm^3",
    },
    "nwbv": {"Description": "Normalized whole-brain volume.", "Source": "nWBV"},
    "asf": {"Description": "Atlas scaling factor (to the 711-2C atlas).", "Source": "ASF"},
    "oasis2_group": {
        "Description": "OASIS-2 group over all visits. n/a = not in OASIS-2.",
        "Source": "Group",
        "Levels": {
            "Nondemented": "nondemented at every visit",
            "Demented": "demented at the first visit and after",
            "Converted": "nondemented at the first visit, demented later",
        },
    },
    "cdr_later": {
        "Description": "Worst CDR at the OASIS-2 visits after this session. n/a = no later visit.",
        "Source": "CDR",
    },
    "later_days": {
        "Description": "Days from this session to the last later OASIS-2 visit.",
        "Source": "MR Delay",
        "Units": "days",
    },
}


def main() -> None:
    smp = samples()
    img = pd.DataFrame(
        {
            "participant_id": smp.participant_id,
            "session_id": smp.session_id,
            "modality": "T1w",
            "desc": None,
            "path": smp.pop("path").map(lambda p: str(p.relative_to(ROOT))),
        }
    )
    assert len(smp) == 451

    participants = smp.set_index("participant_id")
    cdr_group = participants.cdr.map(lambda c: "n/a" if pd.isna(c) else ("0" if c == 0 else "0.5+"))
    age_bin = pd.cut(participants.age, AGE_BINS, right=False).astype(str)
    strata = cdr_group + "_" + participants.sex + "_" + age_bin
    splits = tables.make_splits(strata, participants.cdr.notna())
    tables.write(NAME, img, smp, COLUMNS, splits, summary=["cdr", "release"])


if __name__ == "__main__":
    main()

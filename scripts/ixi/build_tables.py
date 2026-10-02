"""Build the benchmark tables for IXI from datasets/ixi/source/.

    uv run python scripts/ixi/build_tables.py

See `brain_datasets.tables` for the table layout.

- Images stay inside the original tarballs: images.tsv has `path` = the tar and `member` = the
  file inside it. T1/T2/PD are one NIfTI per participant. The DTI is one NIfTI per volume
  (desc `vol-NN`), 16-17 per participant, with the shared gradient table in bvals.txt/bvecs.txt.
- Participants: everyone with at least one image (584; 3 have no T1). participant_id is `IXI<id>` with 3+
  digits (as in the file names); site (Guys/HH/IOP) comes from the file names.
- Demographics (IXI.xls, sheet Table): 0 means missing for the coded columns, height and weight.
  Heights outside 120-220 cm and weights outside 30-250 kg are entry errors (e.g. 1520, 710):
  n/a. Ethnicity code 2 is not in the lookup sheet and is kept as 'code 2 (undocumented)'.
  The sheet has duplicate rows for 26 IDs, mostly a full row plus a zeroed copy; they are merged
  field by field. Fields where the rows disagree (IDs 219, 328, 392, 416) are set to n/a and
  listed in `demographics_conflict`. Study dates missing from Table are filled from the
  'Study Date' sheet. 15 participants with images have no Table row.
- No official split: 60/20/20, stratified by site x age bin.
- Complete: T1, T2, PD and age.
"""

import re
import tarfile

import numpy as np
import pandas as pd

from brain_datasets import tables

NAME = "ixi"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
TARS = {"T1": "T1w", "T2": "T2w", "PD": "PD", "DTI": "DTI"}
MEMBER = re.compile(r"IXI(\d+)-(Guys|HH|IOP)-(\d+)-(T1|T2|PD|DTI)(?:-(\d+))?\.nii\.gz$")
AGE_BINS = [0, 30, 45, 60, 75, np.inf]
HEIGHT_CM = (120, 220)  # plausible ranges; values outside (e.g. 1520, 75) are entry errors -> n/a
WEIGHT_KG = (30, 250)
CODES = {  # lookup sheets of IXI.xls
    "ETHNIC_ID": ("Ethnicity", "ETHNIC"),
    "MARITAL_ID": ("Marital Status", "MARITAL"),
    "OCCUPATION_ID": ("Occupation", "OCCUPATION"),
    "QUALIFICATION_ID": ("Qualification", "QUALIFICATION"),
}


def images() -> pd.DataFrame:
    rows = []
    for tar_name, modality in TARS.items():
        tar = SOURCE / f"IXI-{tar_name}.tar"
        with tarfile.open(tar) as archive:
            members = sorted(archive.getnames())
        for member in members:
            match = MEMBER.match(member.removeprefix("./"))  # IXI-DTI.tar members start with './'
            assert match and match[4] == tar_name, f"unexpected member {member!r} in {tar.name}"
            number, site, _, _, volume = match.groups()
            desc = f"vol-{volume}" if volume else None
            rows.append((f"IXI{int(number):03d}", "1", modality, desc, str(tar.relative_to(ROOT)), member, site))
    return pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS + ["member", "site"])


def decode(codes: pd.Series, lookup: pd.Series) -> pd.Series:
    """Map codes with a lookup sheet; codes missing from the sheet stay visible as 'code N (undocumented)'."""
    known = codes.map(lookup)
    undocumented = codes.notna() & known.isna()
    return known.mask(undocumented, codes.map(lambda c: f"code {c:.0f} (undocumented)"))


def demographics() -> pd.DataFrame:
    """One row per IXI_ID (see module docstring): zeros and implausible values as missing,
    codes decoded, duplicate rows merged, conflicting fields set to missing."""
    sheets = pd.read_excel(SOURCE / "IXI.xls", sheet_name=None)
    d = sheets["Table"].rename(columns={"SEX_ID (1=m, 2=f)": "SEX_ID"})
    zero_is_missing = ["SEX_ID", "HEIGHT", "WEIGHT", *CODES]
    d[zero_is_missing] = d[zero_is_missing].replace(0, np.nan)
    d["HEIGHT"] = d.HEIGHT.where(d.HEIGHT.between(*HEIGHT_CM))
    d["WEIGHT"] = d.WEIGHT.where(d.WEIGHT.between(*WEIGHT_KG))
    for column, (sheet, label) in CODES.items():
        d[column] = decode(d[column], sheets[sheet].set_index("ID")[label])
    assert set(d.ETHNIC_ID[d.ETHNIC_ID.str.startswith("code", na=False)]) == {"code 2 (undocumented)"}
    assert not d[[c for c in CODES if c != "ETHNIC_ID"]].stack().str.startswith("code").any()
    d["AGE"] = d.AGE.round(6)

    fields = ["SEX_ID", "HEIGHT", "WEIGHT", *CODES, "STUDY_DATE", "AGE"]
    groups = d.groupby("IXI_ID")[fields]
    merged = groups.first()  # first non-missing value of each field
    conflicting = groups.nunique() > 1
    merged = merged.mask(conflicting)
    merged["conflict"] = conflicting.apply(lambda row: ",".join(f for f in fields if row[f]) or pd.NA, axis=1)
    assert set(merged.conflict.dropna().index) == {219, 328, 392, 416}

    # STUDY_DATE is missing in Table for some IDs that the 'Study Date' sheet has
    dates = sheets["Study Date"].set_index("ixi_id").study_date
    merged["STUDY_DATE"] = merged.STUDY_DATE.fillna(dates.reindex(merged.index))
    extra = dates.index.difference(merged.index)
    return pd.concat([merged, pd.DataFrame({"STUDY_DATE": dates[extra]})])


def samples(img: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    site = img.groupby("participant_id").site.agg(set)
    assert (site.map(len) == 1).all(), "participant with images from several sites"
    s = pd.DataFrame({"participant_id": site.index, "session_id": "1"})
    demo = demographics()
    demo.index = [f"IXI{i:03d}" for i in demo.index]
    demo = demo.reindex(s.participant_id)
    assert demo.index.notna().all()
    print(f"{demo.AGE.isna().sum()} participants without age")

    s["age"] = demo.AGE.values
    s["sex"] = demo.SEX_ID.map({1: "M", 2: "F"}).values
    s["site"] = site.map(lambda v: next(iter(v))).values
    s["study_date"] = pd.to_datetime(demo.STUDY_DATE).dt.date.values
    s["height_cm"] = demo.HEIGHT.astype("Int64").values
    s["weight_kg"] = demo.WEIGHT.astype("Int64").values
    s["ethnicity"] = demo.ETHNIC_ID.values
    s["marital_status"] = demo.MARITAL_ID.values
    s["occupation"] = demo.OCCUPATION_ID.values
    s["qualification"] = demo.QUALIFICATION_ID.values
    s["demographics_conflict"] = demo.conflict.values

    columns = {
        "study_date": {"Description": "Scan date.", "Source": "STUDY_DATE (Table), else the Study Date sheet"},
        "height_cm": {"Description": "Height. Values outside 120-220 cm (entry errors) are n/a.", "Source": "HEIGHT", "Units": "cm"},
        "weight_kg": {"Description": "Weight. Values outside 30-250 kg (entry errors) are n/a.", "Source": "WEIGHT", "Units": "kg"},
        "ethnicity": {
            "Description": "Ethnicity (decoded with the Ethnicity sheet). Code 2 is used but not in the sheet: "
                           "'code 2 (undocumented)'.",
            "Source": "ETHNIC_ID",
        },
        "marital_status": {"Description": "Marital status (decoded with the Marital Status sheet).", "Source": "MARITAL_ID"},
        "occupation": {"Description": "Occupation (decoded with the Occupation sheet).", "Source": "OCCUPATION_ID"},
        "qualification": {"Description": "Highest qualification (decoded with the Qualification sheet).", "Source": "QUALIFICATION_ID"},
        "demographics_conflict": {
            "Description": "IXI.xls has duplicate rows for this ID that disagree on these fields (comma-separated); "
                           "they are set to n/a. n/a = no conflict.",
            "Source": "IXI.xls Table",
        },
    }
    return s, columns


def main() -> None:
    img = images()
    smp, columns = samples(img)
    img = img.drop(columns="site")
    assert len(smp) == 584

    participants = smp.set_index("participant_id")
    core = img[img.modality.isin(["T1w", "T2w", "PD"])].groupby("participant_id").modality.nunique()
    complete = (core.reindex(participants.index) == 3) & participants.age.notna()
    age_bin = pd.cut(participants.age, AGE_BINS, right=False).astype(str)
    strata = participants.site + "_" + age_bin
    splits = tables.make_splits(strata, complete)
    tables.write(NAME, img, smp, columns, splits)


if __name__ == "__main__":
    main()

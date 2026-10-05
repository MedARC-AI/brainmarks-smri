"""Build the benchmark tables for BraTS 2021 from datasets/brats2021/source/.

    uv run python scripts/brats2021/build_tables.py

See `brainmarks_smri.tables` for the table layout.

- Samples: the 1479 cases of the BraTS<->TCIA crosswalk (metadata/BraTS2021_MappingToTCIA.xlsx).
  session_id is the BraTS 2021 case ID (the 8 Task-2-only cases written as bare numbers, '00169',
  are normalized to 'BraTS2021_00169'). participant_id is the case ID of the patient's first
  case: two UCSF-PDGM follow-up cases belong to patients who also have a baseline case
  (SAME_PATIENT), so 1479 cases are 1477 participants. Other cases are assumed to be distinct
  patients: the real TCIA PatientIDs are unique (checked), but the anonymized '_Additional' and
  'Collection N' cases can't be checked.
- Images: Task 1 NIfTI (skull-stripped, co-registered, SRI-24 1 mm): T1w, T1c (t1ce), T2w,
  FLAIR, and the tumor mask (labels 1/2/4) for the 1251 Task 1 training cases. The 219
  validation cases have no mask. The Task-2-only cases exist only as DICOM (not fetched).
- Overlap: `collection` and `tcia_patient_id` link cases to UPENN-GBM / UCSF-PDGM / TCGA. Evals
  are within-dataset, so this is only for reference.
- Our split: the official Task 1 validation cases have no masks, so all 1479 cases are split
  60/20/20, stratified by has mask x MGMT label. `official_split` is the Task 1 cohort
  (train/val; n/a for Task-2-only); `mgmt_cohort` holds the Task 2 cohort.
- No age or sex in the source (n/a). site = the BraTS institution code.
- Complete: the 4 images and the tumor mask.
"""

import re

import pandas as pd

from brainmarks_smri import tables

NAME = "brats2021"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
PACKAGE = SOURCE / "RSNA-ASNR-MICCAI-BraTS-2021"
FILES = {
    "t1": ("T1w", None),
    "t1ce": ("T1c", None),
    "t2": ("T2w", None),
    "flair": ("FLAIR", None),
    "seg": ("mask", "tumor"),
}
COHORT = {"Training": "train", "Validation": "val"}
CASE_ID = re.compile(r"(?:BraTS2021_)?(\d{5})")
# Follow-up case -> baseline case of the same patient. The crosswalk can't show this: it lists the
# follow-ups under their pre-v3 UCSF-PDGM PatientIDs (138, 315). UCSF-PDGM v5 metadata
# (UCSF-PDGM-metadata_v5.csv, `BraTS21 ID`) renamed them UCSF-PDGM-0429_FU003d and
# UCSF-PDGM-0433_FU007d, whose baselines are UCSF-PDGM-0429 (BraTS2021_00639) and
# UCSF-PDGM-0433 (BraTS2021_00626). The other UCSF follow-ups in BraTS (00499, 00538, 00539) have
# no baseline case in BraTS.
SAME_PATIENT = {"BraTS2021_00557": "BraTS2021_00639", "BraTS2021_00758": "BraTS2021_00626"}


def case_id(raw: str) -> str:
    match = CASE_ID.fullmatch(str(raw))
    assert match, f"unexpected BraTS ID {raw!r}"
    return f"BraTS2021_{match[1]}"


def parse_dates(values: pd.Series) -> pd.Series:
    """Excel dates (read as 'YYYY-MM-DD 00:00:00') and m/d/yyyy text -> YYYY-MM-DD."""
    iso = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    us = pd.to_datetime(values, format="%m/%d/%Y", errors="coerce")
    parsed = iso.fillna(us)
    assert parsed.notna().sum() == values.notna().sum(), "unparsed study dates"
    return parsed.dt.strftime("%Y-%m-%d")


def images() -> pd.DataFrame:
    rows = []
    for split_folder in ("BraTS2021_TrainingSet", "BraTS2021_ValidationSet"):
        for path in sorted((PACKAGE / split_folder).glob("*/BraTS2021_*/*.nii.gz")):
            case = path.parent.name
            modality, desc = FILES[path.name.removeprefix(case + "_").removesuffix(".nii.gz")]
            rows.append(
                (SAME_PATIENT.get(case, case), case, modality, desc, str(path.relative_to(ROOT)))
            )
    return pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS)


def samples() -> tuple[pd.DataFrame, dict[str, dict], pd.Series]:
    meta = pd.read_excel(SOURCE / "metadata" / "BraTS2021_MappingToTCIA.xlsx", dtype=str)
    meta.columns = [
        "collection",
        "site",
        "tcia_patient_id",
        "study_date",
        "brats_id",
        "seg_cohort",
        "mgmt_cohort",
        "mgmt",
    ]
    real_ids = meta.dropna(subset=["tcia_patient_id"])
    real_ids = real_ids[real_ids.tcia_patient_id != "new-not-previously-in-TCIA"]
    assert not real_ids.duplicated(["collection", "tcia_patient_id"]).any(), (
        "a TCIA patient has several cases"
    )

    cases = meta.brats_id.map(case_id)
    s = pd.DataFrame(
        {"participant_id": cases.map(lambda c: SAME_PATIENT.get(c, c)), "session_id": cases}
    )
    s["age"] = pd.NA
    s["sex"] = pd.NA
    s["site"] = meta.site
    s["collection"] = meta.collection
    s["tcia_patient_id"] = meta.tcia_patient_id.replace("new-not-previously-in-TCIA", pd.NA)
    s["study_date"] = parse_dates(meta.study_date)
    s["mgmt"] = pd.to_numeric(meta.mgmt).map({1: "methylated", 0: "unmethylated"})
    s["mgmt_cohort"] = meta.mgmt_cohort.map(COHORT)
    official = meta.seg_cohort.map(COHORT).set_axis(cases)

    columns = {
        "collection": {
            "Description": "Source collection (TCIA collection, '<collection>_Additional', or anonymized 'Collection N').",
            "Source": "Data Collection (as on TCIA+additional)",
        },
        "tcia_patient_id": {
            "Description": "PatientID in the source TCIA collection (for overlap with upenn_gbm / ucsf_pdgm). Formats differ by "
            "collection (UCSF-PDGM: bare number; UPENN-GBM: scan ID). UCSF-PDGM IDs 138, 175, 181, 278, 315 are "
            "pre-v3 IDs of follow-up exams, renamed in UCSF-PDGM v3.",
            "Source": "PatientID on TCIA Radiology Portal",
        },
        "age": {"Description": "Not in the source (all n/a).", "Units": "years"},
        "sex": {"Description": "Not in the source (all n/a)."},
        "study_date": {
            "Description": "Study date (some cases), as de-identified by TCIA (dates are shifted, so they are not real "
            "calendar dates). The crosswalk mixes Excel dates and m/d/yyyy text; written as YYYY-MM-DD.",
            "Source": "Study date (m/d/yyyy) per PatientID",
        },
        "mgmt": {
            "Description": "MGMT promoter methylation (Task 2 target; also given for some non-Task-2 cases).",
            "Source": "MGMT value",
            "Levels": {"methylated": "1", "unmethylated": "0"},
        },
        "mgmt_cohort": {
            "Description": "Official Task 2 (MGMT) cohort.",
            "Source": "MGMT (Task 2) Cohort",
            "Levels": {"train": "Training", "val": "Validation"},
        },
    }
    return s, columns, official


def main() -> None:
    img = images()
    smp, columns, official = samples()
    assert len(smp) == 1479 and smp.session_id.is_unique and smp.participant_id.nunique() == 1477
    masked_cases = img[img.modality == "mask"].session_id
    assert len(masked_cases) == 1251 and set(masked_cases) == set(
        official.index[official == "train"]
    )

    # per participant: complete if every case has the 4 images + mask; strata/official from the first case
    n_images = img.groupby("session_id").size().reindex(smp.session_id, fill_value=0)
    smp_flags = smp.assign(
        complete=(n_images == len(FILES)).values,
        has_mask=smp.session_id.isin(masked_cases).values,
        official=smp.session_id.map(official).values,
    )
    participants = smp_flags.sort_values("session_id").groupby("participant_id")
    first = participants.first()
    complete = participants.complete.all()
    strata = first.has_mask.map({True: "mask", False: "nomask"}) + "_" + first.mgmt.fillna("n/a")
    splits = tables.make_splits(strata, complete, official=first.official)
    tables.write(NAME, img, smp, columns, splits, summary=["mgmt"])


if __name__ == "__main__":
    main()

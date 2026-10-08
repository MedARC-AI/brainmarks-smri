"""Build the benchmark tables for UCSF-PDGM from datasets/ucsf_pdgm/source/.

    uv run python scripts/ucsf_pdgm/build_tables.py

See `brainmarks_smri.utils.tables` for the table layout.

- Sessions: 501 exams of 495 patients. The 6 follow-up exams (e.g. `UCSF-PDGM-0433_FU007d`)
  become session `FU007d` of participant `UCSF-PDGM-0433`; every other exam is session `baseline`.
- IDs: the metadata pads IDs to 3 digits (`UCSF-PDGM-004`), the image folders to 4 (`-0004`);
  participant_id uses 4 digits.
- No official split: 60/20/20 by patient, stratified by IDH status x WHO grade of the baseline exam.
- Complete: all 13 images in every session, IDH status and overall survival.
"""

import re

import pandas as pd

from brainmarks_smri.utils import tables

NAME = "ucsf_pdgm"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
# file name suffix -> (modality, desc)
FILES = {
    "T1": ("T1w", None),
    "T1_bias": ("T1w", "bias"),
    "T1c": ("T1c", None),
    "T1c_bias": ("T1c", "bias"),
    "T2": ("T2w", None),
    "T2_bias": ("T2w", "bias"),
    "FLAIR": ("FLAIR", None),
    "FLAIR_bias": ("FLAIR", "bias"),
    "DWI": ("DWI", None),
    "ADC": ("ADC", None),
    "tumor_segmentation": ("mask", "tumor"),
    "brain_segmentation": ("mask", "brain"),
    "brain_parenchyma_segmentation": ("mask", "parenchyma"),
}
EXAM_ID = re.compile(r"UCSF-PDGM-(\d+)(?:_(FU\d+d))?$")


def parse_exam(exam: str) -> tuple[str, str]:
    """'UCSF-PDGM-433_FU007d' or 'UCSF-PDGM-0433_FU007d' -> ('UCSF-PDGM-0433', 'FU007d')."""
    match = EXAM_ID.match(exam)
    assert match, f"unexpected exam ID {exam!r}"
    number, follow_up = match.groups()
    return f"UCSF-PDGM-{int(number):04d}", follow_up or "baseline"


def images() -> pd.DataFrame:
    rows = []
    for folder in sorted((SOURCE / "UCSF-PDGM-v5").glob("*_nifti")):
        exam = folder.name.removesuffix("_nifti")
        participant, session = parse_exam(exam)
        for path in sorted(folder.glob("*.nii.gz")):
            suffix = path.name.removeprefix(exam + "_").removesuffix(".nii.gz")
            modality, desc = FILES[suffix]
            rows.append((participant, session, modality, desc, str(path.relative_to(ROOT))))
    return pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS)


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = pd.read_csv(SOURCE / "UCSF-PDGM-metadata_v5.csv", dtype=str)
    meta = meta.apply(lambda column: column.str.strip())
    unknown = {"unknown": pd.NA, "indeterminate": pd.NA}

    s = pd.DataFrame(meta.ID.map(parse_exam).tolist(), columns=["participant_id", "session_id"])
    s["age"] = pd.to_numeric(meta["Age at MRI"])
    s["sex"] = meta.Sex
    s["site"] = "UCSF"
    s["who_grade"] = pd.to_numeric(meta["WHO CNS Grade"]).astype("Int64")
    s["diagnosis"] = meta["Final pathologic diagnosis (WHO 2021)"]
    s["idh"] = meta.IDH.map(lambda v: "wildtype" if v == "wildtype" else "mutant")
    s["idh_variant"] = meta.IDH
    s["mgmt"] = meta["MGMT status"].replace(unknown)
    s["mgmt_index"] = pd.to_numeric(meta["MGMT index"], errors="coerce").astype("Int64")
    s["codeletion_1p19q"] = meta["1p/19q"].replace(unknown)
    s["os_days"] = pd.to_numeric(meta.OS)
    s["os_event"] = pd.to_numeric(meta["1-dead 0-alive"]).astype("Int64")
    s["resection"] = meta.EOR
    s["biopsy_prior"] = meta["Biopsy prior to imaging"].str.lower()
    s["brats21_id"] = meta["BraTS21 ID"]
    s["brats21_seg_cohort"] = meta["BraTS21 Segmentation Cohort"]
    s["brats21_mgmt_cohort"] = meta["BraTS21 MGMT Cohort"]

    columns = {
        "who_grade": {
            "Description": "WHO CNS 2021 grade.",
            "Source": "WHO CNS Grade",
            "Levels": {"2": "grade 2", "3": "grade 3", "4": "grade 4"},
        },
        "diagnosis": {
            "Description": "Final integrated pathologic diagnosis (WHO CNS 2021).",
            "Source": "Final pathologic diagnosis (WHO 2021)",
        },
        "idh": {
            "Description": "IDH mutation status (target).",
            "Source": "IDH",
            "Levels": {
                "wildtype": "IDH wildtype",
                "mutant": "any IDH1/IDH2 mutation, incl. 'mutated (NOS)'",
            },
        },
        "idh_variant": {"Description": "IDH mutation subtype as reported.", "Source": "IDH"},
        "mgmt": {
            "Description": "MGMT promoter methylation, clinical interpretation (target). "
            "'unknown' and 'indeterminate' are n/a.",
            "Source": "MGMT status",
            "Levels": {"positive": "methylated", "negative": "unmethylated"},
        },
        "mgmt_index": {
            "Description": "MGMT methylation index: number of methylated promoter sites (0-17).",
            "Source": "MGMT index",
        },
        "codeletion_1p19q": {
            "Description": "1p/19q codeletion by FISH. 'unknown' is n/a.",
            "Source": "1p/19q",
            "Levels": {
                "intact": "no codeletion",
                "co-deletion": "codeleted",
                "relative co-deletion": "relative codeletion",
            },
        },
        "os_days": {
            "Description": "Overall survival from initial diagnosis to last follow-up (target, with os_event).",
            "Source": "OS",
            "Units": "days",
        },
        "os_event": {
            "Description": "Vital status at last follow-up.",
            "Source": "1-dead 0-alive",
            "Levels": {"1": "dead (event)", "0": "alive (censored)"},
        },
        "resection": {
            "Description": "Extent of resection.",
            "Source": "EOR",
            "Levels": {"GTR": "gross total", "STR": "subtotal", "biopsy": "biopsy only"},
        },
        "biopsy_prior": {
            "Description": "Burr-hole biopsy before the MRI.",
            "Source": "Biopsy prior to imaging",
            "Levels": {"yes": "biopsy before MRI", "no": "no biopsy before MRI"},
        },
        "brats21_id": {
            "Description": "Case ID in BraTS 2021, if included (overlap with the brats2021 dataset).",
            "Source": "BraTS21 ID",
        },
        "brats21_seg_cohort": {
            "Description": "BraTS 2021 Task 1 cohort of this case.",
            "Source": "BraTS21 Segmentation Cohort",
        },
        "brats21_mgmt_cohort": {
            "Description": "BraTS 2021 Task 2 cohort of this case.",
            "Source": "BraTS21 MGMT Cohort",
        },
    }
    return s, columns


def main() -> None:
    img = images()
    smp, columns = samples()
    assert len(smp) == 501 and smp.participant_id.nunique() == 495
    keys = ["participant_id", "session_id"]
    assert set(img[keys].itertuples(index=False)) == set(smp[keys].itertuples(index=False))

    n_images = img.groupby(keys).size().rename("n_images")
    smp_complete = smp.join(n_images, on=keys)
    smp_complete["complete"] = (
        (smp_complete.n_images == len(FILES))
        & smp_complete.idh.notna()
        & smp_complete.os_days.notna()
        & smp_complete.os_event.notna()
    )
    complete = smp_complete.groupby("participant_id").complete.all()

    baseline = smp[smp.session_id == "baseline"].set_index("participant_id")
    strata = baseline.idh + "_" + baseline.who_grade.astype(str)

    splits = tables.make_splits(strata, complete)
    tables.write(NAME, img, smp, columns, splits, summary=["idh", "who_grade", "mgmt", "os_event"])


if __name__ == "__main__":
    main()

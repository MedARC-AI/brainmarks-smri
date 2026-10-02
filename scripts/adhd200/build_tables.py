"""Build the benchmark tables for ADHD-200 from datasets/adhd200/source/.

    uv run python scripts/adhd200/build_tables.py

See `brainmarks_smri.tables` for the table layout.

- Participants: the per-site RawDataBIDS/<site>/participants.tsv files (read as latin-1), one
  session each: the BIDS session of the T1w ('1', except 9 WashU participants in ses-2/3/4).
  participant_id is the BIDS label: the participants.tsv IDs are unpadded (`26001`), the folders
  7-digit (`sub-0026001`).
- Cleaning: NYU, Peking_1 and Pittsburgh list their test participants twice; after normalizing
  handedness ('L' -> 'Left') the duplicates are identical and dropped. -999 is missing.
  Handedness is categorical at most sites but a score (-1..1) at NYU: `handedness` and
  `handedness_score`. 'Medication Na´ve' is a mis-encoded 'Medication Naive'. KKI's iq_measure and
  WashU's QC are numeric codes, decoded with nitrc/general/ADHD-200_PhenotypicKey.pdf.
- Site-specific phenotypes that are only in the per-site *_phenotypic.csv files (e.g. KKI/OHSU
  ADHD scores) are not merged; they stay in source/.
- Official split: the competition holdout (nitrc/general/allSubs_testSet_phenotypic_dx.csv,
  197 participants; 186 have images) is `test`. The training release is split 75/25 into train
  and val, stratified by diagnosis x site. `official_split` is train/test. The participants.tsv
  diagnoses of the test participants match the released test labels (checked).
- Brown (26, all test) never had labels released: diagnosis n/a, not complete.
- Complete: T1w and diagnosis.
"""

import pandas as pd

from brainmarks_smri import tables

NAME = "adhd200"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
BIDS = SOURCE / "RawDataBIDS"
DX_CODES = {"0": "TDC", "1": "ADHD-Combined", "2": "ADHD-Hyperactive/Impulsive", "3": "ADHD-Inattentive"}
DX_NAMES = {"Typically Developing Children": "TDC"}
HANDEDNESS = {"L": "Left", "R": "Right"}
# codes used by some sites instead of labels (ADHD-200_PhenotypicKey.pdf)
IQ_MEASURE_CODES = {"1": "Wechsler Intelligence Scale for Children, Fourth Edition (WISC-IV)"}
QC_CODES = {"1": "Pass", "0": "Questionable"}


def participant_id(raw: str | int) -> str:
    return f"sub-{int(raw):07d}"


def read_participants() -> pd.DataFrame:
    frames = []
    for path in sorted(BIDS.glob("*/participants.tsv")):
        site = pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1")
        site.columns = site.columns.str.strip()
        site["site"] = path.parent.name
        frames.append(site)
    meta = pd.concat(frames, ignore_index=True).replace("-999", pd.NA)
    meta["participant_id"] = meta.participant_id.map(participant_id)
    meta["handedness"] = meta.handedness.replace(HANDEDNESS)
    meta = meta.drop_duplicates()
    assert meta.participant_id.is_unique, "conflicting duplicate rows"
    return meta.reset_index(drop=True)


def samples(meta: pd.DataFrame, session: pd.Series) -> tuple[pd.DataFrame, dict[str, dict]]:
    """`session` maps participant_id to the BIDS session of its T1w (WashU uses ses-2..4)."""
    handedness_score = pd.to_numeric(meta.handedness, errors="coerce")

    s = pd.DataFrame({"participant_id": meta.participant_id, "session_id": meta.participant_id.map(session).fillna("1")})
    s["age"] = pd.to_numeric(meta.age)
    s["sex"] = meta.gender.map({"Male": "M", "Female": "F"})
    s["site"] = meta.site
    s["diagnosis"] = meta.dx.replace(DX_NAMES)
    s["adhd"] = s.diagnosis.map(lambda d: pd.NA if pd.isna(d) else d != "TDC")
    s["secondary_diagnosis"] = meta.secondary_dx.fillna(meta.secondary_dx_).str.strip()
    s["adhd_measure"] = meta.adhd_measure
    s["adhd_index"] = pd.to_numeric(meta.adhd_index)
    s["inattentive"] = pd.to_numeric(meta.inattentive)
    s["hyper_impulsive"] = pd.to_numeric(meta.hyper_impulsive)
    s["med_status"] = meta.med_status.str.replace("Na´ve", "Naive")
    s["handedness"] = meta.handedness.where(handedness_score.isna())
    s["handedness_score"] = handedness_score
    s["iq_measure"] = meta.iq_measure.replace(IQ_MEASURE_CODES)
    s["verbal_iq"] = pd.to_numeric(meta.verbal_iq)
    s["performance_iq"] = pd.to_numeric(meta.performance_iq)
    s["full4_iq"] = pd.to_numeric(meta.full4_iq)
    s["full2_iq"] = pd.to_numeric(meta.full2_iq)
    s["qc_anatomical"] = meta.qc_anatomical_1.fillna(meta.qc_s1_anat).fillna(meta.qc_s2_anat).replace(QC_CODES)

    columns = {
        "diagnosis": {
            "Description": "Diagnosis (target). n/a for Brown (labels never released).",
            "Source": "dx",
            "Levels": {"TDC": "typically developing", "ADHD-Combined": "", "ADHD-Hyperactive/Impulsive": "", "ADHD-Inattentive": ""},
        },
        "adhd": {"Description": "ADHD of any subtype vs TDC (binary target).", "Source": "dx"},
        "secondary_diagnosis": {"Description": "Secondary diagnoses (free text; some sites).", "Source": "secondary_dx"},
        "adhd_measure": {"Description": "ADHD rating instrument (some sites).", "Source": "adhd_measure"},
        "adhd_index": {"Description": "ADHD index score (instrument varies by site).", "Source": "adhd_index"},
        "inattentive": {"Description": "Inattentive subscale score.", "Source": "inattentive"},
        "hyper_impulsive": {"Description": "Hyperactive/impulsive subscale score.", "Source": "hyper_impulsive"},
        "med_status": {"Description": "Medication status (some sites).", "Source": "med_status", "Levels": {"Medication Naive": "", "Not Medication Naive": ""}},
        "handedness": {"Description": "Handedness category (all sites except NYU).", "Source": "handedness", "Levels": {"Left": "", "Right": "", "Ambidextrous": ""}},
        "handedness_score": {"Description": "Edinburgh handedness score, -1 (left) to 1 (right) (NYU only).", "Source": "handedness"},
        "iq_measure": {"Description": "IQ instrument.", "Source": "iq_measure"},
        "verbal_iq": {"Description": "Verbal IQ.", "Source": "verbal_iq"},
        "performance_iq": {"Description": "Performance IQ.", "Source": "performance_iq"},
        "full4_iq": {"Description": "Full-scale IQ, four subtests.", "Source": "full4_iq"},
        "full2_iq": {"Description": "Full-scale IQ, two subtests.", "Source": "full2_iq"},
        "qc_anatomical": {"Description": "Quality control of the (first) anatomical scan. WashU codes (1/0) decoded with the phenotypic key.", "Source": "qc_anatomical_1 (WashU: qc_s1_anat, else qc_s2_anat)", "Levels": {"Pass": "", "Questionable": ""}},
    }
    return s, columns


def official_test() -> pd.Series:
    """Released test-set labels, by participant_id."""
    test = pd.read_csv(SOURCE / "nitrc/general/allSubs_testSet_phenotypic_dx.csv", dtype=str)
    return test.set_index(test.ID.map(participant_id)).DX.map(DX_CODES)


def main() -> None:
    per_site = [tables.bids_images(folder, ROOT) for folder in sorted(BIDS.iterdir()) if folder.is_dir()]
    images = pd.concat(per_site, ignore_index=True)
    meta = read_participants()
    smp, columns = samples(meta, images.set_index("participant_id").session_id)
    assert images.participant_id.is_unique and (images.modality == "T1w").all() and len(images) == 961
    assert set(images.participant_id) <= set(smp.participant_id)

    participants = smp.set_index("participant_id")
    test_labels = official_test()
    in_test = participants.index.isin(test_labels.index)
    labeled_test = test_labels.dropna()[test_labels.dropna().index.isin(participants.index)]
    assert (participants.diagnosis[labeled_test.index] == labeled_test).all(), "test labels disagree"
    official = pd.Series(["test" if t else "train" for t in in_test], index=participants.index)

    strata = participants.diagnosis + "_" + participants.site
    train_val = tables.stratified_split(strata[~in_test], {"train": 0.75, "val": 0.25})
    split = pd.concat([train_val, official[in_test]]).reindex(participants.index)
    complete = pd.Series(participants.index.isin(images.participant_id) & participants.diagnosis.notna().values,
                         index=participants.index)
    splits = tables.make_splits(strata, complete, official=official, split=split)
    tables.write(NAME, images, smp, columns, splits, summary=["adhd"])


if __name__ == "__main__":
    main()

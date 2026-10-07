"""Build the benchmark tables for OpenBHB from datasets/openbhb/source/.

    uv run python scripts/openbhb/build_tables.py

See `brainmarks_smri.datasets.tables` for the table layout.

- 3984 healthy participants (public part), one session each (the BIDS ses-1), one original
  T1w each in {train,val}/rawdata/. participant_id gets the BIDS `sub-` prefix.
- `site` and `study` are the dataset's integer codes (62 sites, 10 cohort studies).
- QC metrics from qc.tsv are merged in.
- Official split (participants.tsv `split`): train / internal_test / external_test (the private
  test set is not public). The external test is defined by site x acquisition setting: 5 of its
  6 sites are absent from train, but site 36 is also in train with another acquisition setting. Both test
  sets are our `test` (`official_split` tells them apart). The official train set is split
  80/20 into train and val, stratified by site, as in Neuro-JEPA.
- Complete: T1w and age.
"""

import pandas as pd

from brainmarks_smri.datasets import tables

NAME = "openbhb"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"


def samples() -> tuple[pd.DataFrame, dict[str, dict], pd.Series]:
    meta = pd.read_csv(SOURCE / "participants.tsv", sep="\t", dtype={"participant_id": str})
    qc = pd.read_csv(SOURCE / "qc.tsv", sep="\t", dtype={"participant_id": str})
    assert meta.participant_id.is_unique and qc.participant_id.is_unique
    meta = meta.merge(qc, on="participant_id", how="left", validate="one_to_one")

    s = pd.DataFrame({"participant_id": "sub-" + meta.participant_id, "session_id": "1"})
    s["age"] = meta.age
    s["sex"] = meta.sex.map({"male": "M", "female": "F"})
    s["site"] = meta.site.astype("Int64").astype(str)
    s["study"] = meta.study.astype("Int64").astype(str)
    s["diagnosis"] = meta.diagnosis
    s["magnetic_field_strength"] = meta.magnetic_field_strength
    s["acquisition_setting"] = meta.acquisition_setting.astype("Int64")
    s["site_x_acquisition"] = meta.siteXacq.astype("Int64")
    s["tiv"] = meta.tiv
    s["csfv"] = meta.csfv
    s["gmv"] = meta.gmv
    s["wmv"] = meta.wmv
    s["qc_freesurfer_euler"] = meta["reconall-euler"]
    s["qc_cat12_ncr"] = meta["cat12vbm-ncr"]
    s["qc_cat12_iqr"] = meta["cat12vbm-iqr"]
    s["qc_quasiraw_corr"] = meta["quasiraw-corr"]
    official = meta.split.set_axis(s.participant_id)

    columns = {
        "study": {"Description": "Source cohort study (integer code).", "Source": "study"},
        "diagnosis": {"Description": "Diagnosis (all healthy controls).", "Source": "diagnosis"},
        "magnetic_field_strength": {
            "Description": "Scanner field strength.",
            "Source": "magnetic_field_strength",
            "Units": "T",
        },
        "acquisition_setting": {
            "Description": "Acquisition setting code within a site.",
            "Source": "acquisition_setting",
        },
        "site_x_acquisition": {
            "Description": "Site x acquisition setting code (64 values).",
            "Source": "siteXacq",
        },
        "tiv": {
            "Description": "Total intracranial volume (CAT12).",
            "Source": "tiv",
            "Units": "cm^3",
        },
        "csfv": {"Description": "CSF volume (CAT12).", "Source": "csfv", "Units": "cm^3"},
        "gmv": {"Description": "Grey matter volume (CAT12).", "Source": "gmv", "Units": "cm^3"},
        "wmv": {"Description": "White matter volume (CAT12).", "Source": "wmv", "Units": "cm^3"},
        "qc_freesurfer_euler": {
            "Description": "FreeSurfer recon-all Euler number (QC).",
            "Source": "qc.tsv reconall-euler",
        },
        "qc_cat12_ncr": {
            "Description": "CAT12 noise-to-contrast ratio (QC).",
            "Source": "qc.tsv cat12vbm-ncr",
        },
        "qc_cat12_iqr": {
            "Description": "CAT12 image quality rating (QC).",
            "Source": "qc.tsv cat12vbm-iqr",
        },
        "qc_quasiraw_corr": {
            "Description": "Correlation of the quasi-raw image with the template (QC).",
            "Source": "qc.tsv quasiraw-corr",
        },
    }
    return s, columns, official


def main() -> None:
    images = pd.concat(
        [tables.bids_images(SOURCE / part / "rawdata", ROOT) for part in ("train", "val")],
        ignore_index=True,
    )
    smp, columns, official = samples()
    assert len(smp) == 3984 and images.participant_id.is_unique and len(images) == 3984
    assert (images.modality == "T1w").all() and set(images.participant_id) == set(
        smp.participant_id
    )
    assert set(official) == {"train", "internal_test", "external_test"}

    participants = smp.set_index("participant_id")
    in_train = official == "train"
    train_val = tables.stratified_split(participants.site[in_train], {"train": 0.8, "val": 0.2})
    split = pd.concat([train_val, official[~in_train].map(lambda _: "test")]).reindex(
        participants.index
    )
    complete = participants.age.notna()
    splits = tables.make_splits(participants.site, complete, official=official, split=split)
    tables.write(NAME, images, smp, columns, splits, summary=[])


if __name__ == "__main__":
    main()

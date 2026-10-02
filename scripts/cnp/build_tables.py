"""Build the benchmark tables for CNP / LA5c (OpenNeuro ds000030) from datasets/cnp/source/.

    uv run python scripts/cnp/build_tables.py

See `brain_datasets.tables` for the table layout.

- 272 participants in participants.tsv, one session each (session_id '1'). 265 have a T1w and
  262 a raw multi-direction DWI (modality DTI: 4D, with .bval/.bvec next to the NIfTI). The 7
  without a T1w (fMRI only) are kept in the tables but not complete.
- Single site (UCLA), two 3T Siemens Trio scanners (`scanner_serial`).
- The per-task availability flags of participants.tsv (fMRI tasks) are left out. The phenotype/
  questionnaires (~30 tables) stay in source/ and are not merged here.
- No official split: 60/20/20, stratified by diagnosis.
- Complete: T1w and diagnosis.
"""

import pandas as pd

from brain_datasets import tables

NAME = "cnp"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = pd.read_csv(SOURCE / "participants.tsv", sep="\t", na_values="n/a")

    s = pd.DataFrame({"participant_id": meta.participant_id, "session_id": "1"})
    s["age"] = meta.age.astype(float)
    s["sex"] = meta.gender
    s["site"] = "UCLA"
    s["diagnosis"] = meta.diagnosis
    s["scanner_serial"] = meta.ScannerSerialNumber.astype("Int64")
    s["ghost_artifact"] = meta.ghost_NoGhost.map({"ghost": True, "No_ghost": False})

    columns = {
        "diagnosis": {
            "Description": "Diagnostic group (target).",
            "Source": "diagnosis",
            "Levels": {"CONTROL": "healthy control", "SCHZ": "schizophrenia", "BIPOLAR": "bipolar disorder", "ADHD": "adult ADHD"},
        },
        "scanner_serial": {"Description": "Serial number of the 3T Siemens Trio scanner used.", "Source": "ScannerSerialNumber"},
        "ghost_artifact": {
            "Description": "Ghosting artifact through the temporal lobes on the T1w (QC covariate).",
            "Source": "ghost_NoGhost",
        },
    }
    return s, columns


def main() -> None:
    img = tables.bids_images(SOURCE, ROOT)
    img["modality"] = img.modality.replace({"dwi": "DTI"})  # raw multi-direction DWI series
    smp, columns = samples()
    assert len(smp) == 272 and smp.participant_id.is_unique
    assert set(img.modality) == {"T1w", "DTI"}

    participants = smp.set_index("participant_id")
    t1w = img[img.modality == "T1w"].participant_id
    assert t1w.is_unique and len(t1w) == 265
    complete = pd.Series(participants.index.isin(t1w) & participants.diagnosis.notna().values,
                         index=participants.index)
    splits = tables.make_splits(participants.diagnosis, complete)
    tables.write(NAME, img, smp, columns, splits)


if __name__ == "__main__":
    main()

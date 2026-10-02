"""Build the benchmark tables for ABIDE I from datasets/abide1/source/.

    uv run python scripts/abide1/build_tables.py

See `brain_datasets.tables` for the table layout.

- 1112 participants (Phenotypic_V1_0b.csv), one session each (session_id '1'); 1102 have a T1w
  (10 UCLA participants have only rs-fMRI and are not complete).
- Images: the per-site BIDS folders RawDataBIDS/<site folder>/sub-*/anat/. participant_id is the
  BIDS label (`sub-0050002`, the 7-digit zero-padded SUB_ID).
- site = SITE_ID (20 sites); `site_folder` = the BIDS folder (24, e.g. CMU_a/CMU_b).
- Phenotypes: a selection of the composite file's 75 columns (diagnosis, DSM-IV subtype, IQ,
  ADOS/SRS totals, handedness, medication). -9999 and the stray '`' are missing values. The full
  file and its legend (ABIDE_LEGEND_V1.02.pdf) stay in source/.
- No official split: 60/20/20, stratified by diagnosis x site.
- Complete: T1w and diagnosis.
"""

import pandas as pd

from brain_datasets import tables

NAME = "abide1"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
BIDS = SOURCE / "RawDataBIDS"


def images() -> pd.DataFrame:
    """Images of all BIDS site folders (`sidecards/` only holds JSON sidecars)."""
    per_site = [tables.bids_images(folder, ROOT) for folder in sorted(BIDS.iterdir())
                if folder.is_dir() and folder.name != "sidecards"]
    return pd.concat(per_site, ignore_index=True)


def samples(img: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = pd.read_csv(SOURCE / "Phenotypic_V1_0b.csv", dtype={"CURRENT_MED_STATUS": str})
    meta = meta.replace({-9999: pd.NA, "-9999": pd.NA, "`": pd.NA})

    s = pd.DataFrame({"participant_id": meta.SUB_ID.map(lambda i: f"sub-{i:07d}"), "session_id": "1"})
    s["age"] = pd.to_numeric(meta.AGE_AT_SCAN)
    s["sex"] = meta.SEX.map({1: "M", 2: "F"})
    s["site"] = meta.SITE_ID
    folder_of = img.assign(folder=img.path.str.split("/").str[2]).groupby("participant_id").folder.first()
    s["site_folder"] = s.participant_id.map(folder_of)
    s["diagnosis"] = meta.DX_GROUP.map({1: "ASD", 2: "TDC"})
    s["dsm_iv_tr"] = meta.DSM_IV_TR.map({0: "control", 1: "autism", 2: "asperger", 3: "PDD-NOS", 4: "asperger or PDD-NOS"})
    s["handedness"] = meta.HANDEDNESS_CATEGORY
    s["fiq"] = pd.to_numeric(meta.FIQ)
    s["viq"] = pd.to_numeric(meta.VIQ)
    s["piq"] = pd.to_numeric(meta.PIQ)
    s["ados_total"] = pd.to_numeric(meta.ADOS_TOTAL)
    s["ados_gotham_severity"] = pd.to_numeric(meta.ADOS_GOTHAM_SEVERITY)
    s["srs_raw_total"] = pd.to_numeric(meta.SRS_RAW_TOTAL)
    s["current_med_status"] = meta.CURRENT_MED_STATUS.map({"0": False, "1": True})

    columns = {
        "site_folder": {"Description": "BIDS site folder under RawDataBIDS/ (some SITE_IDs are split into sub-sites, e.g. CMU_a/CMU_b)."},
        "diagnosis": {"Description": "Diagnostic group (target).", "Source": "DX_GROUP", "Levels": {"ASD": "autism spectrum disorder", "TDC": "typically developing control"}},
        "dsm_iv_tr": {
            "Description": "DSM-IV-TR diagnostic subtype. As in the source, 20 ASD participants are coded 'control' here; "
                           "use `diagnosis` as the label.",
            "Source": "DSM_IV_TR",
            "Levels": {"control": "0", "autism": "1", "asperger": "2", "PDD-NOS": "3", "asperger or PDD-NOS": "4"},
        },
        "handedness": {
            "Description": "Handedness category, as coded by each site.",
            "Source": "HANDEDNESS_CATEGORY",
            "Levels": {"R": "right", "L": "left", "Ambi": "ambidextrous", "Mixed": "mixed", "L->R": "left, converted to right"},
        },
        "fiq": {"Description": "Full-scale IQ (test type varies by site: FIQ_TEST_TYPE).", "Source": "FIQ"},
        "viq": {"Description": "Verbal IQ.", "Source": "VIQ"},
        "piq": {"Description": "Performance IQ.", "Source": "PIQ"},
        "ados_total": {"Description": "ADOS total score (communication + social).", "Source": "ADOS_TOTAL"},
        "ados_gotham_severity": {"Description": "ADOS calibrated severity score (Gotham).", "Source": "ADOS_GOTHAM_SEVERITY"},
        "srs_raw_total": {"Description": "Social Responsiveness Scale raw total.", "Source": "SRS_RAW_TOTAL"},
        "current_med_status": {"Description": "Taking medication at the time of scan.", "Source": "CURRENT_MED_STATUS"},
    }
    return s, columns


def main() -> None:
    img = images()
    smp, columns = samples(img)
    assert len(smp) == 1112 and smp.participant_id.is_unique
    assert (img.modality == "T1w").all() and img.participant_id.is_unique and len(img) == 1102
    assert set(img.participant_id) <= set(smp.participant_id)

    participants = smp.set_index("participant_id")
    complete = pd.Series(participants.index.isin(img.participant_id) & participants.diagnosis.notna().values,
                         index=participants.index)
    strata = participants.diagnosis + "_" + participants.site
    splits = tables.make_splits(strata, complete)
    tables.write(NAME, img, smp, columns, splits, summary=["diagnosis"])


if __name__ == "__main__":
    main()

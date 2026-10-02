"""Build the benchmark tables for SOOP (OpenNeuro ds004889) from datasets/soop/source/.

    uv run python scripts/soop/build_tables.py

See `brain_datasets.tables` for the table layout.

- 1715 imaged participants, one session each (session_id '1'): T1w, FLAIR, DWI (rec-TRACE,
  b1000 trace) and ADC (rec-ADC). Lesion masks from derivatives/lesion_masks/ are modality
  `mask` with desc `lesion` (combined, 1455), `lesionAcute` (1451) or `lesionChronic` (203).
- The masks are on the TRACE/ADC grid (space-TRACE); T1w and FLAIR are native high-resolution
  volumes in a different space.
- participants.tsv covers 1505 of the 1715; 399 of those rows are n/a in every clinical field.
  Participants without a row are kept with n/a metadata. Ages above 89 are clamped by the
  source ('89+'): age 89 and `age_clamped`.
- No site is given (single cohort from South Carolina; site n/a). `scanner_model` comes from the
  T1w sidecars (6 Philips/GE/Siemens models).
- The mRS is at discharge, not 90 days. `mrs_poor` = mRS 3-6, as in Neuro-JEPA.
- No official split: 60/20/20, stratified by mrs_poor x has combined lesion mask.
- Complete: T1w, FLAIR, DWI, ADC, combined lesion mask and discharge mRS.
"""

import json

import pandas as pd

from brain_datasets import tables

NAME = "soop"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
DIFFUSION = {"rec-TRACE": "DWI", "rec-ADC": "ADC"}
CORE = {("T1w", "n/a"), ("FLAIR", "n/a"), ("DWI", "n/a"), ("ADC", "n/a"), ("mask", "lesion")}


def images() -> pd.DataFrame:
    raw = tables.bids_images(SOURCE, ROOT)
    is_dwi = raw.modality == "dwi"
    raw.loc[is_dwi, "modality"] = raw.desc[is_dwi].map(DIFFUSION)
    raw.loc[is_dwi, "desc"] = None
    masks = tables.bids_images(SOURCE / "derivatives" / "lesion_masks", ROOT)
    assert (masks.desc.str.startswith("space-TRACE_desc-")).all()
    masks["desc"] = masks.desc.str.removeprefix("space-TRACE_desc-")
    images = pd.concat([raw, masks], ignore_index=True)
    assert set(images.modality) == {"T1w", "FLAIR", "DWI", "ADC", "mask"}
    return images


def scanner_model(participant_id: str) -> str | None:
    """ManufacturersModelName from the T1w sidecar (the source names no site or institution)."""
    sidecar = SOURCE / participant_id / "anat" / f"{participant_id}_T1w.json"
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text()).get("ManufacturersModelName")


def samples(participant_ids: list[str]) -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = pd.read_csv(SOURCE / "participants.tsv", sep="\t", na_values="n/a", dtype={"age": str})
    meta = meta.set_index("participant_id").reindex(participant_ids)

    s = pd.DataFrame({"participant_id": participant_ids, "session_id": "1"})
    s["age"] = pd.to_numeric(meta.age.str.removesuffix("+")).values
    s["sex"] = meta.sex.values
    s["site"] = pd.NA
    s["scanner_model"] = [scanner_model(p) for p in participant_ids]
    s["age_clamped"] = meta.age.eq("89+").values
    s["race"] = meta.race.map({"b": "Black", "w": "White"}).values
    s["acute_ischemic_stroke"] = meta.acuteischaemicstroke.astype("Int64").values
    s["prior_stroke"] = meta.priorstroke.astype("Int64").values
    s["bmi"] = meta.bmi.values
    s["nihss"] = meta.nihss.astype("Int64").values
    s["mrs_discharge"] = meta.gs_rankin_6isdeath.astype("Int64").values
    s["mrs_poor"] = (s.mrs_discharge >= 3).where(s.mrs_discharge.notna())
    s["etiology"] = meta.etiology.str.split(":").str[0].astype("Int64").values

    columns = {
        "scanner_model": {"Description": "Scanner model of the T1w (from its JSON sidecar).", "Source": "sub-*_T1w.json ManufacturersModelName"},
        "age_clamped": {"Description": "Age was reported as '89+' (age set to 89)."},
        "race": {"Description": "Race from electronic health records.", "Source": "race", "Levels": {"Black": "Black or African American", "White": "White"}},
        "acute_ischemic_stroke": {"Description": "Acute ischemic stroke diagnosed at admission.", "Source": "acuteischaemicstroke", "Levels": {"1": "yes"}},
        "prior_stroke": {"Description": "Evidence of a prior (chronic) stroke.", "Source": "priorstroke", "Levels": {"0": "no", "1": "yes"}},
        "bmi": {"Description": "Body mass index.", "Source": "bmi"},
        "nihss": {"Description": "NIH Stroke Scale at admission (target).", "Source": "nihss"},
        "mrs_discharge": {"Description": "Modified Rankin Scale at discharge, 0-6 (6 = death) (target).", "Source": "gs_rankin_6isdeath"},
        "mrs_poor": {"Description": "Poor outcome: discharge mRS 3-6 (binary target).", "Source": "gs_rankin_6isdeath"},
        "etiology": {
            "Description": "Stroke etiology (TOAST class).",
            "Source": "etiology",
            "Levels": {"1": "large-artery atherosclerosis", "2": "cardioembolism", "3": "small-vessel disease",
                       "4": "other determined etiology", "5": "cryptogenic"},
        },
    }
    return s, columns


def main() -> None:
    img = images()
    participant_ids = sorted(img.participant_id.unique())
    smp, columns = samples(participant_ids)
    assert len(smp) == 1715

    participants = smp.set_index("participant_id")
    desc = img.desc.fillna("n/a")
    have = img.assign(desc=desc).groupby("participant_id").apply(lambda g: set(zip(g.modality, g.desc)))
    has_mask = have.map(lambda h: ("mask", "lesion") in h)
    complete = have.map(lambda h: CORE <= h) & participants.mrs_discharge.notna()
    strata = participants.mrs_poor.astype(str) + "_mask-" + has_mask.astype(str)
    splits = tables.make_splits(strata, complete)
    tables.write(NAME, img, smp, columns, splits)


if __name__ == "__main__":
    main()

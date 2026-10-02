"""Build the benchmark tables for UPENN-GBM from datasets/upenn_gbm/source/.

    uv run python scripts/upenn_gbm/build_tables.py

See `brain_datasets.tables` for the table layout.

- Sessions: scan IDs `UPENN-GBM-NNNNN_11` (pre-operative baseline, 611) and `_21` (follow-up,
  60; 41 of them have a baseline here) become participant `UPENN-GBM-NNNNN` with session
  `baseline` or `followup`.
- Images (all co-registered, SRI-24 1 mm): T1w, T1c (T1GD), T2w, FLAIR skull-stripped (desc n/a)
  and unstripped (desc `unstripped`); masks `tumor_automated` (611 baselines) and
  `tumor_corrected` (expert-corrected, 147 baselines). Two baselines (00260, 00509) also have an
  older version of their images in an `old/` subfolder: desc `old` / `old_unstripped`.
- Survival: Survival_from_surgery_days_UPDATED is the time to death for 'Deceased'; for 'Alive'
  and 'Lost to Follow-up' it is n/a and Survival_Censor holds the censoring time. These become
  os_days + os_event. The 3 'Deceased - uncertain date of death' are censored at Survival_Censor
  (a lower bound). Follow-up sessions measure survival from the follow-up scan.
  'Not Available' / 'Indeterminate' / 'NOS/NEC' labels are n/a.
- No official split: 60/20/20 by patient, stratified by has corrected mask x OS known.
- Complete: a baseline session with the 8 structural images and the automated mask, and OS.
"""

import re

import pandas as pd

from brain_datasets import tables

NAME = "upenn_gbm"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
NIFTI = SOURCE / "UPENN-GBM" / "NIfTI-files"
SESSIONS = {"11": "baseline", "21": "followup"}
MODALITIES = {"T1": "T1w", "T1GD": "T1c", "T2": "T2w", "FLAIR": "FLAIR"}
SCAN_ID = re.compile(r"(UPENN-GBM-\d{5})_(11|21)")
MISSING = ["Not Available", "Not Applicable", "Indeterminate", "NOS/NEC"]


def parse_scan(scan: str) -> tuple[str, str]:
    """'UPENN-GBM-00001_11' -> ('UPENN-GBM-00001', 'baseline')."""
    match = SCAN_ID.fullmatch(scan)
    assert match, f"unexpected scan ID {scan!r}"
    return match[1], SESSIONS[match[2]]


def images() -> pd.DataFrame:
    rows = []
    for folder in ("images_structural", "images_structural_unstripped"):
        for path in sorted((NIFTI / folder).glob("*/**/*.nii.gz")):
            scan = path.relative_to(NIFTI / folder).parts[0]
            name = path.name.removeprefix(scan + "_").removesuffix(".nii.gz")
            modality = MODALITIES[name.removesuffix("_unstripped")]
            variants = ["old"] if path.parent.name == "old" else []
            variants += ["unstripped"] if name.endswith("_unstripped") else []
            rows.append((*parse_scan(scan), modality, "_".join(variants) or None, path))
    for folder, suffix, desc in (("automated_segm", "_automated_approx_segm", "tumor_automated"),
                                 ("images_segm", "_segm", "tumor_corrected")):
        for path in sorted((NIFTI / folder).glob("*.nii.gz")):
            scan = path.name.removesuffix(suffix + ".nii.gz")
            rows.append((*parse_scan(scan), "mask", desc, path))
    img = pd.DataFrame(rows, columns=tables.IMAGE_COLUMNS)
    img["path"] = img.path.map(lambda p: str(p.relative_to(ROOT)))
    return img


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    meta = pd.read_csv(SOURCE / "UPENN-GBM_clinical_info_v2.1.csv", dtype=str).replace(MISSING, pd.NA)

    s = pd.DataFrame(meta.ID.map(parse_scan).tolist(), columns=["participant_id", "session_id"])
    s["age"] = pd.to_numeric(meta.Age_at_scan_years)
    s["sex"] = meta.Gender
    s["site"] = "UPenn"
    s["days_since_baseline"] = pd.to_numeric(meta.Time_since_baseline_preop).astype("Int64")

    status = meta.Survival_Status
    deceased = status == "Deceased"
    # an uncertain date of death only gives a lower bound on survival: censored at Survival_Censor
    censored = status.isin(["Alive", "Lost to Follow-up", "Deceased - uncertain date of death"])
    death_days = pd.to_numeric(meta.Survival_from_surgery_days_UPDATED)
    censor_days = pd.to_numeric(meta.Survival_Censor)
    assert death_days[deceased].notna().all() and censor_days[censored].notna().all()
    s["os_days"] = death_days.where(deceased, censor_days.where(censored)).astype("Int64")
    s["os_event"] = pd.Series(pd.NA, index=s.index, dtype="Int64").mask(deceased, 1).mask(censored, 0)
    s["survival_status"] = status
    s["idh1"] = meta.IDH1.str.lower()
    s["mgmt"] = meta.MGMT.str.lower()
    s["kps"] = pd.to_numeric(meta.KPS).astype("Int64")
    s["gross_total_resection"] = meta.GTR_over90percent.map({"Y": True, "N": False})
    s["psp_tp_score"] = pd.to_numeric(meta.PsP_TP_score).astype("Int64")

    columns = {
        "days_since_baseline": {"Description": "Days from the baseline pre-operative scan.", "Source": "Time_since_baseline_preop", "Units": "days"},
        "os_days": {
            "Description": "Overall survival: time to death if os_event = 1, else censoring time (target). For baseline "
                           "sessions it is from surgery; for follow-up sessions the source measures it from the "
                           "follow-up scan (about the baseline value minus days_since_baseline). Use the baseline "
                           "session for survival tasks.",
            "Source": "Survival_from_surgery_days_UPDATED, Survival_Censor",
            "Units": "days",
        },
        "os_event": {"Description": "Death observed.", "Source": "Survival_Status", "Levels": {"1": "deceased", "0": "censored: alive, lost to follow-up, or deceased with an uncertain date"}},
        "survival_status": {"Description": "Survival status as reported.", "Source": "Survival_Status"},
        "idh1": {"Description": "IDH1 mutation status (target).", "Source": "IDH1", "Levels": {"wildtype": "", "mutated": ""}},
        "mgmt": {"Description": "MGMT promoter methylation (target).", "Source": "MGMT", "Levels": {"methylated": "", "unmethylated": ""}},
        "kps": {"Description": "Karnofsky performance status.", "Source": "KPS"},
        "gross_total_resection": {"Description": "More than 90% of the tumor resected.", "Source": "GTR_over90percent"},
        "psp_tp_score": {"Description": "Pseudoprogression vs true progression score, 1-6 (follow-ups only).", "Source": "PsP_TP_score"},
    }
    return s, columns


def main() -> None:
    img = images()
    smp, columns = samples()
    assert len(smp) == 671 and smp.participant_id.nunique() == 630
    keys = ["participant_id", "session_id"]
    assert set(img[keys].itertuples(index=False)) <= set(smp[keys].itertuples(index=False))
    assert (img.desc == "tumor_corrected").sum() == 147 and (img.desc == "tumor_automated").sum() == 611

    participants = smp.groupby("participant_id").first()
    baseline_images = img[img.session_id == "baseline"].fillna({"desc": "n/a"})
    have = baseline_images.groupby("participant_id").apply(lambda g: set(zip(g.modality, g.desc)))
    core = {(m, d) for m in MODALITIES.values() for d in ("n/a", "unstripped")} | {("mask", "tumor_automated")}
    has_core = have.map(lambda h: core <= h).reindex(participants.index, fill_value=False)
    has_corrected = have.map(lambda h: ("mask", "tumor_corrected") in h).reindex(participants.index, fill_value=False)
    baseline = smp[smp.session_id == "baseline"].set_index("participant_id")
    os_known = (baseline.os_days.notna() & baseline.os_event.notna()).reindex(participants.index, fill_value=False)

    complete = has_core & os_known
    strata = "corrected-" + has_corrected.astype(str) + "_os-" + os_known.astype(str)
    splits = tables.make_splits(strata, complete)
    tables.write(NAME, img, smp, columns, splits)


if __name__ == "__main__":
    main()

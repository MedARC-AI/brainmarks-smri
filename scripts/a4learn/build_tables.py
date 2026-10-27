"""Build the benchmark tables for A4/LEARN from datasets/a4learn/source/.

    uv run python scripts/a4learn/build_tables.py

Table layout: `README_hf.md`.

- Participants: A4 BID (sub-B10081264). Groups: A4 (amyloid-elevated, randomized to solanezumab or
  placebo), LEARN (amyloid not elevated, observational) and SF (screen-fails with an MRI).
  178 people were rescreened under a new BID (PREVBID); no such pair has T1w images under both.
- Sessions (select_images.py): bl (first visit with a T1w; 004 or 006), wk84 (027), wk240 (066).
  One T1w per session, as distributed (de-faced NIfTI).
- Dates are days from consent. 5 LEARN wk240 sessions have no date in any table (age n/a). Visit-level targets come from the same visit (MRI volumes, WMH,
  MRI reads) or the nearest record: cognition within 90 days, PET and plasma within 1 year.
- PACC change: from the A4 analysis dataset (ADQS QSCHANGE) at week 240, per participant.
- Codes (SEX, RACE, ETHNIC) are mapped with the A4 data dictionary (derived_datadic.csv, part of the
  A4 documentation on IDA). Site IDs are not shared.
- No official split: 60/20/20 by participant, stratified by group (A4 placebo, A4 solanezumab,
  LEARN, SF). Complete: bl T1w and amyloid PET SUVR.
"""

import re

import pandas as pd

from brainmarks_smri.utils import tables

NAME = "a4learn"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
DERIVED = SOURCE / "Clinical" / "Derived Data"
EXTERNAL = SOURCE / "Clinical" / "External Data"

# SUBJINFO codes, from the A4 data dictionary (derived_datadic.csv)
SEX = {1: "F", 2: "M"}
RACE = {
    1: "White",
    2: "Black or African American",
    58: "Asian",
    79: "Native Hawaiian or Other Pacific Islander",
    84: "American Indian or Alaskan Native",
    97: None,  # unknown or not reported
    100: "More than one race",
}
ETHNICITY = {50: "Hispanic or Latino", 56: "Not Hispanic or Latino", 97: None}

# ADQS test code -> (column, description)
SCORES = {
    "PACC": ("pacc", "Preclinical Alzheimer Cognitive Composite (z-score sum)."),
    "MMSE": ("mmse", "Mini-Mental State Examination (0-30)."),
    "FCSRT96": ("fcsrt96", "Free and Cued Selective Reminding Test (0-96)."),
    "LMIIa": ("logical_memory_delayed", "Logical Memory delayed paragraph recall (0-25)."),
    "DIGIT": ("digit_symbol", "Digit Symbol Substitution score."),
    "CFITOTAL": ("cfi", "Cognitive Function Index, participant + partner."),
    "ADLTOTAL": ("adl", "ADCS Activities of Daily Living (prevention), participant + partner."),
    "CDGLOBAL": ("cdr_global", "Clinical Dementia Rating, global score."),
}


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"BID": str, "VISCODE": str, "VISITCD": str}, low_memory=False)


def nearest(
    sessions: pd.DataFrame, table: pd.DataFrame, day_column: str, columns: dict[str, str], days: int
) -> pd.DataFrame:
    """Join `columns` (source -> target) from the record of the same BID nearest the scan day."""
    table = table.dropna(subset=[day_column]).dropna(subset=list(columns), how="all")
    merged = sessions[["bid", "session_id", "scan_day"]].merge(
        table[["BID", day_column, *columns]], left_on="bid", right_on="BID"
    )
    merged["gap"] = (merged[day_column] - merged.scan_day).abs()
    merged = merged[merged.gap <= days].sort_values(["gap", day_column])
    merged = merged.groupby(["bid", "session_id"]).head(1)
    merged = merged.rename(columns=columns)[["bid", "session_id", *columns.values()]]
    return sessions.merge(merged, on=["bid", "session_id"], how="left")


def same_visit(
    sessions: pd.DataFrame, table: pd.DataFrame, columns: dict[str, str]
) -> pd.DataFrame:
    table = table.rename(columns={"BID": "bid", "VISCODE": "viscode", **columns})
    table = table.drop_duplicates(["bid", "viscode"])[["bid", "viscode", *columns.values()]]
    return sessions.merge(table, on=["bid", "viscode"], how="left")


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    selected = pd.read_csv(SOURCE / "selected_images.tsv", sep="\t", dtype=str)
    s = selected[["participant_id", "session_id", "bid", "viscode"]].copy()

    # Scan day: the MRI study date from the MRI reads, else the visit date.
    reads = read_csv(EXTERNAL / "imaging_MRI_reads.csv").drop_duplicates(["BID", "VISCODE"])
    reads = reads.set_index(["BID", "VISCODE"])[["STUDYDATE_DAYS_CONSENT", "STUDYDATE_DAYS_T0"]]
    visits = read_csv(DERIVED / "SV.csv")
    visits = visits.groupby(["BID", "VISITCD"])[["SVSTDTC_DAYS_CONSENT", "SVSTDTC_DAYS_T0"]].min()
    key = pd.MultiIndex.from_frame(s[["bid", "viscode"]])
    s["scan_day"] = reads.STUDYDATE_DAYS_CONSENT.reindex(key).values
    s["scan_day"] = s.scan_day.fillna(
        pd.Series(visits.SVSTDTC_DAYS_CONSENT.reindex(key).values, index=s.index)
    )
    s["days_from_randomization"] = reads.STUDYDATE_DAYS_T0.reindex(key).values
    s["days_from_randomization"] = s.days_from_randomization.fillna(
        pd.Series(visits.SVSTDTC_DAYS_T0.reindex(key).values, index=s.index)
    )

    info = read_csv(DERIVED / "SUBJINFO.csv").set_index("BID")
    s["group"] = s.bid.map(info.SUBSTUDY)
    s["treatment"] = s.bid.map(info.TX)
    s["age"] = (s.bid.map(info.AGEYR) + s.scan_day / 365.25).round(1)
    s["sex"] = s.bid.map(info.SEX.map(SEX))
    s["education"] = s.bid.map(info.EDCCNTU).astype("Int64")
    assert info.RACE.isin(list(RACE)).all() and info.ETHNIC.isin(list(ETHNICITY)).all()
    s["race"] = s.bid.map(info.RACE.map(RACE))
    s["ethnicity"] = s.bid.map(info.ETHNIC.map(ETHNICITY))
    s["apoe_genotype"] = s.bid.map(info.APOEGN)
    s["apoe4"] = s.apoe_genotype.str.count("4").astype("Int64")
    s["screening_centiloid"] = s.bid.map(info.AMYLCENT)
    s["screening_amyloid_suvr"] = s.bid.map(info.SUVRCER)

    adqs = read_csv(DERIVED / "ADQS.csv")
    for code, (column, _) in SCORES.items():
        part = adqs[adqs.QSTESTCD == code]
        s = nearest(s, part, "QSDTC_DAYS_CONSENT", {"QSSTRESN": column}, 90)
    cdr_sb = adqs[adqs.QSTESTCD.isin(["CDRSB", "CDSOB"])]
    s = nearest(s, cdr_sb, "QSDTC_DAYS_CONSENT", {"QSSTRESN": "cdr_sb"}, 90)
    pacc_wk240 = (
        adqs[(adqs.QSTESTCD == "PACC") & (adqs.VISITCD == "066")].groupby("BID").QSCHANGE.first()
    )
    s["pacc_change_wk240"] = s.bid.map(pacc_wk240)

    amyloid = read_csv(EXTERNAL / "imaging_SUVR_amyloid.csv")
    amyloid = amyloid[amyloid.brain_region == "Composite_Summary"]
    s = nearest(s, amyloid, "scan_date_DAYS_CONSENT", {"suvr_cer": "amyloid_suvr"}, 365)
    # Visual reads exist only for the screening PET, and their days don't match the visit dates.
    visual = read_csv(EXTERNAL / "imaging_PET_VA.csv").drop_duplicates("BID").set_index("BID")
    s["screening_amyloid_read"] = s.bid.map(
        visual.overall_score.map({"positive": 1, "negative": 0})
    ).astype("Int64")

    tau = read_csv(EXTERNAL / "imaging_SUVR_tau.csv")
    tau = tau.pivot_table(
        index=["BID", "scan_date_DAYS_CONSENT"], columns="brain_region", values="suvr_persi"
    ).reset_index()
    tau["entorhinal"] = tau[["entorhinal_lh_VOI", "entorhinal_rh_VOI"]].mean(axis=1)
    tau_columns = {
        "MUBADA Mask": "tau_mubada_suvr",
        "EarlyTau_VOI": "tau_early_suvr",
        "entorhinal": "tau_entorhinal_suvr",
    }
    s = nearest(s, tau, "scan_date_DAYS_CONSENT", tau_columns, 365)

    ptau = read_csv(EXTERNAL / "biomarker_pTau217.csv")
    ptau = ptau[ptau.TESTCD == "PTAU217"].copy()
    ptau["value"] = pd.to_numeric(ptau.ORRES, errors="coerce")
    s = nearest(s, ptau, "COLLECTION_DATE_DAYS_CONSENT", {"value": "plasma_ptau217"}, 365)

    volumes = read_csv(EXTERNAL / "imaging_volumetric_mri.csv")
    volume_codes = [
        c
        for c in volumes.columns
        if c not in {"SUBSTUDY", "BID", "VISCODE", "Date_DAYS_CONSENT", "Date_DAYS_T0"}
    ]
    volume_columns = {code: "vol_" + snake(code) for code in volume_codes}
    volumes[volume_codes] = volumes[volume_codes].where(volumes[volume_codes] >= 0)  # -4 = missing
    s = same_visit(s, volumes, volume_columns)
    wmh = read_csv(EXTERNAL / "imaging_FLAIR_WMH_QC.csv")
    wmh = wmh[wmh.QC == "pass"].copy()
    wmh["wmh"] = pd.to_numeric(wmh.WMH_corrected, errors="coerce")
    s = same_visit(s, wmh, {"wmh": "wmh_corrected"})
    mri_reads = read_csv(EXTERNAL / "imaging_MRI_reads.csv")
    s = same_visit(
        s, mri_reads, {"Total #MCH": "microhemorrhages", "Total #SS": "superficial_siderosis"}
    )

    columns = {
        "bid": {"Description": "A4 participant ID (BID).", "Source": "BID"},
        "viscode": {"Description": "A4 visit code of the MRI.", "Source": "VISCODE"},
        "scan_day": {
            "Description": "Day of the MRI, from consent.",
            "Source": "imaging_MRI_reads STUDYDATE_DAYS_CONSENT, else SV SVSTDTC_DAYS_CONSENT",
            "Units": "days",
        },
        "days_from_randomization": {
            "Description": "Day of the MRI, from randomization (A4) or baseline (LEARN).",
            "Source": "imaging_MRI_reads STUDYDATE_DAYS_T0, else SV SVSTDTC_DAYS_T0",
            "Units": "days",
        },
        "group": {
            "Description": "Substudy.",
            "Source": "SUBJINFO SUBSTUDY",
            "Levels": {
                "A4": "amyloid elevated, randomized",
                "LEARN": "amyloid not elevated, observational",
                "SF": "screen-fail",
            },
        },
        "treatment": {
            "Description": "Randomized treatment (A4 only).",
            "Source": "SUBJINFO TX",
            "Levels": {"Placebo": "placebo", "Solanezumab": "solanezumab"},
        },
        "education": {
            "Description": "Years of education.",
            "Source": "SUBJINFO EDCCNTU",
            "Units": "years",
        },
        "race": {
            "Description": "Race (n/a: unknown or not reported).",
            "Source": "SUBJINFO RACE",
        },
        "ethnicity": {
            "Description": "Ethnicity (n/a: unknown or not reported).",
            "Source": "SUBJINFO ETHNIC",
        },
        "apoe_genotype": {"Description": "APOE genotype.", "Source": "SUBJINFO APOEGN"},
        "apoe4": {"Description": "Number of APOE e4 alleles (0-2).", "Source": "SUBJINFO APOEGN"},
        "screening_centiloid": {
            "Description": "Florbetapir Centiloid at screening (per participant).",
            "Source": "SUBJINFO AMYLCENT",
        },
        "screening_amyloid_read": {
            "Description": "Florbetapir visual read at screening (per participant).",
            "Source": "imaging_PET_VA overall_score",
            "Levels": {"0": "negative", "1": "positive"},
        },
        "screening_amyloid_suvr": {
            "Description": "Florbetapir cortical SUVR (cerebellar reference) at screening (per participant).",
            "Source": "SUBJINFO SUVRCER",
        },
        **{
            column: {"Description": f"{text} Within 90 days of the MRI.", "Source": f"ADQS {code}"}
            for code, (column, text) in SCORES.items()
        },
        "cdr_sb": {
            "Description": "Clinical Dementia Rating, sum of boxes. Within 90 days of the MRI.",
            "Source": "ADQS CDRSB, CDSOB",
        },
        "pacc_change_wk240": {
            "Description": "PACC change from baseline at week 240 (per participant).",
            "Source": "ADQS PACC QSCHANGE at visit 066",
        },
        "amyloid_suvr": {
            "Description": "Florbetapir composite SUVR (cerebellar reference), within 1 year of the MRI.",
            "Source": "imaging_SUVR_amyloid Composite_Summary suvr_cer",
        },
        "tau_mubada_suvr": {
            "Description": "Flortaucipir MUBADA SUVR (PERSI reference), within 1 year.",
            "Source": "imaging_SUVR_tau 'MUBADA Mask' suvr_persi",
        },
        "tau_early_suvr": {
            "Description": "Flortaucipir early-tau region SUVR (PERSI reference), within 1 year.",
            "Source": "imaging_SUVR_tau EarlyTau_VOI suvr_persi",
        },
        "tau_entorhinal_suvr": {
            "Description": "Flortaucipir entorhinal SUVR, mean of left and right (PERSI reference), within 1 year.",
            "Source": "imaging_SUVR_tau entorhinal_{lh,rh}_VOI suvr_persi",
        },
        "plasma_ptau217": {
            "Description": "Plasma pTau217 (Lilly), within 1 year of the MRI.",
            "Source": "biomarker_pTau217 ORRES",
            "Units": "U/mL",
        },
        **{
            name: {
                "Description": f"Volume of {code} (A4 volumetric MRI, same visit).",
                "Source": f"imaging_volumetric_mri {code}",
            }
            for code, name in volume_columns.items()
        },
        "wmh_corrected": {
            "Description": "White matter hyperintensity volume, ICV-corrected (QC pass, same visit).",
            "Source": "imaging_FLAIR_WMH_QC WMH_corrected",
        },
        "microhemorrhages": {
            "Description": "Number of microhemorrhages on the MRI read (same visit).",
            "Source": "imaging_MRI_reads 'Total #MCH'",
        },
        "superficial_siderosis": {
            "Description": "Number of superficial siderosis areas on the MRI read (same visit).",
            "Source": "imaging_MRI_reads 'Total #SS'",
        },
    }
    s = s[["participant_id", "session_id", "age", "sex", *columns]]
    return s, columns


def images(selected: pd.DataFrame) -> pd.DataFrame:
    img = selected[["participant_id", "session_id"]].copy()
    img["modality"] = "T1w"
    img["desc"] = None
    img["path"] = "source/" + selected.file
    return img


def main() -> None:
    selected = pd.read_csv(SOURCE / "selected_images.tsv", sep="\t", dtype=str)
    img = images(selected)
    smp, columns = samples()
    info = read_csv(DERIVED / "SUBJINFO.csv")
    rescreened = info.dropna(subset=["PREVBID"])
    both = rescreened.BID.isin(smp.bid) & rescreened.PREVBID.isin(smp.bid)
    assert not both.any(), "a rescreened person has images under two BIDs"
    assert smp.scan_day.isna().sum() == 5, (
        "scan day missing beyond the 5 known LEARN wk240 sessions"
    )
    assert smp.age.dropna().between(60, 95).all()

    bl = smp[smp.session_id == "bl"].set_index("participant_id")
    complete = bl.amyloid_suvr.notna()
    strata = bl.group + " " + bl.treatment.fillna("")
    splits = tables.make_splits(strata, complete)
    tables.write(NAME, img, smp, columns, splits, track_samples=False)
    print(tables.summary_table(smp, splits, ["session_id", "group", "screening_amyloid_read"]))


if __name__ == "__main__":
    main()

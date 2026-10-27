"""Build the benchmark tables for ADNI from datasets/adni/source/.

    uv run python scripts/adni/build_tables.py

Table layout: `README_hf.md`.

- Participants: ADNI PTID without underscores (002_S_0295 -> sub-002S0295). Sessions: y0 (first
  visit with a usable T1w), y2 and y4 (visits nearest 2 and 4 years later, within 6 months).
  Image choice: select_images.py.
- Images per session: T1w (1.5 T in ADNI1, else 3 T), `acq-3T` T1w at ADNI1 visits that also
  have a 3 T scan, FLAIR (ADNIGO onward).
- Targets come from ADNIMERGE2 0.1.1, matched to the scan date by participant and the nearest
  record within a window: diagnosis 180 days; cognitive scores 90 days; amyloid and tau PET, CSF
  and plasma 1 year; FreeSurfer and WMH 60 days. Missing codes (negative values) -> n/a.
- Progression: years from the scan to the first later dementia diagnosis, to the first later MCI
  or dementia diagnosis, and years of diagnostic follow-up, so tasks can set their own horizon
  (e.g. pMCI vs sMCI at 3 years, stable CN).
- FreeSurfer 7 (UCSFFSX7): subcortical volumes, cortical volumes and mean cortical thickness.
  Most rows are not yet QC'd (fs_status 'partial').
- No official split: 60/20/20 by participant, stratified by y0 diagnosis x enrolling protocol.
- Complete: y0 T1w and y0 diagnosis.
"""

import json
import re

import numpy as np
import pandas as pd
import pyreadr

from brainmarks_smri.utils import tables

NAME = "adni"
ROOT = tables.dataset_dir(NAME)
SOURCE = ROOT / "source"
RDA = SOURCE / "ADNIMERGE2" / "data"

# ADQS parameter code -> (column, description)
SCORES = {
    "CDGLOBAL": ("cdr_global", "Clinical Dementia Rating, global score (0, 0.5, 1, 2, 3)."),
    "CDRSB": ("cdr_sb", "Clinical Dementia Rating, sum of boxes (0-18)."),
    "MMSCORE": ("mmse", "Mini-Mental State Examination (0-30)."),
    "ADASTT11": ("adas11", "ADAS-Cog 11-item total (0-70, higher is worse)."),
    "ADASTT13": ("adas13", "ADAS-Cog 13-item total (0-85, higher is worse)."),
    "MOCA": ("moca", "Montreal Cognitive Assessment (0-30)."),
    "FAQTOTAL": ("faq", "Functional Activities Questionnaire total (0-30, higher is worse)."),
    "MPACCDIGIT": ("mpacc_digit", "Modified Preclinical Alzheimer Cognitive Composite (digit)."),
    "MPACCTRAILSB": ("mpacc_trailsb", "Modified PACC with Trails B in place of digit symbol."),
    "RAVLTIMM": ("ravlt_immediate", "RAVLT immediate recall, sum of 5 trials."),
    "RAVLTLRN": ("ravlt_learning", "RAVLT learning (trial 5 - trial 1)."),
    "RAVLTFG": ("ravlt_forgetting", "RAVLT forgetting (trial 5 - delayed)."),
    "RAVLTFGP": ("ravlt_perc_forgetting", "RAVLT percent forgetting."),
    "LDELTOTL": ("logical_memory_delayed", "Logical Memory delayed recall."),
    "LIMMTOTL": ("logical_memory_immediate", "Logical Memory immediate recall."),
    "DIGITSCR": ("digit_symbol", "Digit Symbol Substitution score."),
    "TRABSCOR": ("trails_b", "Trail Making Test part B time (seconds)."),
    "GDTOTAL": ("gds", "Geriatric Depression Scale (0-15)."),
    "NPIQTOTL": ("npiq", "Neuropsychiatric Inventory Questionnaire total."),
    "ECOGPTTT": ("ecog_participant", "Everyday Cognition, participant report, total."),
    "ECOGSPTT": ("ecog_partner", "Everyday Cognition, study partner report, total."),
}
FREESURFER_KINDS = {"SV": "vol", "CV": "ctxvol", "TA": "thick"}


def read_rda(name: str) -> pd.DataFrame:
    return next(iter(pyreadr.read_r(RDA / f"{name}.rda").values()))


def nearest(
    sessions: pd.DataFrame,
    table: pd.DataFrame,
    date_column: str,
    columns: dict[str, str],
    days: int,
) -> pd.DataFrame:
    """Join `columns` (source -> target) from the record of the same PTID nearest the scan date."""
    table = table.dropna(subset=[date_column]).copy()
    table["record_date"] = pd.to_datetime(table[date_column])
    table = table.dropna(subset=list(columns), how="all")
    merged = sessions[["ptid", "session_id", "scan_date"]].merge(
        table[["PTID", "record_date", *columns]], left_on="ptid", right_on="PTID"
    )
    merged["gap"] = (merged.record_date - merged.scan_date).dt.days.abs()
    merged = merged[merged.gap <= days].sort_values(["gap", "record_date"])
    merged = merged.groupby(["ptid", "session_id"]).head(1)
    merged = merged.rename(columns=columns)[["ptid", "session_id", *columns.values()]]
    return sessions.merge(merged, on=["ptid", "session_id"], how="left")


def demographics(sessions: pd.DataFrame) -> pd.DataFrame:
    demog = read_rda("PTDEMOG").sort_values(["PTID", "VISDATE"])
    first = demog.groupby("PTID").first()  # first non-null value per column
    birth = pd.to_datetime(first.PTDOB, format="%m/%Y", errors="coerce")
    birth = birth.fillna(pd.to_datetime(first.PTDOB, format="%Y-%m-%d", errors="coerce"))
    birth = birth.fillna(
        pd.to_datetime(first.PTDOBYY.astype("Int64").astype(str), format="%Y", errors="coerce")
    )
    birth = birth + pd.Timedelta(days=14)  # middle of the birth month
    sessions["age"] = ((sessions.scan_date - sessions.ptid.map(birth)).dt.days / 365.25).round(1)
    sessions["sex"] = sessions.ptid.map(first.PTGENDER.map({"Male": "M", "Female": "F"}))
    sessions["education"] = sessions.ptid.map(first.PTEDUCAT).astype("Int64")
    sessions["ethnicity"] = sessions.ptid.map(first.PTETHCAT)
    sessions["race"] = sessions.ptid.map(first.PTRACCAT)
    sessions["handedness"] = sessions.ptid.map(first.PTHAND)
    apoe = read_rda("APOERES").dropna(subset=["GENOTYPE"]).groupby("PTID").GENOTYPE.first()
    sessions["apoe_genotype"] = sessions.ptid.map(apoe)
    sessions["apoe4"] = sessions.apoe_genotype.str.count("4").astype("Int64")
    return sessions


def diagnosis(sessions: pd.DataFrame) -> pd.DataFrame:
    dx = read_rda("DXSUM")
    sessions["protocol"] = sessions.ptid.map(
        dx.dropna(subset=["ORIGPROT"]).groupby("PTID").ORIGPROT.first()
    )
    sessions = nearest(sessions, dx, "EXAMDATE", {"DIAGNOSIS": "dx"}, 180)
    sessions["dx"] = sessions.dx.astype(object)

    dx = dx.dropna(subset=["EXAMDATE", "DIAGNOSIS"]).copy()
    dx["date"] = pd.to_datetime(dx.EXAMDATE)
    merged = sessions[["ptid", "session_id", "scan_date"]].merge(
        dx[["PTID", "date", "DIAGNOSIS"]], left_on="ptid", right_on="PTID"
    )
    merged["years"] = (merged.date - merged.scan_date).dt.days / 365.25
    after = merged[merged.years > 0]
    followup = after.groupby(["ptid", "session_id"]).years.max().rename("dx_followup_years")
    dementia = after[after.DIAGNOSIS == "Dementia"].groupby(["ptid", "session_id"]).years.min()
    impaired = after[after.DIAGNOSIS.isin(["MCI", "Dementia"])]
    impairment = impaired.groupby(["ptid", "session_id"]).years.min()
    sessions = sessions.join(followup, on=["ptid", "session_id"])
    sessions = sessions.join(dementia.rename("years_to_dementia"), on=["ptid", "session_id"])
    sessions = sessions.join(impairment.rename("years_to_impairment"), on=["ptid", "session_id"])
    sessions["dx_followup_years"] = sessions.dx_followup_years.round(2)
    sessions["years_to_dementia"] = sessions.years_to_dementia.round(2)
    sessions["years_to_impairment"] = sessions.years_to_impairment.round(2)
    return sessions


def scores(sessions: pd.DataFrame) -> pd.DataFrame:
    adqs = read_rda("ADQS")
    rid_to_ptid = read_rda("DXSUM").dropna(subset=["RID"]).groupby("RID").PTID.first()
    adqs["PTID"] = adqs.SUBJID.astype(float).map(rid_to_ptid)
    adqs.loc[adqs.AVAL < 0, "AVAL"] = np.nan
    for code, (column, _) in SCORES.items():
        part = adqs[adqs.PARAMCD == code]
        sessions = nearest(sessions, part, "ADT", {"AVAL": column}, 90)
    return sessions


def biomarkers(sessions: pd.DataFrame) -> pd.DataFrame:
    amyloid = read_rda("UCBERKELEY_AMY_6MM")
    amyloid = amyloid[amyloid.qc_flag.isin([1, 2, -1])]  # drop failed (0) and unprocessable (-2)
    sessions = nearest(
        sessions,
        amyloid,
        "SCANDATE",
        {
            "AMYLOID_STATUS": "amyloid_status",
            "CENTILOIDS": "centiloid",
            "SUMMARY_SUVR": "amyloid_suvr",
            "TRACER": "amyloid_tracer",
        },
        365,
    )
    sessions["amyloid_status"] = sessions.amyloid_status.astype("Int64")
    tau = read_rda("UCBERKELEY_TAU_6MM")
    tau = tau[tau.qc_flag.isin([1, 2, -1])]
    sessions = nearest(
        sessions,
        tau,
        "SCANDATE",
        {
            "META_TEMPORAL_SUVR": "tau_meta_temporal_suvr",
            "CTX_ENTORHINAL_SUVR": "tau_entorhinal_suvr",
            "TRACER": "tau_tracer",
        },
        365,
    )
    csf = read_rda("UPENNBIOMK_ROCHE_ELECSYS")
    sessions = nearest(
        sessions,
        csf,
        "EXAMDATE",
        {
            "ABETA40": "csf_abeta40",
            "ABETA42": "csf_abeta42",
            "TAU": "csf_tau",
            "PTAU": "csf_ptau181",
        },
        365,
    )
    plasma = read_rda("UPENN_PLASMA_FUJIREBIO_QUANTERIX")
    plasma_columns = {
        "pT217_F": "plasma_ptau217",
        "AB42_AB40_F": "plasma_ab42_ab40",
        "NfL_Q": "plasma_nfl",
        "GFAP_Q": "plasma_gfap",
    }
    for column in plasma_columns:
        plasma.loc[plasma[column] < 0, column] = np.nan
    sessions = nearest(sessions, plasma, "EXAMDATE", plasma_columns, 365)
    wmh = read_rda("UCD_WMH")
    sessions = nearest(sessions, wmh, "EXAMDATE", {"TOTAL_WMH": "wmh_volume"}, 60)
    return sessions


def freesurfer(sessions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    fs = read_rda("UCSFFSX7")
    dictionary = read_rda("DATADIC")
    dictionary = dictionary[dictionary.TBLNAME == "UCSFFSX7"].set_index("FLDNAME").TEXT
    columns = {"OVERALLQC": "fs_qc", "STATUS": "fs_status"}
    docs = {
        "fs_qc": {
            "Description": "FreeSurfer overall QC rating (n/a: not rated).",
            "Source": "UCSFFSX7 OVERALLQC",
        },
        "fs_status": {
            "Description": "FreeSurfer row status: 'complete' = QC'd and final, 'partial' = no QC.",
            "Source": "UCSFFSX7 STATUS",
        },
    }
    for code in fs.columns:
        match = re.fullmatch(r"ST\d+(SV|CV|TA)", code)
        if not match:
            continue
        text = dictionary[code]
        region = text.rsplit(" of ", 1)[1]
        name = (
            "fs_"
            + FREESURFER_KINDS[match[1]]
            + "_"
            + re.sub(r"(?<!^)(?=[A-Z])", "_", region).lower()
        )
        columns[code] = name
        units = "mm" if match[1] == "TA" else "mm^3"
        docs[name] = {
            "Description": f"FreeSurfer 7 (UCSF): {text}.",
            "Source": f"UCSFFSX7 {code}",
            "Units": units,
        }
    assert len(set(columns.values())) == len(columns), "duplicate FreeSurfer column names"
    sessions = nearest(sessions, fs, "EXAMDATE", columns, 60)
    return sessions, docs


def manufacturers(sessions: pd.DataFrame) -> pd.Series:
    """Scanner manufacturer of each session's main T1w, from its dcm2niix sidecar."""
    values = []
    for row in sessions.itertuples():
        sidecar = SOURCE / row.participant_id / f"ses-{row.session_id}" / "anat"
        sidecar = sidecar / f"{row.participant_id}_ses-{row.session_id}_T1w.json"
        values.append(json.loads(sidecar.read_text()).get("Manufacturer", "").upper()[:7] or None)
    return pd.Series(values, index=sessions.index).replace(
        {"PHILIPS": "Philips", "SIEMENS": "Siemens"}
    )


def samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    selected = pd.read_csv(SOURCE / "selected_images.tsv", sep="\t", na_values="n/a")
    dates = pd.read_csv(SOURCE / "image_dates.tsv", sep="\t")
    selected = selected.merge(dates, on="image_id", validate="one_to_one")
    main_t1w = selected[(selected.modality == "T1w") & selected.desc.isna()]
    s = main_t1w[
        [
            "participant_id",
            "session_id",
            "ptid",
            "acq_date",
            "years",
            "phase",
            "field_strength",
            "series_description",
        ]
    ].copy()
    s = s.rename(
        columns={
            "acq_date": "scan_date",
            "years": "years_from_y0",
            "series_description": "t1w_series",
        }
    )
    s["scan_date"] = pd.to_datetime(s.scan_date)
    s["site"] = s.ptid.str[:3]
    s["manufacturer"] = manufacturers(s)
    s = demographics(s)
    s = diagnosis(s)
    s = scores(s)
    s = biomarkers(s)
    s, fs_docs = freesurfer(s)
    s["scan_date"] = s.scan_date.dt.strftime("%Y-%m-%d")
    s["phase"] = s.phase.replace({"UNKNOWN": None})

    columns = {
        "ptid": {"Description": "ADNI participant ID (PTID: site_S_number).", "Source": "PTID"},
        "scan_date": {
            "Description": "Date of the main T1w scan.",
            "Source": "IDA acquisition date",
        },
        "years_from_y0": {"Description": "Years from the y0 scan to this scan.", "Units": "years"},
        "phase": {
            "Description": "ADNI phase protocol the scan was acquired under (n/a: not listed).",
            "Source": "MRIQC MRIProtocolPhase",
        },
        "field_strength": {"Description": "Field strength of the main T1w.", "Units": "T"},
        "t1w_series": {"Description": "IDA series description of the main T1w."},
        "manufacturer": {
            "Description": "Scanner manufacturer of the main T1w.",
            "Source": "DICOM Manufacturer",
        },
        "education": {
            "Description": "Years of education.",
            "Source": "PTDEMOG PTEDUCAT",
            "Units": "years",
        },
        "ethnicity": {"Description": "Ethnicity, as reported.", "Source": "PTDEMOG PTETHCAT"},
        "race": {"Description": "Race, as reported.", "Source": "PTDEMOG PTRACCAT"},
        "handedness": {"Description": "Handedness.", "Source": "PTDEMOG PTHAND"},
        "apoe_genotype": {"Description": "APOE genotype.", "Source": "APOERES GENOTYPE"},
        "apoe4": {"Description": "Number of APOE e4 alleles (0-2).", "Source": "APOERES GENOTYPE"},
        "protocol": {
            "Description": "ADNI phase the participant enrolled in.",
            "Source": "DXSUM ORIGPROT",
        },
        "dx": {
            "Description": "Clinical diagnosis within 180 days of the scan.",
            "Source": "DXSUM DIAGNOSIS",
            "Levels": {
                "CN": "cognitively normal",
                "MCI": "mild cognitive impairment",
                "Dementia": "dementia",
            },
        },
        "dx_followup_years": {
            "Description": "Years from the scan to the last later diagnosis (n/a: none).",
            "Source": "DXSUM",
            "Units": "years",
        },
        "years_to_dementia": {
            "Description": "Years from the scan to the first later dementia diagnosis (n/a: none).",
            "Source": "DXSUM",
            "Units": "years",
        },
        "years_to_impairment": {
            "Description": "Years from the scan to the first later MCI or dementia diagnosis (n/a: none).",
            "Source": "DXSUM",
            "Units": "years",
        },
        **{
            column: {"Description": f"{text} Within 90 days of the scan.", "Source": f"ADQS {code}"}
            for code, (column, text) in SCORES.items()
        },
        "amyloid_status": {
            "Description": "Amyloid PET positivity (UC Berkeley cortical summary threshold), within 1 year.",
            "Source": "UCBERKELEY_AMY_6MM AMYLOID_STATUS",
            "Levels": {"0": "negative", "1": "positive"},
        },
        "centiloid": {
            "Description": "Amyloid PET Centiloid, within 1 year.",
            "Source": "UCBERKELEY_AMY_6MM CENTILOIDS",
        },
        "amyloid_suvr": {
            "Description": "Amyloid PET cortical summary SUVR (whole cerebellum reference), within 1 year.",
            "Source": "UCBERKELEY_AMY_6MM SUMMARY_SUVR",
        },
        "amyloid_tracer": {
            "Description": "Amyloid PET tracer.",
            "Source": "UCBERKELEY_AMY_6MM TRACER",
            "Levels": {"FBP": "florbetapir", "FBB": "florbetaben", "NAV": "NAV4694"},
        },
        "tau_meta_temporal_suvr": {
            "Description": "Tau PET meta-temporal SUVR (inferior cerebellum reference), within 1 year.",
            "Source": "UCBERKELEY_TAU_6MM META_TEMPORAL_SUVR",
        },
        "tau_entorhinal_suvr": {
            "Description": "Tau PET entorhinal SUVR, within 1 year.",
            "Source": "UCBERKELEY_TAU_6MM CTX_ENTORHINAL_SUVR",
        },
        "tau_tracer": {
            "Description": "Tau PET tracer.",
            "Source": "UCBERKELEY_TAU_6MM TRACER",
            "Levels": {"FTP": "flortaucipir", "MK6240": "MK-6240", "PI2620": "PI-2620"},
        },
        "csf_abeta40": {
            "Description": "CSF Abeta40 (Roche Elecsys), within 1 year.",
            "Source": "UPENNBIOMK_ROCHE_ELECSYS ABETA40",
            "Units": "pg/mL",
        },
        "csf_abeta42": {
            "Description": "CSF Abeta42 (Roche Elecsys), within 1 year.",
            "Source": "UPENNBIOMK_ROCHE_ELECSYS ABETA42",
            "Units": "pg/mL",
        },
        "csf_tau": {
            "Description": "CSF total tau (Roche Elecsys), within 1 year.",
            "Source": "UPENNBIOMK_ROCHE_ELECSYS TAU",
            "Units": "pg/mL",
        },
        "csf_ptau181": {
            "Description": "CSF pTau181 (Roche Elecsys), within 1 year.",
            "Source": "UPENNBIOMK_ROCHE_ELECSYS PTAU",
            "Units": "pg/mL",
        },
        "plasma_ptau217": {
            "Description": "Plasma pTau217 (Fujirebio), within 1 year.",
            "Source": "UPENN_PLASMA_FUJIREBIO_QUANTERIX pT217_F",
            "Units": "pg/mL",
        },
        "plasma_ab42_ab40": {
            "Description": "Plasma Abeta42/40 ratio (Fujirebio), within 1 year.",
            "Source": "UPENN_PLASMA_FUJIREBIO_QUANTERIX AB42_AB40_F",
        },
        "plasma_nfl": {
            "Description": "Plasma neurofilament light (Quanterix), within 1 year.",
            "Source": "UPENN_PLASMA_FUJIREBIO_QUANTERIX NfL_Q",
            "Units": "pg/mL",
        },
        "plasma_gfap": {
            "Description": "Plasma GFAP (Quanterix), within 1 year.",
            "Source": "UPENN_PLASMA_FUJIREBIO_QUANTERIX GFAP_Q",
            "Units": "pg/mL",
        },
        "wmh_volume": {
            "Description": "White matter hyperintensity volume (UC Davis), within 60 days.",
            "Source": "UCD_WMH TOTAL_WMH",
            "Units": "cm^3",
        },
        **fs_docs,
    }
    s = s[["participant_id", "session_id", "age", "sex", "site", *columns]]
    return s, columns


def main() -> None:
    img = tables.bids_images(SOURCE, ROOT)
    smp, columns = samples()
    selected = pd.read_csv(SOURCE / "selected_images.tsv", sep="\t", na_values="n/a")
    assert len(img) == len(selected), f"{len(img)} images, {len(selected)} selected"
    assert not smp.duplicated(["participant_id", "session_id"]).any()
    assert smp.sex.notna().all() and smp.age.between(40, 100).all()
    assert set(smp.dx.dropna()) <= {"CN", "MCI", "Dementia"}

    y0 = smp[smp.session_id == "y0"].set_index("participant_id")
    complete = y0.dx.notna()
    strata = y0.dx.fillna("n/a") + " " + y0.protocol.fillna("n/a")
    splits = tables.make_splits(strata, complete)
    tables.write(NAME, img, smp, columns, splits, track_samples=False)
    print(tables.summary_table(smp, splits, ["session_id", "dx", "amyloid_status"]))


if __name__ == "__main__":
    main()

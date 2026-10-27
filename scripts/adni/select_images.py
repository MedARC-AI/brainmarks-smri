"""Choose the ADNI images to convert: one T1w and one FLAIR at baseline, ~2 and ~4 years.

    ADNI_ZIPS=<dir of IDA image zips> ADNIMERGE2=<ADNIMERGE2 tarball> \
        uv run python scripts/adni/select_images.py

Writes scripts/adni/selected_images.tsv: the LONI image IDs to download from IDA and convert.
Scan dates are left out (they are DUA data); convert.py reads them from the zips.

- Inputs: the series in the IDA zips (`ADNI/<PTID>/<description>/<datetime>/I<image id>/*.dcm`),
  one DICOM header per T1w series, and the Mayo QC tables in ADNIMERGE2 (MRIQC, MRIMPRANK,
  MAYOADIRL_MRI_QUALITY_ADNI3).
- Visits: T1w scans of a participant within 60 days of each other. Sessions: y0 = the first
  visit with a usable T1w; y2 and y4 = the usable visit nearest 2 and 4 years after y0, within
  6 months.
- T1w per visit, in order: drop distortion-uncorrected copies (`_ND`), Mayo quality 4 (unusable),
  and coverage under 140 mm; prefer 1.5 T (the ADNI1 protocol; all later scans are 3 T); prefer
  Mayo's selected series, then the best MRIMPRANK rank; prefer accelerated (as in ADNI3/4);
  prefer the protocol scan over its "repeat"; then the later series (a rescan); then the lowest
  image ID. Visits with both 1.5 T and 3 T also keep the best 3 T scan (desc `acq-3T`).
- FLAIR: the FLAIR series nearest the visit (within 60 days); prefer Mayo's selected series, drop
  Mayo quality 4, then the later series, then the lowest image ID.
"""

import io
import os
import re
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pyreadr

OUT = Path(__file__).parent / "selected_images.tsv"
SERIES_PATH = re.compile(r"ADNI/(\d{3}_S_\d{4,5})/([^/]+)/([\d-]+)_[\d_.]+/I(\d+)/([^/]+\.dcm)")
T1W = re.compile(r"MP-?RAGE|SPGR", re.IGNORECASE)
FLAIR = re.compile(r"FLAIR", re.IGNORECASE)
UNCORRECTED = re.compile(r"_ND(?:_|$)")
VISIT_GAP_DAYS = 60
SESSION_YEARS = {"y2": 2.0, "y4": 4.0}
SESSION_WINDOW_YEARS = 0.5
MIN_COVERAGE_MM = 140
# Series that fail to convert (dcm2niix finds no consistent slice geometry).
EXCLUDE = {"I820315"}


def list_series(zip_dir: Path) -> pd.DataFrame:
    """One row per MR series in the zips: image ID, participant, description, date, files."""
    files = []
    for zip_path in sorted(zip_dir.glob("*.zip")):
        if not zipfile.is_zipfile(zip_path):
            print(f"skipping {zip_path.name}: not a readable zip (truncated download?)")
            continue
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                match = SERIES_PATH.fullmatch(name)
                if match and "_MR_" in match[5]:
                    ptid, folder, date, image_number, _ = match.groups()
                    files.append((f"I{image_number}", ptid, folder, date, zip_path.name, name))
    files = pd.DataFrame(files, columns=["image_id", "ptid", "folder", "date", "zip", "member"])
    series = files.groupby("image_id").agg(
        ptid=("ptid", "first"),
        folder=("folder", "first"),
        date=("date", "first"),
        zip=("zip", "first"),
        member=("member", "min"),
        n_files=("member", "size"),
        n_zips=("zip", "nunique"),
    )
    assert (series.n_zips == 1).all(), "a series is split across zips"
    series["date"] = pd.to_datetime(series.date)
    series["modality"] = None
    series.loc[series.folder.str.contains(T1W), "modality"] = "T1w"
    series.loc[series.folder.str.contains(FLAIR), "modality"] = "FLAIR"
    return series.dropna(subset=["modality"]).reset_index()


def read_header(archive: zipfile.ZipFile, member: str, n_files: int) -> dict:
    header = pydicom.dcmread(io.BytesIO(archive.read(member)), stop_before_pixels=True)
    # One file per slice, or one multi-frame file (single-slice files may say NumberOfFrames 1).
    n_slices = n_files if n_files > 1 else int(getattr(header, "NumberOfFrames", None) or 1)
    # Enhanced (multi-frame) DICOM keeps the voxel size in the shared or per-frame groups.
    measures = header
    for groups in ["SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"]:
        if groups in header and "PixelMeasuresSequence" in header[groups][0]:
            measures = header[groups][0].PixelMeasuresSequence[0]
    spacing = float(getattr(measures, "SpacingBetweenSlices", None) or measures.SliceThickness)
    row_mm, column_mm = (float(value) for value in measures.PixelSpacing)
    extents = [header.Rows * row_mm, header.Columns * column_mm, n_slices * spacing]
    return {
        "field_strength": float(header.MagneticFieldStrength),
        "series_time": str(getattr(header, "SeriesTime", "") or ""),
        "coverage_mm": min(extents),
    }


def read_headers(series: pd.DataFrame, zip_dir: Path) -> pd.DataFrame:
    """Field strength, series time and smallest extent from one DICOM file per series."""

    def read_zip(group: pd.DataFrame) -> list[dict]:
        with zipfile.ZipFile(zip_dir / group.zip.iloc[0]) as archive:
            rows = []
            for row in group.itertuples():
                rows.append(
                    {"image_id": row.image_id, **read_header(archive, row.member, row.n_files)}
                )
            return rows

    with ThreadPoolExecutor(8) as pool:
        results = pool.map(read_zip, [group for _, group in series.groupby("zip")])
    headers = pd.DataFrame([row for rows in results for row in rows])
    # Some headers give the field in gauss (15000, 30000) or slightly off (1.494, 2.89).
    tesla = headers.field_strength.where(headers.field_strength < 100, headers.field_strength / 1e4)
    headers["field_strength"] = np.where(tesla < 2, 1.5, 3.0)
    return series.merge(headers, on="image_id", how="left")


def read_rda(tarball: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    tables = {}
    with tarfile.open(tarball) as archive, tempfile.TemporaryDirectory() as tmp:
        for name in names:
            member = archive.getmember(f"ADNIMERGE2/data/{name}.rda")
            archive.extract(member, tmp, filter="data")
            tables[name] = next(iter(pyreadr.read_r(Path(tmp) / member.name).values()))
    return tables


def mayo_qc(tarball: Path) -> pd.DataFrame:
    """Per image ID: Mayo's selected flag and quality grade, MRIMPRANK rank, protocol phase."""
    tables = read_rda(tarball, ["MRIQC", "MRIMPRANK", "MAYOADIRL_MRI_QUALITY_ADNI3"])
    mriqc = tables["MRIQC"].dropna(subset=["LONIImage"])
    mriqc = pd.DataFrame(
        {
            "image_id": "I" + mriqc.LONIImage.astype(int).astype(str),
            "series_id": "S" + mriqc.LONISeries.astype(int).astype(str),
            "phase": mriqc.MRIProtocolPhase,
        }
    ).drop_duplicates("image_id")
    rank = tables["MRIMPRANK"][["LONIUID", "RANK"]].dropna()
    rank = rank.rename(columns={"LONIUID": "series_id", "RANK": "mayo_rank"})
    rank = rank.groupby("series_id").mayo_rank.min().reset_index()
    quality = tables["MAYOADIRL_MRI_QUALITY_ADNI3"].dropna(subset=["LONI_IMAGE"])
    quality = pd.DataFrame(
        {
            "image_id": "I" + quality.LONI_IMAGE.astype(int).astype(str),
            "mayo_selected": quality.SERIES_SELECTED.astype(bool),
            "mayo_quality": quality.SERIES_QUALITY,
        }
    ).drop_duplicates("image_id")
    return mriqc.merge(rank, on="series_id", how="left").merge(quality, on="image_id", how="left")


def add_visits(t1w: pd.DataFrame) -> pd.DataFrame:
    """Number the visits of each participant: T1w dates within VISIT_GAP_DAYS of the previous."""
    dates = t1w[["ptid", "date"]].drop_duplicates().sort_values(["ptid", "date"])
    new_visit = dates.groupby("ptid").date.diff().dt.days.fillna(np.inf) > VISIT_GAP_DAYS
    dates["visit"] = new_visit.groupby(dates.ptid).cumsum().astype(int)
    return t1w.merge(dates, on=["ptid", "date"])


def rank_t1w(t1w: pd.DataFrame) -> pd.DataFrame:
    """Sort the usable T1w scans so that the first of each (visit, field strength) is the pick."""
    usable = (
        ~t1w.folder.str.contains(UNCORRECTED)
        & (t1w.mayo_quality != 4)
        & (t1w.coverage_mm >= MIN_COVERAGE_MM)
        & ~t1w.image_id.isin(EXCLUDE)
    )
    t1w = t1w[usable].copy()
    t1w["not_selected"] = t1w.mayo_selected != True  # noqa: E712 (NaN counts as not selected)
    t1w["rank_key"] = t1w.mayo_rank.fillna(99)
    t1w["not_accelerated"] = ~t1w.folder.str.contains("Accel|GRAPPA|SENSE|ASSET", case=False)
    t1w["is_repeat"] = t1w.folder.str.contains("repeat", case=False)
    t1w["later_first"] = -pd.to_numeric(t1w.series_time, errors="coerce")
    t1w["image_number"] = t1w.image_id.str[1:].astype(int)
    order = [
        "not_selected",
        "rank_key",
        "not_accelerated",
        "is_repeat",
        "later_first",
        "image_number",
    ]
    return t1w.sort_values(["ptid", "visit", "field_strength", *order])


def pick_sessions(t1w: pd.DataFrame) -> pd.DataFrame:
    """The visit for each session: y0 = first usable visit, y2/y4 = nearest to 2/4 years."""
    visits = t1w.groupby(["ptid", "visit"]).date.min().reset_index()
    baseline = visits.groupby("ptid").date.transform("min")
    visits["years"] = (visits.date - baseline).dt.days / 365.25
    sessions = [visits[visits.years == 0].assign(session_id="y0")]
    for session_id, target in SESSION_YEARS.items():
        visits["distance"] = (visits.years - target).abs()
        near = visits[visits.distance <= SESSION_WINDOW_YEARS].sort_values("distance")
        sessions.append(near.groupby("ptid").head(1).assign(session_id=session_id))
    return pd.concat(sessions)[["ptid", "visit", "date", "years", "session_id"]]


def pick_t1w(t1w: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    best_per_field = t1w.groupby(["ptid", "visit", "field_strength"]).head(1)
    picks = sessions.merge(best_per_field, on=["ptid", "visit"], suffixes=("_visit", ""))
    # 1.5 T is the main image where present; a 3 T scan at the same visit is kept as acq-3T.
    has_15t = picks.groupby(["ptid", "visit"]).field_strength.transform("min") == 1.5
    picks["desc"] = np.where(has_15t & (picks.field_strength == 3.0), "acq-3T", None)
    return picks


def pick_flair(flair: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    flair = flair[(flair.mayo_quality != 4) & ~flair.image_id.isin(EXCLUDE)].copy()
    candidates = sessions.merge(flair, on="ptid", suffixes=("_visit", ""))
    candidates["gap_days"] = (candidates.date - candidates.date_visit).dt.days.abs()
    candidates = candidates[candidates.gap_days <= VISIT_GAP_DAYS]
    candidates["not_selected"] = candidates.mayo_selected != True  # noqa: E712
    candidates["later_first"] = candidates.image_id.str[1:].astype(int) * -1
    candidates["image_number"] = candidates.image_id.str[1:].astype(int)
    order = ["gap_days", "not_selected", "later_first", "image_number"]
    candidates = candidates.sort_values(["ptid", "session_id", *order])
    return candidates.groupby(["ptid", "session_id"]).head(1).assign(desc=None)


def main() -> None:
    zip_dir = Path(os.environ["ADNI_ZIPS"])
    tarball = Path(os.environ["ADNIMERGE2"])
    series = list_series(zip_dir).merge(mayo_qc(tarball), on="image_id", how="left")
    print(f"{len(series)} T1w/FLAIR series in the zips:", series.modality.value_counts().to_dict())

    t1w = read_headers(series[series.modality == "T1w"], zip_dir)
    t1w = rank_t1w(add_visits(t1w))
    sessions = pick_sessions(t1w)
    picks = pd.concat(
        [pick_t1w(t1w, sessions), pick_flair(series[series.modality == "FLAIR"], sessions)]
    )
    picks["participant_id"] = "sub-" + picks.ptid.str.replace("_", "")
    picks["years"] = picks.years.round(2)
    columns = [
        "participant_id", "session_id", "modality", "desc", "image_id", "ptid", "years",
        "folder", "field_strength", "phase",
    ]  # fmt: skip
    picks = picks[columns].rename(columns={"folder": "series_description"})
    picks = picks.sort_values(
        ["participant_id", "session_id", "modality", "desc"], na_position="first"
    )
    assert not picks.duplicated(["participant_id", "session_id", "modality", "desc"]).any()
    picks.to_csv(OUT, sep="\t", index=False, na_rep="n/a", lineterminator="\n")
    print(picks.groupby(["session_id", "modality"]).size().unstack().to_string())


if __name__ == "__main__":
    main()

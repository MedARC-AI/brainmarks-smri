"""Choose the A4/LEARN T1w images to extract: baseline, week 84 and week 240.

    A4_ZIPS=<dir of IDA zips> uv run python scripts/a4learn/select_images.py

Writes scripts/a4learn/selected_images.tsv: one row per image, named as in the IDA zips
(`ImageData/<A4|LEARN|SF>_MR_T1_<BID>_<VISCODE>.nii.gz`).

- There is one T1w per participant and visit. Sessions: bl = the first usable T1w at visit 004
  (screening; A4 and screen-fails), 006 (most of LEARN) or 009 (week 12); wk84 = visit 027;
  wk240 = visit 066 (end of the double-blind phase). Unusable images (EXCLUDE) are skipped.
"""

import os
import re
import zipfile
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "selected_images.tsv"
T1W = re.compile(r"ImageData/(?:A4|LEARN|SF)_MR_T1_(B\d+)_(\d{3})\.nii\.gz")
SESSIONS = {"027": "wk84", "066": "wk240"}
# Visits that count as baseline: screening (004), LEARN baseline (006), week 12 (009).
BASELINE_VISITS = ["004", "006", "009"]
# <BID>_<VISCODE> of unusable images, as distributed: 35 whose intensity scaling puts the
# background far below zero (about -60,000 to -160,000), and one blank image (B69617870_066).
EXCLUDE = {
    "B11180690_027",
    "B11714080_027",
    "B17183859_066",
    "B19547692_066",
    "B20658632_027",
    "B20685017_066",
    "B20715085_006",
    "B22485102_006",
    "B28645650_006",
    "B29568436_006",
    "B30904648_027",
    "B35900517_006",
    "B36702787_006",
    "B37234681_027",
    "B42157834_006",
    "B46883668_006",
    "B46920242_027",
    "B46920242_066",
    "B49156238_027",
    "B54288722_027",
    "B55630269_066",
    "B57296708_027",
    "B60039657_004",
    "B60039657_027",
    "B63182965_004",
    "B63595805_066",
    "B65723979_066",
    "B68058577_027",
    "B69617870_066",
    "B77596233_006",
    "B80467260_027",
    "B85877142_027",
    "B89365069_006",
    "B92202894_009",
    "B92202894_027",
    "B93549418_006",
}


def main() -> None:
    zip_dir = Path(os.environ["A4_ZIPS"])
    rows = []
    for zip_path in sorted(zip_dir.glob("ImageData - T1 - *.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                match = T1W.fullmatch(name)
                if match:
                    rows.append((match[1], match[2], name))
    images = pd.DataFrame(rows, columns=["bid", "viscode", "file"])
    assert not images.duplicated(["bid", "viscode"]).any()
    excluded = (images.bid + "_" + images.viscode).isin(EXCLUDE)
    assert excluded.sum() == len(EXCLUDE)
    images = images[~excluded]
    print(f"{len(images)} T1w images, {images.bid.nunique()} participants")

    early = images[images.viscode.isin(BASELINE_VISITS)]
    first = early.sort_values("viscode").groupby("bid").head(1).assign(session_id="bl")
    later = images[images.viscode.isin(list(SESSIONS))].assign(
        session_id=lambda x: x.viscode.map(SESSIONS)
    )
    later = later[
        ~later.set_index(["bid", "viscode"]).index.isin(first.set_index(["bid", "viscode"]).index)
    ]
    selected = pd.concat([first, later])
    selected["participant_id"] = "sub-" + selected.bid
    selected = selected[["participant_id", "session_id", "bid", "viscode", "file"]]
    selected = selected.sort_values(["participant_id", "viscode"])
    selected.to_csv(OUT, sep="\t", index=False, lineterminator="\n")
    print(selected.groupby(["session_id", "viscode"]).size().to_string())


if __name__ == "__main__":
    main()

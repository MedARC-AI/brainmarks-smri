"""Choose the A4/LEARN T1w images to extract: baseline, week 84 and week 240.

    A4_ZIPS=<dir of IDA zips> uv run python scripts/a4learn/select_images.py

Writes scripts/a4learn/selected_images.tsv: one row per image, named as in the IDA zips
(`ImageData/<A4|LEARN|SF>_MR_T1_<BID>_<VISCODE>.nii.gz`).

- There is one T1w per participant and visit. Sessions: bl = the first visit with a T1w
  (screening visit 004 for A4 and screen-fails, 006 for most of LEARN); wk84 = visit 027;
  wk240 = visit 066 (end of the double-blind phase).
"""

import os
import re
import zipfile
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "selected_images.tsv"
T1W = re.compile(r"ImageData/(?:A4|LEARN|SF)_MR_T1_(B\d+)_(\d{3})\.nii\.gz")
SESSIONS = {"027": "wk84", "066": "wk240"}
EARLY_TERMINATION = "999"


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
    print(f"{len(images)} T1w images, {images.bid.nunique()} participants")

    scheduled = images[images.viscode != EARLY_TERMINATION]
    first = scheduled.sort_values("viscode").groupby("bid").head(1).assign(session_id="bl")
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

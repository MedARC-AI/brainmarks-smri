"""List the BraTS 2021 Faspex package on TCIA over HTTPS (no Aspera needed for listing).

Prints, per split and source folder, the number of case directories and the file names
in the first case. Used to verify case counts and which splits carry segmentation labels.

    uv run python scripts/brats2021/list_package.py [--all-cases]

--all-cases lists every case directory (~1,500 paced requests) instead of one per folder.
Login and listing use `brainmarks_smri.tcia_faspex` (public-link OAuth + Faspex v5 API).
"""

import sys

from brainmarks_smri import tcia_faspex

# "Challenge data both tasks" public link on the TCIA page (package 636), as in download.sh.
LINK = ("https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context="
        "eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjYzNiIsInBhc3Njb2Rl"
        "IjoiNDM5YTVhZjM3NGRhYjk3OGExYjExMzA4MTcyZDhlMDdkY2Q5OWMzMSIsInBhY2thZ2VfaWQiOiI2MzYiLCJlbWFpbCI6ImhlbHBA"
        "Y2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=")
ROOT = "/RSNA-ASNR-MICCAI-BraTS-2021"
SPLITS = ["BraTS2021_TrainingSet", "BraTS2021_ValidationSet",
          "BraTS2021_TrainingSet_dcm", "BraTS2021_ValidationSet_dcm"]


def main() -> None:
    all_cases = "--all-cases" in sys.argv
    s, package = tcia_faspex.session(LINK)
    info = s.get(f"{tcia_faspex.BASE}/api/v5/packages/{package}", timeout=60).json()
    print(f"# package {package}: {info['title']} | released {info['release_date']} | {info['message'].strip()}")
    print("\t".join(["split", "source", "n_cases", "case", "files"]))
    for split in SPLITS:
        for source in tcia_faspex.ls(s, package, f"{ROOT}/{split}"):
            cases = [c for c in tcia_faspex.ls(s, package, source["path"]) if c["type"] == "directory"]
            for case in (cases if all_cases else cases[:1]):
                files = ",".join(sorted(f["basename"] for f in tcia_faspex.ls(s, package, case["path"])))
                print("\t".join([split, source["basename"], str(len(cases)), case["basename"], files]), flush=True)


if __name__ == "__main__":
    main()

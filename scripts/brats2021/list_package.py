"""List the BraTS 2021 Faspex package on TCIA over HTTPS (no Aspera needed for listing).

Prints, per split and source folder, the number of case directories and the file names
in the first case. Used to verify case counts and which splits carry segmentation labels.

    uv run --with requests python scripts/brats2021/list_package.py [--all-cases]

--all-cases lists every case directory (~1,500 paced requests) instead of one per folder.

How the public link works (Faspex 5): the link on the TCIA page carries a base64 JSON
`context` with the package id and passcode. /auth/authorize_public_link turns it into an
OAuth code, /auth/token into a bearer token, and the v5 API then allows listing.
Downloading still needs FASP (port 33001): the server has no HTTP gateway configured.
"""

import base64
import json
import sys
import time
import urllib.parse

import requests

BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
CLIENT_ID = "ff9aa63a-72e1-436f-82ef-5677eb1f7aee"  # public web-app client id, from /aspera/faspex/config.js
REDIRECT = "/aspera/faspex/token"
# "Challenge data both tasks" public link on the TCIA page (package 636).
CONTEXT = base64.b64encode(json.dumps({
    "resource": "packages", "type": "external_download_package", "id": "636",
    "passcode": "439a5af374dab978a1b11308172d8e07dcd99c31", "package_id": "636",
    "email": "help@cancerimagingarchive.net",
}, separators=(",", ":")).encode()).decode()
ROOT = "/RSNA-ASNR-MICCAI-BraTS-2021"
SPLITS = ["BraTS2021_TrainingSet", "BraTS2021_ValidationSet",
          "BraTS2021_TrainingSet_dcm", "BraTS2021_ValidationSet_dcm"]


def session():
    r = requests.get(f"{BASE}/auth/authorize_public_link", allow_redirects=False, timeout=60,
                     params=dict(response_type="code", client_id=CLIENT_ID, redirect_uri=REDIRECT, state=CONTEXT))
    code = urllib.parse.parse_qs(urllib.parse.urlparse(r.headers["location"]).query)["code"][0]
    t = requests.post(f"{BASE}/auth/token", timeout=60, json=dict(
        code=code, state=CONTEXT, grant_type="authorization_code", client_id=CLIENT_ID, redirect_uri=REDIRECT))
    t.raise_for_status()
    s = requests.Session()
    s.headers["Authorization"] = "Bearer " + t.json()["access_token"]
    return s


def ls(s, path):
    """All entries of a package directory (the API pages by limit/offset, max 100)."""
    out = []
    while True:
        r = s.post(f"{BASE}/api/v5/packages/636/files/received", json={"path": path},
                   params={"limit": 100, "offset": len(out)}, timeout=120)
        r.raise_for_status()
        d = r.json()
        out += d["items"]
        time.sleep(0.5)  # be polite
        if len(out) >= d["total_count"] or not d["items"]:
            return out


def main():
    all_cases = "--all-cases" in sys.argv
    s = session()
    pkg = s.get(f"{BASE}/api/v5/packages/636", timeout=60).json()
    print(f"# package 636: {pkg['title']} | released {pkg['release_date']} | {pkg['message'].strip()}")
    print("\t".join(["split", "source", "n_cases", "case", "files"]))
    for split in SPLITS:
        for src in ls(s, f"{ROOT}/{split}"):
            cases = [c for c in ls(s, src["path"]) if c["type"] == "directory"]
            for c in (cases if all_cases else cases[:1]):
                files = ",".join(sorted(i["basename"] for i in ls(s, c["path"])))
                print("\t".join([split, src["basename"], str(len(cases)), c["basename"], files]), flush=True)


if __name__ == "__main__":
    main()

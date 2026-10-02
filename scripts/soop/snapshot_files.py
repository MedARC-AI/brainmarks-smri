"""List or verify the files of an OpenNeuro snapshot, via the OpenNeuro GraphQL API.

    snapshot_files.py curl-config <dataset> <tag> <dir>   # curl -K config for files missing/incomplete in <dir>
    snapshot_files.py verify <dataset> <tag> <dir> <manifest.sha256>

Why not openneuro-py: for ds004889 OpenNeuro hands out S3 URLs pre-signed for GET only, and
openneuro-py 2026.9.1 sends a HEAD first, which S3 rejects with 403. The same object
version is public, so we fetch `...?versionId=<v>` without the signature.

verify checks every file against the snapshot's own checksums: the SHA256 in the git-annex
key (`SHA256E-s<size>--<sha256>`) for annexed files, the git blob SHA1 for the rest. It
also flags missing and extra files.
"""

import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

QUERY = (
    '{ snapshot(datasetId: "%s", tag: "%s") '
    "{ files(recursive: true) { id filename size urls annexed directory } } }"
)


def snapshot_files(dataset, tag):
    req = urllib.request.Request(
        "https://openneuro.org/crn/graphql",
        data=json.dumps({"query": QUERY % (dataset, tag)}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        files = json.load(r)["data"]["snapshot"]["files"]
    return [f for f in files if not f["directory"]]


def unsigned_url(url):
    u = urllib.parse.urlsplit(url)
    version = urllib.parse.parse_qs(u.query)["versionId"][0]
    return urllib.parse.urlunsplit(u._replace(query="versionId=" + version))


def curl_config(files, root):
    for f in files:
        out = root / f["filename"]
        if out.exists() and out.stat().st_size == f["size"]:
            continue
        print(f'url = "{unsigned_url(f["urls"][0])}"\noutput = "{out}"')


def verify(files, root, manifest):
    sha256 = {}
    for line in Path(manifest).read_text().splitlines():
        digest, path = line.split("  ", 1)
        sha256[path.removeprefix("source/")] = digest
    bad = 0
    for f in files:
        name, fid = f["filename"], f["id"]
        if name not in sha256:
            print("missing:", name)
            bad += 1
            continue
        if fid.startswith("SHA256E-"):
            ok = sha256.pop(name) == fid.split("--", 1)[1].split(".", 1)[0]
        else:
            sha256.pop(name)
            data = (root / name).read_bytes()
            ok = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == fid
        if not ok:
            print("checksum mismatch:", name)
            bad += 1
    for name in sha256:
        print("extra:", name)
        bad += 1
    print(f"verify: {len(files)} files in snapshot, {bad} problems", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    cmd, dataset, tag, root = sys.argv[1:5]
    files = snapshot_files(dataset, tag)
    if cmd == "curl-config":
        curl_config(files, Path(root))
    else:
        sys.exit(verify(files, Path(root), sys.argv[5]))

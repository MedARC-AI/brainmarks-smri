"""List and download files from a TCIA Aspera Faspex package via its public link.

TCIA distributes its NIfTI and analysis-result releases as Faspex 5 packages. Each
collection page links a public download URL of the form
    https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=<base64 json>
No account is needed: the link's `context` is exchanged for an OAuth bearer token, the
v5 API lists the package and issues a transfer spec (with a FASP token) for chosen paths,
and `ascp` (installed by `ascli config ascp install`) downloads them over FASP (port 33001).

    uv run python -m brainmarks_smri.tcia_faspex ls  <public-link> [<path>]
    uv run python -m brainmarks_smri.tcia_faspex get <public-link> <dest-dir> <path>... [--exclude=<glob>]...

`ls` prints `type<TAB>path` for one directory of the package (default: the root).
`get` downloads each package path (file or directory, recursively) into <dest-dir>/<basename>,
skipping files whose name matches an --exclude glob (`ascp -E`). Files already present with
the right size are skipped (`ascp -k 1`; ascp writes `<file>.partial` until a file is complete).
TCIA's server sometimes stalls or drops a session, so ascp is killed after STALL seconds
without progress and restarted (it resumes), up to ATTEMPTS times. Progress is new bytes on
disk or new lines in ascp's log (skipping complete files writes no data but logs each file).
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

import requests

BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
CLIENT_ID = "ff9aa63a-72e1-436f-82ef-5677eb1f7aee"  # public web-app client id, from /aspera/faspex/config.js
REDIRECT = "/aspera/faspex/token"
SDK = Path.home() / ".aspera" / "sdk"
STALL = 120  # s without progress before ascp is restarted
POLL = 10  # s between progress checks
ATTEMPTS = 20


def query_param(url: str, name: str) -> str:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)[name][0]


def session(link: str) -> tuple[requests.Session, str]:
    """Bearer-token session and package id for a public link."""
    context = query_param(link, "context")
    package = json.loads(base64.b64decode(context + "=="))["package_id"]
    oauth = dict(client_id=CLIENT_ID, redirect_uri=REDIRECT, state=context)

    authorize = requests.get(
        f"{BASE}/auth/authorize_public_link",
        allow_redirects=False,
        timeout=60,
        params=dict(response_type="code", **oauth),
    )
    code = query_param(authorize.headers["location"], "code")
    token = requests.post(
        f"{BASE}/auth/token",
        timeout=60,
        json=dict(code=code, grant_type="authorization_code", **oauth),
    )
    token.raise_for_status()

    s = requests.Session()
    s.headers["Authorization"] = "Bearer " + token.json()["access_token"]
    return s, package


def ls(s: requests.Session, package: str, path: str) -> list[dict]:
    """All entries of a package directory (the API pages by limit/offset, max 100)."""
    entries: list[dict] = []
    while True:
        r = s.post(
            f"{BASE}/api/v5/packages/{package}/files/received",
            json={"path": path},
            params={"limit": 100, "offset": len(entries)},
            timeout=120,
        )
        r.raise_for_status()
        page = r.json()
        entries += page["items"]
        time.sleep(0.5)  # be polite
        if len(entries) >= page["total_count"] or not page["items"]:
            return entries


def disk_bytes(folder: Path) -> int:
    """Total size of the files under `folder`."""
    total = 0
    for path in folder.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:  # ascp's .partial/.aspera-ckpt files come and go
            pass
    return total


def transfer_spec(s: requests.Session, package: str, paths: list[str]) -> dict:
    """FASP transfer spec (host, token, source paths) for downloading `paths`."""
    # The body must be {"paths": [{"path": ...}]}; other shapes (incl. what ascli 4.27 sends) give a 500.
    r = s.post(
        f"{BASE}/api/v5/packages/{package}/transfer_spec/download",
        timeout=60,
        params={"transfer_type": "connect", "type": "received"},
        json={"paths": [{"path": p} for p in paths]},
    )
    r.raise_for_status()
    return r.json()


def run_ascp(spec: dict, dest: Path, exclude: list[str]) -> int:
    """Run ascp for one transfer spec; kill it if it makes no progress for STALL seconds."""
    env = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
    exclude_args = [arg for glob in exclude for arg in ("-E", glob)]
    sources = [p["source"] for p in spec["paths"]]
    with tempfile.TemporaryDirectory() as log_dir:
        log = Path(log_dir)
        cmd = [
            str(SDK / "ascp"),
            "-L",
            str(log),
            "-i",
            str(SDK / "aspera_bypass_rsa.pem"),
            "--mode",
            "recv",
            "--host",
            spec["remote_host"],
            "--user",
            spec["remote_user"],
            "-P",
            str(spec["ssh_port"]),
            "-O",
            str(spec["fasp_port"]),
            "-l",
            "1g",
            "-k",
            "1",
            *exclude_args,
            *sources,
            str(dest),
        ]
        proc = subprocess.Popen(cmd, env=env)
        last, idle = -1, 0
        while proc.poll() is None:
            time.sleep(POLL)
            progress = disk_bytes(dest) + disk_bytes(log)
            idle = idle + POLL if progress == last else 0
            last = progress
            if idle >= STALL:
                print(
                    f"tcia_faspex: no progress for {STALL} s, restarting ascp",
                    file=sys.stderr,
                    flush=True,
                )
                proc.kill()
        return proc.wait()


def get(link: str, dest: Path, paths: list[str], exclude: list[str] | None = None) -> int:
    """Download package `paths` into `dest`, retrying (with a fresh token) until ascp succeeds."""
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        s, package = session(link)
        if run_ascp(transfer_spec(s, package, paths), dest, exclude or []) == 0:
            return 0
        print(f"tcia_faspex: attempt {attempt}/{ATTEMPTS} failed", file=sys.stderr, flush=True)
    return 1


def main() -> None:
    command, link, *args = sys.argv[1:]
    if command == "ls":
        s, package = session(link)
        for entry in ls(s, package, args[0] if args else "/"):
            print(f"{entry['type']}\t{entry['path']}")
    elif command == "get":
        exclude = [a.removeprefix("--exclude=") for a in args if a.startswith("--exclude=")]
        dest, *paths = [a for a in args if not a.startswith("--exclude=")]
        sys.exit(get(link, Path(dest), paths, exclude))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

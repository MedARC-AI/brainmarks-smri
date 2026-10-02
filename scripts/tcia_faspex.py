"""List and download files from a TCIA Aspera Faspex package via its public link.

TCIA distributes its NIfTI and analysis-result releases as Faspex 5 packages. Each
collection page links a public download URL of the form
    https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=<base64 json>
No account is needed: the link's `context` is exchanged for an OAuth bearer token, the
v5 API lists the package and issues a transfer spec (with a FASP token) for chosen paths,
and `ascp` (installed by `ascli config ascp install`) downloads them over FASP (port 33001).

    uv run --with requests python scripts/tcia_faspex.py ls  <public-link> [<path>]
    uv run --with requests python scripts/tcia_faspex.py get <public-link> <dest-dir> <path>... [--exclude=<glob>]...

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

import requests

BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
CLIENT_ID = "ff9aa63a-72e1-436f-82ef-5677eb1f7aee"  # public web-app client id, from /aspera/faspex/config.js
REDIRECT = "/aspera/faspex/token"
SDK = os.path.expanduser("~/.aspera/sdk")
STALL = 120  # s without progress before ascp is restarted
ATTEMPTS = 20


def session(link):
    """Bearer-token session and package id for a public link."""
    context = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)["context"][0]
    pkg = json.loads(base64.b64decode(context + "=="))["package_id"]
    r = requests.get(f"{BASE}/auth/authorize_public_link", allow_redirects=False, timeout=60,
                     params=dict(response_type="code", client_id=CLIENT_ID, redirect_uri=REDIRECT, state=context))
    code = urllib.parse.parse_qs(urllib.parse.urlparse(r.headers["location"]).query)["code"][0]
    t = requests.post(f"{BASE}/auth/token", timeout=60, json=dict(
        code=code, state=context, grant_type="authorization_code", client_id=CLIENT_ID, redirect_uri=REDIRECT))
    t.raise_for_status()
    s = requests.Session()
    s.headers["Authorization"] = "Bearer " + t.json()["access_token"]
    return s, pkg


def ls(s, pkg, path):
    """All entries of a package directory (the API pages by limit/offset, max 100)."""
    out = []
    while True:
        r = s.post(f"{BASE}/api/v5/packages/{pkg}/files/received", json={"path": path},
                   params={"limit": 100, "offset": len(out)}, timeout=120)
        r.raise_for_status()
        d = r.json()
        out += d["items"]
        time.sleep(0.5)  # be polite
        if len(out) >= d["total_count"] or not d["items"]:
            return out


def disk_bytes(dest):
    total = 0
    for d, _, fs in os.walk(dest):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except FileNotFoundError:  # ascp's .partial/.aspera-ckpt files come and go
                pass
    return total


def get(link, dest, paths, exclude=()):
    os.makedirs(dest, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        s, pkg = session(link)  # fresh token per attempt
        # The body must be {"paths": [{"path": ...}]}; other shapes (incl. what ascli 4.27 sends) give a 500.
        r = s.post(f"{BASE}/api/v5/packages/{pkg}/transfer_spec/download", timeout=60,
                   params={"transfer_type": "connect", "type": "received"},
                   json={"paths": [{"path": p} for p in paths]})
        r.raise_for_status()
        ts = r.json()
        env = dict(os.environ, ASPERA_SCP_TOKEN=ts["token"], ASPERA_SCP_COOKIE=ts.get("cookie", ""))
        cmd = [f"{SDK}/ascp", "-i", f"{SDK}/aspera_bypass_rsa.pem", "--mode", "recv",
               "--host", ts["remote_host"], "--user", ts["remote_user"],
               "-P", str(ts["ssh_port"]), "-O", str(ts["fasp_port"]), "-l", "1g", "-k", "1",
               *[a for g in exclude for a in ("-E", g)], *[p["source"] for p in ts["paths"]], dest]
        with tempfile.TemporaryDirectory() as log:
            proc, last, idle = subprocess.Popen(cmd[:1] + ["-L", log] + cmd[1:], env=env), -1, 0
            while proc.poll() is None:
                time.sleep(10)
                now = disk_bytes(dest) + disk_bytes(log)
                idle, last = (idle + 10 if now == last else 0), now
                if idle >= STALL:
                    print(f"tcia_faspex: no progress for {STALL} s, restarting ascp", file=sys.stderr, flush=True)
                    proc.kill()
            if proc.wait() == 0:
                return 0
        print(f"tcia_faspex: attempt {attempt}/{ATTEMPTS} failed", file=sys.stderr, flush=True)
    return 1


def main():
    cmd, link, *args = sys.argv[1:]
    if cmd == "ls":
        s, pkg = session(link)
        for i in ls(s, pkg, args[0] if args else "/"):
            print(f"{i['type']}\t{i['path']}")
    elif cmd == "get":
        exclude = [a.removeprefix("--exclude=") for a in args if a.startswith("--exclude=")]
        dest, *paths = [a for a in args if not a.startswith("--exclude=")]
        sys.exit(get(link, dest, paths, exclude))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

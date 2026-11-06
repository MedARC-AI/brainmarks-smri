import importlib.metadata
import json
import logging
import random
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import torch


def random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_info(path: Path) -> dict[str, Any] | None:
    """Commit, branch and dirty state of the git repo containing `path`; None outside a repo."""
    kwargs = dict(cwd=path, capture_output=True, text=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs)
    if sha.returncode != 0:
        return None
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], **kwargs).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "-uno"], **kwargs).stdout.strip()
    return {"sha": sha.stdout.strip(), "branch": branch, "dirty": bool(status)}


def package_info(distribution_name: str) -> dict[str, Any]:
    """Installed package version, and its commit when installed from git or editable."""
    distribution = importlib.metadata.distribution(distribution_name)
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    git = None
    if "vcs_info" in direct_url:
        git = {"sha": direct_url["vcs_info"]["commit_id"]}
    elif direct_url.get("dir_info", {}).get("editable"):
        git = git_info(Path(urllib.parse.urlparse(direct_url["url"]).path))
    return {"version": distribution.version, "git": git}


def setup_logging(logger: logging.Logger, log_path: Path | None = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    level = logging.INFO
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path))
    logger.setLevel(level)
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.propagate = False

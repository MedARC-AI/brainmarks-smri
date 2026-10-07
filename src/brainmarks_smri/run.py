"""Run one model on one benchmark task; writes `<output_dir>/<model>/<task>.json`.

python -m brainmarks_smri neurojepa abide_diagnosis --overrides max_per_split=50
"""

import argparse
import datetime
import json
import logging
import time
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from brainmarks_smri import probes
from brainmarks_smri.misc import git_info, random_seed, setup_logging
from brainmarks_smri.models import create_model
from brainmarks_smri.tasks import create_task, list_tasks

DEFAULT_CONFIG = resources.files("brainmarks_smri") / "config" / "default.yaml"
PROBES = {
    "classification": probes.probe_classification,
    "regression": probes.probe_regression,
    "segmentation": probes.probe_segmentation,
}

logger = logging.getLogger("brainmarks_smri")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model")
    parser.add_argument("task", choices=list_tasks())
    parser.add_argument("--config", type=Path, help="yaml merged over the default config")
    parser.add_argument("--overrides", nargs="*", help="config overrides, e.g. max_per_split=50")
    args = parser.parse_args()

    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    setup_logging(logger)

    path = Path(cfg.output_dir) / args.model / f"{args.task}.json"
    if path.exists() and not cfg.overwrite:
        logger.info(f"{path} exists, skipping")
        return
    logger.info(f"evaluating {args.model} on {args.task}")
    logger.info(f"start: {datetime.datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"cwd: {Path.cwd()}")
    logger.info(f"git: {git_info()}")
    logger.info(f"cwd git: {git_info(Path.cwd())}")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg)}")

    random_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, **cfg.model_kwargs).to(device).eval()
    task = create_task(args.task, max_per_split=cfg.max_per_split)
    task.dataset.transform = getattr(model, "transform", None)

    start = time.perf_counter()
    result = PROBES[task.type](model, task, cfg.batch_size, cfg.num_workers)
    elapsed = time.perf_counter() - start
    result = {
        "model": args.model,
        "task": args.task,
        "type": task.type,
        "dataset": task.dataset.name,
        "config": OmegaConf.to_container(cfg),
        "git": git_info(),
        # the calling repo, e.g. a model's research code
        "cwd": str(Path.cwd()),
        "cwd_git": git_info(Path.cwd()),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "seconds": elapsed,
        **result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    rounded_metrics = {
        name: np.round(value, 4).tolist() for name, value in result["metrics"].items()
    }
    logger.info(f"done {args.model} {args.task} in {elapsed:.0f}s\n{json.dumps(rounded_metrics)}")

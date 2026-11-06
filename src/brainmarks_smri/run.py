"""Run one model on benchmark tasks; writes `<output_dir>/<name>/<task>.json`.

python -m brainmarks_smri neurojepa abide_diagnosis ixi_age --overrides num_workers=8
python -m brainmarks_smri neurojepa core

Saves the model's task and group scores (`core` once all core results exist) to
`<output_dir>/<name>/aggregate.json`.
"""

import argparse
import datetime
import importlib
import json
import logging
import os
import sys
import time
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from brainmarks_smri import aggregate, probes
from brainmarks_smri.utils.misc import git_info, package_info, random_seed, setup_logging
from brainmarks_smri.models import create_model
from brainmarks_smri.models.base import Model
from brainmarks_smri.tasks import TASK_GROUPS, create_task, expand_tasks, list_tasks

DEFAULT_CONFIG = resources.files("brainmarks_smri") / "config" / "default.yaml"
PROBES = {
    "classification": probes.probe_classification,
    "regression": probes.probe_regression,
    "segmentation": probes.probe_segmentation,
}
SEED = 0

logger = logging.getLogger("brainmarks_smri")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model")
    parser.add_argument(
        "tasks", nargs="+", choices=list_tasks() + list(TASK_GROUPS), metavar="task"
    )
    parser.add_argument("--config", type=Path, help="yaml merged over the default config")
    parser.add_argument("--overrides", nargs="*", help="config overrides, e.g. num_workers=8")
    args = parser.parse_args()

    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    # the root logger, replacing handlers that model packages install on import (e.g. neurojepa)
    setup_logging(logging.getLogger())

    name = cfg.name or args.model
    subset = "test" if os.environ.get("BRAINMARKS_SMRI_TEST_SUBSET") == "1" else "dev"
    tasks = []
    for task_name in expand_tasks(args.tasks):
        path = Path(cfg.output_dir) / name / f"{task_name}.json"
        if path.exists() and not cfg.overwrite:
            logger.info(f"{path} exists, skipping")
        else:
            tasks.append(task_name)
    if not tasks:
        save_aggregate(Path(cfg.output_dir) / name)
        return
    logger.info(f"evaluating {args.model} on {', '.join(tasks)} ({subset})")
    logger.info(f"start: {datetime.datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"version: {package_info('brainmarks-smri')}")
    logger.info(f"cwd: {Path.cwd()}")
    logger.info(f"cwd git: {git_info(Path.cwd())}")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg)}")

    # external models register themselves on import
    for module in cfg.imports:
        importlib.import_module(module)

    random_seed(SEED)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, **cfg.model_kwargs).to(device).eval()

    failed = []
    for task_name in tasks:
        try:
            run_task(model, args.model, name, task_name, subset, cfg)
        except Exception:
            logger.exception(f"{name} {task_name} failed")
            failed.append(task_name)
    save_aggregate(Path(cfg.output_dir) / name)
    if failed:
        logger.error(f"failed: {', '.join(failed)}")
        sys.exit(1)


def save_aggregate(folder: Path) -> None:
    """Task and group scores with bootstrap resamples, from all results in `folder`, in
    `<folder>/aggregate.json`."""
    paths = [path for path in sorted(folder.glob("*.json")) if path.name != "aggregate.json"]
    if not paths:
        return
    results = [json.loads(path.read_text()) for path in paths]
    summary, boot = aggregate.aggregate_scores(results)
    scores = {"summary": summary.to_dict(orient="index"), "bootstrap": boot.to_dict(orient="list")}
    (folder / "aggregate.json").write_text(json.dumps(scores, indent=1))
    logger.info(f"{folder.name} scores\n{summary.round(3).to_markdown()}")
    # last, so they are easy to find
    groups = [group for group in ["core", "open"] if group in summary.index] or ["all"]
    for group in groups:
        row = summary.loc[group]
        logger.info(f"{folder.name} {group} score: {row.score:.3f} [{row.low:.3f}, {row.high:.3f}]")


def run_task(
    model: Model,
    model_name: str,
    name: str,
    task_name: str,
    subset: str,
    cfg: DictConfig,
) -> None:
    logger.info(f"evaluating {model_name} on {task_name}")
    random_seed(SEED)
    task = create_task(task_name, subset=subset)
    model_transform = getattr(model, "transform", None)
    if task.transform is None:
        task.dataset.transform = model_transform
    else:
        task.dataset.transform = lambda sample: model_transform(task.transform(sample))

    start = time.perf_counter()
    result = PROBES[task.type](model, task, cfg.batch_size, cfg.num_workers)
    elapsed = time.perf_counter() - start
    result = {
        "model": model_name,
        "name": name,
        "task": task_name,
        "subset": subset,
        "type": task.type,
        "target": task.target,
        "dataset": task.dataset.name,
        "config": OmegaConf.to_container(cfg),
        "version": package_info("brainmarks-smri"),
        # the calling repo, e.g. a model's research code
        "cwd": str(Path.cwd()),
        "cwd_git": git_info(Path.cwd()),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "seconds": elapsed,
        **result,
    }
    path = Path(cfg.output_dir) / name / f"{task_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    rounded_metrics = {
        metric: np.round(value, 4).tolist() for metric, value in result["metrics"].items()
    }
    logger.info(f"done {name} {task_name} in {elapsed:.0f}s\n{json.dumps(rounded_metrics)}")

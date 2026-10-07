"""Run one model on a list of benchmark tasks; one JSON per task in `<output>/<model>/<task>.json`.
Tasks with an existing result are skipped.

    python -m brainmarks_smri.run --model neurojepa --tasks abide_diagnosis ixi_age --max-per-split 50
"""

import argparse
import datetime
import json
import logging
import subprocess
import time
from pathlib import Path

import torch

from brainmarks_smri import probes
from brainmarks_smri.models import create_model
from brainmarks_smri.tasks import TASKS

PROBES = {
    "classification": probes.probe_classification,
    "regression": probes.probe_regression,
    "segmentation": probes.probe_segmentation,
}

logger = logging.getLogger("brainmarks_smri.run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-kwargs", default="{}", help="JSON kwargs for create_model")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--max-per-split", type=int, default=None, help="mini-splits")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_kwargs = json.loads(args.model_kwargs)
    model = create_model(args.model, **model_kwargs).to(args.device).eval()
    git_sha = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
    output_dir = args.output / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.tasks:
        path = output_dir / f"{name}.json"
        if path.exists() and not args.overwrite:
            logger.info(f"{name}: result exists, skipping")
            continue
        logger.info(f"{name}: running")
        start = time.perf_counter()
        # one failed task shouldn't lose the others
        try:
            task = TASKS[name](eval_split=args.eval_split, max_per_split=args.max_per_split)
            task.dataset.transform = getattr(model, "transform", None)
            result = PROBES[task.type](model, task, args.batch_size, args.num_workers)
        except Exception:
            logger.exception(f"{name}: failed")
            continue
        result = {
            "task": name,
            "type": task.type,
            "dataset": task.dataset.name,
            "model": args.model,
            "model_kwargs": model_kwargs,
            "eval_split": args.eval_split,
            "max_per_split": args.max_per_split,
            "git_sha": git_sha,
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "seconds": time.perf_counter() - start,
            **result,
        }
        path.write_text(json.dumps(result, indent=1))
        logger.info(f"{name}: {result['metrics']}")


if __name__ == "__main__":
    main()

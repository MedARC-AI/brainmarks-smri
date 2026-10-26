"""Build Harbor tasks into $BRAINMARKS_SMRI_DATA/harbor/<task>/.

python harbor/build_tasks.py ixi_age --max-train 300 --max-test 100
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from brainmarks_smri.datasets import DATA_ROOT, BrainDataset
from brainmarks_smri.tasks import create_task

HARBOR_DIR = Path(__file__).resolve().parent
SEED = 0


def write_image(dataset: BrainDataset, index: int, path: Path) -> None:
    """Skull-stripped image on the original grid, int16 on disk, with a fresh header."""
    image_input, _ = dataset[index]
    image = image_input["image"]
    brain = np.asanyarray(image_input["brain_mask"].dataobj) > 0
    nib.save(nib.Nifti1Image(image.get_fdata() * brain, image.affine, dtype=np.int16), path)


def build_global(name: str, out_dir: Path, max_train: int, max_test: int, workers: int) -> None:
    """Classification and regression tasks, with one label per subject."""
    task = create_task(name, max_per_split={"train": max_train, "val": max_test})
    assert task.type in SCORERS, f"{name}: no Harbor scorer for {task.type}"
    # Classification labels are 0/1, with 1 for the positive class (the last of task.classes).
    assert task.type != "classification" or len(task.classes) == 2, f"{name}: not binary"
    samples = task.dataset.samples

    rng = np.random.default_rng(SEED)
    ids = rng.permutation(np.concatenate([task.train_ids, task.eval_ids]))
    subject_of_id = {index: f"sub-{number:04d}" for number, index in enumerate(ids, start=1)}

    shutil.copytree(HARBOR_DIR / "task-template", out_dir)
    jobs = []
    for split, split_ids in [("train", task.train_ids), ("test", task.eval_ids)]:
        images_dir = out_dir / "environment" / "data" / split / "images"
        images_dir.mkdir(parents=True)
        for index in split_ids:
            jobs.append((index, images_dir / f"{subject_of_id[index]}.nii.gz"))

        labels = samples.loc[split_ids, [task.target]]
        if task.type == "classification":
            labels[task.target] = (labels[task.target] == task.classes[-1]).astype(int)
        labels.insert(0, "subject", [subject_of_id[index] for index in split_ids])
        labels = labels.sort_values("subject")
        if split == "train":
            labels.to_csv(out_dir / "environment" / "data" / "train" / "labels.csv", index=False)
            labels.to_csv(out_dir / "tests" / "train_labels.csv", index=False)
        else:
            labels.to_csv(out_dir / "tests" / "labels.csv", index=False)
            (out_dir / "solution" / "output").mkdir()
            labels.to_csv(out_dir / "solution" / "output" / "predictions.csv", index=False)

    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(write_image, task.dataset, index, path) for index, path in jobs]
        for future in futures:
            future.result()

    shutil.copy(HARBOR_DIR / "score" / SCORERS[task.type], out_dir / "tests" / "score.py")
    instruction = (HARBOR_DIR / name / "instruction.md").read_text()
    (out_dir / "instruction.md").write_text(instruction)
    title = instruction.splitlines()[0].removeprefix("# ")
    config = (HARBOR_DIR / "task-template" / "task.toml").read_text()
    (out_dir / "task.toml").write_text(config.format(name=name, description=title))


SCORERS = {"classification": "classification.py", "regression": "regression.py"}
# Tasks with a Harbor instruction.
TASKS = sorted(path.parent.name for path in HARBOR_DIR.glob("*/instruction.md"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--max-train", type=int, default=300)
    parser.add_argument("--max-test", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-root", type=Path, default=Path(DATA_ROOT) / "harbor")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_root / args.task
    if out_dir.exists():
        assert args.overwrite, f"{out_dir} exists, pass --overwrite"
        shutil.rmtree(out_dir)
    build_global(args.task, out_dir, args.max_train, args.max_test, args.workers)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

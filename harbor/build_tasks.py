"""Build Harbor tasks into $BRAINMARKS_SMRI_DATA/harbor/<task>/.

python harbor/build_tasks.py ixi_age --max-train 300 --max-test 100

The sandbox test split is the benchmark val split. Build and run the tasks on a host with Docker:
harbor run -p <task dir> -a oracle
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
    """Skull-stripped int16 image on the original grid, with a fresh header."""
    image_input, _ = dataset[index]
    image = image_input["image"]
    brain = np.asanyarray(image_input["brain_mask"].dataobj) > 0
    data = np.round(image.get_fdata() * brain)
    assert np.abs(data).max() <= np.iinfo(np.int16).max, f"{path}: out of int16 range"
    nib.save(nib.Nifti1Image(data.astype(np.int16), image.affine), path)


def build_ixi_age(out_dir: Path, max_train: int, max_test: int, workers: int) -> None:
    task = create_task("ixi_age", max_per_split={"train": max_train, "val": max_test})
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

        labels = samples.loc[split_ids, ["age"]].round(2)
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

    shutil.copy(HARBOR_DIR / "ixi_age" / "score.py", out_dir / "tests" / "score.py")
    instruction = (HARBOR_DIR / "ixi_age" / "instruction.md").read_text()
    instruction = instruction.format(n_train=len(task.train_ids), n_test=len(task.eval_ids))
    (out_dir / "instruction.md").write_text(instruction)
    config = (HARBOR_DIR / "task-template" / "task.toml").read_text()
    config = config.format(name="ixi_age", description="Brain age from T1-weighted MRI (IXI).")
    (out_dir / "task.toml").write_text(config)


BUILDERS = {"ixi_age": build_ixi_age}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", choices=sorted(BUILDERS))
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
    BUILDERS[args.task](out_dir, args.max_train, args.max_test, args.workers)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

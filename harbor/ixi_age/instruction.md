# Brain age from T1-weighted MRI

Predict the age in years of each test subject from their brain MRI.

## Data

- `/data/train/images/<subject>.nii.gz`: T1-weighted MRI of training subjects.
- `/data/train/labels.csv`: columns `subject` and `age`.
- `/data/test/images/<subject>.nii.gz`: T1-weighted MRI of test subjects.

## Output

Write `/output/predictions.csv` with columns `subject` and `age`, one row per test subject.

Predictions are scored by R² against the true ages. A missing or invalid prediction counts as the mean training age.

## Setup

You have 1 hour, 4 CPUs, 16 GB of RAM, no GPU and no internet access. Python 3.12 has numpy, scipy, pandas, scikit-learn, scikit-image, nibabel and matplotlib installed.

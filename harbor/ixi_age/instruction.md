# Brain age from T1-weighted MRI

Predict the age in years of each test subject from their brain MRI.

## Data

- `/data/train/images/<subject>.nii.gz`: T1-weighted MRI of {n_train} training subjects.
- `/data/train/labels.csv`: columns `subject` and `age`.
- `/data/test/images/<subject>.nii.gz`: T1-weighted MRI of {n_test} test subjects.

The subjects are healthy adults. The images are in scanner space, not registered to a template. They are skull-stripped (voxels outside the brain are 0) and stored as int16.

## Output

Write `/output/predictions.csv` with columns `subject` and `age`, one row per test subject.

Predictions are scored by R² against the true ages. A missing or invalid prediction counts as the mean training age.

## Setup

You have 1 hour, 4 CPUs, 16 GB of RAM, no GPU and no internet access. Python 3.12 has numpy, scipy, pandas, scikit-learn, scikit-image, nibabel, nilearn and matplotlib installed. You can look at the images yourself, for example by saving slices as PNG files and viewing them, as well as analyze them with code.

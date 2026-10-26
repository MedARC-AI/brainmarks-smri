# Dementia from T1-weighted MRI

Predict whether each test subject has dementia from their brain MRI.

## Data

- `/data/train/images/<subject>.nii.gz`: T1-weighted MRI of training subjects.
- `/data/train/labels.csv`: columns `subject` and `dementia`. Dementia is 1 for a Clinical Dementia Rating above 0 (very mild dementia or worse), else 0.
- `/data/test/images/<subject>.nii.gz`: T1-weighted MRI of test subjects.

## Output

Write `/output/predictions.csv` with columns `subject` and `dementia`, one row per test subject. Dementia is the probability of dementia, from 0 to 1.

Predictions are scored by AUROC. A missing or invalid prediction counts as the fraction of training subjects with dementia.

## Setup

You have 1 hour, 4 CPUs, 16 GB of RAM, no GPU and no internet access. Python 3.12 has numpy, scipy, pandas, scikit-learn, scikit-image, nibabel and matplotlib installed.

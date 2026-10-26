# Brain tumor segmentation from FLAIR MRI

Segment the whole tumor in each test subject's brain MRI.

## Data

- `/data/train/images/<subject>.nii.gz`: FLAIR MRI of training subjects with glioma.
- `/data/train/labels/<subject>.nii.gz`: whole-tumor masks of the training subjects, 1 inside the tumor and 0 outside. The whole tumor is the necrotic core, the enhancing tumor and the surrounding edema.
- `/data/test/images/<subject>.nii.gz`: FLAIR MRI of test subjects.

## Output

Write `/output/predictions/<subject>.nii.gz` for each test subject: a mask with 1 inside the tumor and 0 outside, with the same shape as the subject's image.

Predictions are scored by Dice per subject, averaged over test subjects. A missing or invalid mask counts as empty.

For reference, a linear probe on features from a pretrained brain MRI foundation model (not available here) reaches a mean Dice of 0.90 on these test subjects. Try to use your coding, reasoning, and native vision abilities to match it.

## Setup

You have 1 hour, 4 CPUs, 16 GB of RAM, no GPU and no internet access. Python 3.13 has numpy, scipy, pandas, scikit-learn, scikit-image, nibabel and matplotlib installed.
